
def run_prediction(csv_path, user_id):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.arima.model import ARIMA
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    from scipy.stats import zscore
    import scipy.stats as stats
    import math
    import calendar

    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    import os
    from dotenv import load_dotenv

    import models

    load_dotenv()
    host=os.getenv("DB_HOST")
    port=os.getenv("DB_PORT")
    user=os.getenv("DB_USER")
    password=os.getenv("DB_PASS")
    database=os.getenv("DB_NAME")

    DB_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

    engine = create_engine(DB_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    # def load_data_from_google_sheets(sheet_url):
    #     sheet_url = sheet_url.replace('/edit?usp=sharing', '/gviz/tq?tqx=out:csv')
    #     df = pd.read_csv(sheet_url)
    #     return df

    # sheet_url = "https://docs.google.com/spreadsheets/d/1ZwOzvturDWqYXVXOGfTQVFY0Bxfv1zZ_/edit?usp=sharing&ouid=104512244025441839866&rtpof=true&sd=true"
    # df = load_data_from_google_sheets(sheet_url)
    # df = pd.read_csv("D:/Arya/Project/lala/data/tokped_tiny.csv")
    # csv_path = "D:/Arya/Project/lala/data/tokped_tiny.csv"
    db = SessionLocal()
    try:
        job = models.PredictionJob(
            status='running',
            user_id=user_id
        )

        db.add(job)
        db.commit()

        df = pd.read_csv(csv_path)

        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')

        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        duplicates = df.duplicated()

        df['Year'] = df['Tanggal Pembayaran'].dt.year
        df['Month'] = df['Tanggal Pembayaran'].dt.month
        df['Day'] = df['Tanggal Pembayaran'].dt.day

        df['Z_Score_Total_Penjualan'] = zscore(df['Total Penjualan (IDR)'])
        df['Z_Score_Harga_Jual'] = zscore(df['Harga Jual (IDR)'])

        outliers_nominal = df[df['Z_Score_Total_Penjualan'].abs() > 3]
        outliers_balance = df[df['Z_Score_Harga_Jual'].abs() > 3]

        transactions_per_year = df.groupby('Year').size()
        transactions_per_month = df.groupby(['Year', 'Month']).size()
        transactions_per_day = df.groupby('Tanggal Pembayaran').size()
        most_transacted_day = transactions_per_day.idxmax()
        least_transacted_day = transactions_per_day.idxmin()

        print(f"Hari dengan transaksi terbanyak: {most_transacted_day} ({transactions_per_day.max()} transaksi)")
        print(f"Hari dengan transaksi tersedikit: {least_transacted_day} ({transactions_per_day.min()} transaksi)")


        balance_trend = df.groupby('Tanggal Pembayaran')['Harga Jual (IDR)'].last()


        status_counts = df['Status Terakhir'].value_counts()


        status_description = {
            'Pesanan Selesai': 'Pesanan berhasil diselesaikan',
            'Dibatalkan Sistem': 'Pesanan dibatalkan oleh sistem',
            'Dibatalkan Penjual': 'Pesanan dibatalkan oleh penjual',
            'Dibatalkan Pembeli': 'Pesanan dibatalkan oleh pembeli'
        }

        status_percent = (status_counts / status_counts.sum()) * 100
        print(f"Persentase Status Transaksi:\n{status_percent}")

        total_transactions = df.groupby('Status Terakhir')['Total Penjualan (IDR)'].sum()

        average_transactions = df.groupby('Status Terakhir')['Total Penjualan (IDR)'].mean()



        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'])
        df['Month'] = df['Tanggal Pembayaran'].dt.to_period('M')

        daily_balance = df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].mean()
        print("\nRata-rata saldo harian:")
        print(daily_balance.describe())

        daily_mean_balance = daily_balance.mean()
        daily_formatted_balance = f"Rp{daily_mean_balance:,.0f}".replace(',', '.')
        print(f"Rata-rata saldo harian: {daily_formatted_balance}")

        monthly_balance = df.groupby('Month')['Total Penjualan (IDR)'].mean()
        print("Rata-rata saldo bulanan:")
        print(monthly_balance.describe())


        monthly_mean_balance = monthly_balance.mean()
        monthly_formatted_balance = f"Rp{monthly_mean_balance:,.0f}".replace(',', '.')
        print(f"Rata-rata saldo bulanan: {monthly_formatted_balance}")


        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')


        df['Month'] = df['Tanggal Pembayaran'].dt.month


        monthly_sales = df.groupby('Month')['Total Penjualan (IDR)'].sum()


        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')

        df['DayOfWeek'] = df['Tanggal Pembayaran'].dt.dayofweek  


        data_sales = df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].sum().values.reshape(-1, 1)


        scaler_sales = MinMaxScaler(feature_range=(0, 1))
        scaled_sales = scaler_sales.fit_transform(data_sales)

        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)


        time_step = 30

        X_sales, y_sales = create_dataset(scaled_sales, time_step)

        train_size = int(len(X_sales) * 0.7)  
        val_size = int(len(X_sales) * 0.15)  
        X_train_sales, X_val_sales, X_test_sales = X_sales[:train_size], X_sales[train_size:train_size+val_size], X_sales[train_size+val_size:]
        y_train_sales, y_val_sales, y_test_sales = y_sales[:train_size], y_sales[train_size:train_size+val_size], y_sales[train_size+val_size:]

        X_train_sales = X_train_sales.reshape(X_train_sales.shape[0], X_train_sales.shape[1], 1)
        X_test_sales = X_test_sales.reshape(X_test_sales.shape[0], X_test_sales.shape[1], 1)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        import tracemalloc
        import time

        tracemalloc.start()

        model = Sequential()
        model.add(Input(shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9),
            metrics=[tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.RootMeanSquaredError()]
        )

        start_lstm = time.time()
        history = model.fit(X_train_sales, y_train_sales, epochs=100, batch_size=32, verbose=1)
        end_lstm = time.time()

        current_lstm, peak_lstm = tracemalloc.get_traced_memory()

        tracemalloc.stop()


        df_arima = df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].sum()

        df_arima = df_arima.asfreq('D')

        from statsmodels.tsa.stattools import adfuller


        df_arima = df_arima.fillna(0)

        result = adfuller(df_arima)
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")

        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import matplotlib.pyplot as plt

        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error

        tracemalloc.start()
        best_score, best_cfg = float("inf"), None
        start_arima = time.time()
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        arima_model = ARIMA(df_arima[:train_size], order=(p, d, q))
                        arima_fit = arima_model.fit()
                        predictions = arima_fit.forecast(len(df_arima[train_size:train_size+val_size]))  
                        error = mean_squared_error(df_arima[train_size:train_size+val_size], predictions)
                        if error < best_score:
                            best_score, best_cfg = error, (p, d, q)
                    except:
                        continue
        end_arima = time.time()
        current_arima, peak_arima = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Best ARIMA config: {best_cfg} with error: {best_score}")

        print(arima_fit.summary())



        IS_LOG = True
        y_log = np.log1p(df_arima.clip(lower=0))
        arima_model = ARIMA(y_log, order=(1, 0, 1), trend='c')
        arima_fit = arima_model.fit()


        steps = 30
        forecast_index = pd.date_range(
            df_arima.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq=(df_arima.index.freq or 'D')
        )


        fc = arima_fit.forecast(steps=steps)
        if IS_LOG:
            fc = np.expm1(fc)


        forecast_arima = pd.Series(np.asarray(fc).reshape(-1), index=forecast_index)


        forecast_arima = forecast_arima.clip(lower=0)



        IS_LOG = True
        y_fit = np.log1p(df_arima.clip(lower=0)) if IS_LOG else df_arima

        from statsmodels.tsa.arima.model import ARIMA
        arima_model = ARIMA(y_fit, order=(1, 0, 1), trend='c')  
        arima_fit = arima_model.fit()



        steps = 30
        forecast_index = pd.date_range(df_arima.index[-1] + pd.Timedelta(days=1),
                                    periods=steps, freq=(df_arima.index.freq or 'D'))

        fc = arima_fit.forecast(steps=steps)
        if IS_LOG:
            fc = np.expm1(fc)  
        forecast_arima = pd.Series(np.asarray(fc).reshape(-1), index=forecast_index).clip(lower=0)





        df_daily_raw = (
            df.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)']
            .sum()
            .sort_index()
        )





        observed_mask = df_arima.index.isin(df_daily_raw.index)





        cut_90 = df_arima.index.max() - pd.Timedelta(days=90-1)
        ref_vals_90 = df_arima.loc[(df_arima.index >= cut_90) & observed_mask].replace(0, np.nan)
        ref_mean_90 = float(ref_vals_90.mean())





        if not np.isfinite(ref_mean_90) or ref_mean_90 == 0:
            end_idx = df_arima.index.max()
            start_180 = end_idx - pd.Timedelta(days=180-1)
            ref_vals_180 = df_arima.loc[observed_mask & (df_arima.index >= start_180)].replace(0, np.nan)
            ref_mean_90 = float(ref_vals_180.mean())

        print("[Sanity Check] Rata-rata 90 hari terakhir (aktual, observed-only):", ref_mean_90)





        ins_pred = arima_fit.get_prediction(start=0, end=len(df_arima)-1).predicted_mean
        if IS_LOG:
            ins_pred = np.expm1(ins_pred)

        ins_series = pd.Series(np.asarray(ins_pred), index=df_arima.index)
        ins_obs_90 = ins_series.loc[(df_arima.index >= cut_90) & observed_mask].replace(0, np.nan)
        ins_mean_90 = float(ins_obs_90.mean())

        print("[Sanity Check] Rata-rata in-sample 90 hari (observed-only):", ins_mean_90)


        if not np.isfinite(ins_mean_90) or ins_mean_90 == 0:
            end_idx = df_arima.index.max()
            start_180 = end_idx - pd.Timedelta(days=180-1)
            ins_obs_180 = ins_series.loc[observed_mask & (ins_series.index >= start_180)].replace(0, np.nan)
            ins_mean_90 = float(ins_obs_180.mean())






        num = np.nanmedian(
            df_arima.loc[(df_arima.index >= cut_90) & observed_mask].replace(0, np.nan).values
        )
        den = np.nanmedian(
            ins_series.loc[(ins_series.index >= cut_90) & observed_mask].replace(0, np.nan).values
        )

        raw_bias = float(num / max(den, 1e-9))

        bias = float(np.clip(raw_bias, 0.05, 20.0))


        if not globals().get("ARIMA_BIAS_APPLIED", False):
            forecast_arima = (forecast_arima * bias).clip(lower=0)
            ARIMA_BIAS_APPLIED = True

        forecast_arima_future = forecast_arima.copy()


        print("[Sanity Check] Rata-rata forecast ARIMA 30 hari (sesudah bias):", float(forecast_arima.mean()))
        try:
            print("[Sanity Check] Rata-rata forecast LSTM 30 hari:", float(np.mean(np.asarray(predicted_sales).reshape(-1))))
        except:
            pass



        history_loss = history.history['loss']
        history_mae = history.history['mean_absolute_error']
        history_rmse = history.history['root_mean_squared_error']


        test_loss, test_mae, test_rmse = model.evaluate(X_test_sales, y_test_sales, verbose=1)


        results = {
            "Test Loss": [test_loss],
            "Test MAE": [test_mae],
            "Test RMSE": [test_rmse],
        }

        results_df = pd.DataFrame(results)
        print(results_df)

        y_pred_sales = model.predict(X_test_sales)

        def predict_future_sales(model, scaler, data, time_step, days):
            last_data = data[-time_step:].reshape(1, time_step, 1)
            predicted_sales = []
            for _ in range(days):
                pred_sales = model.predict(last_data)
                predicted_sales.append(pred_sales[0, 0])
                last_data = np.append(last_data[:, 1:, :], pred_sales.reshape(1, 1, 1), axis=1)

            predicted_sales = scaler.inverse_transform(np.array(predicted_sales).reshape(-1, 1))
            return predicted_sales


        predicted_sales = predict_future_sales(model, scaler_sales, scaled_sales, time_step, days=30)


        predicted_sales_df = pd.DataFrame(predicted_sales, columns=["Predicted Sales"])
        predicted_sales_df['Day'] = predicted_sales_df.index + 1
        predicted_sales_df.set_index('Day', inplace=True)

        predicted_sales_df = pd.DataFrame(predicted_sales, columns=["Predicted Sales"])
        predicted_sales_df['Day'] = predicted_sales_df.index + 1


        predicted_sales_df['Day'] = predicted_sales_df.index + 1
        predicted_sales_df.set_index('Day', inplace=True)


        predicted_sales_df['Predicted Sales (IDR)'] = predicted_sales_df['Predicted Sales'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
        predicted_sales_df.drop(columns=['Predicted Sales'], inplace=True)


        predicted_sales_df = pd.DataFrame(predicted_sales, columns=["Predicted Sales"])
        predicted_sales_df['Day'] = predicted_sales_df.index + 1
        predicted_sales_df.set_index('Day', inplace=True)
        predicted_sales_df['Predicted Sales (IDR)'] = predicted_sales_df['Predicted Sales'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.'))
        predicted_sales_df.drop(columns=['Predicted Sales'], inplace=True)

        print(predicted_sales_df)


        from sklearn.metrics import mean_absolute_error, mean_squared_error



        pred_last30 = arima_fit.get_prediction(start=len(df_arima)-30, end=len(df_arima)-1)
        y_pred_last30 = np.clip(pred_last30.predicted_mean.values, 0, None)


        arima_mae = mean_absolute_error(df_arima[-30:], y_pred_last30)
        arima_rmse = np.sqrt(mean_squared_error(df_arima[-30:], y_pred_last30))


        print(f'ARIMA MAE: {arima_mae:.2f}')
        print(f'ARIMA RMSE: {arima_rmse:.2f}')



        print(f"Ukuran df_arima[-30:]: {df_arima[-30:].shape}")
        print(f"Ukuran forecast_arima: {len(forecast_arima)}")



        results = {
            "Test Loss": [test_loss],  
            "Test MAE": [arima_mae],  
            "Test RMSE": [arima_rmse],  
        }


        results_df = pd.DataFrame(results)
        print(results_df)



        residuals = df_arima[-30:].values - forecast_arima


        def predict_future_arima(model, data, steps):
            idx = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=steps, freq=(data.index.freq or 'D'))
            fc = model.forecast(steps=steps)
            fc = np.clip(np.asarray(fc).reshape(-1), 0, None)
            return pd.Series(fc, index=idx)





        forecast_arima_future = predict_future_arima(arima_fit, df_arima, steps=30)



        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], errors='coerce')





        produk_terjual = (
            df.loc[df['Total Penjualan (IDR)'] > 0, 'Nama Produk']
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )


        print(f"Jumlah produk terjual: {len(produk_terjual)}")
        print("Contoh 20 produk pertama:")
        print(produk_terjual[:20])










        produk_per_hari = (
            df[df['Total Penjualan (IDR)'] > 0]
            .groupby(df['Tanggal Pembayaran'].dt.date)['Nama Produk']
            .unique()
            .apply(list)
        )


        print("Produk terjual per hari (30 baris):")
        print(produk_per_hari.head(30))










        pd.Series(produk_terjual, name="Nama Produk").to_csv("models/produk_terjual.csv", index=False)
        produk_per_hari.to_csv("models/produk_terjual_per_hari.csv")








        rev_per_produk_harian = (
            df.groupby(['Tanggal Pembayaran','Nama Produk'])['Total Penjualan (IDR)']
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )





        window_hari = 60
        cutoff_date = rev_per_produk_harian.index.max() - pd.Timedelta(days=window_hari-1)
        recent_rev = rev_per_produk_harian.loc[rev_per_produk_harian.index >= cutoff_date]
        if recent_rev.empty:
            recent_rev = rev_per_produk_harian.copy()

        total_rev_recent_per_produk = recent_rev.sum(axis=0)
        epsilon = 1e-9
        total_recent_all = total_rev_recent_per_produk.sum()
        produk_share = (total_rev_recent_per_produk / (total_recent_all + epsilon)).fillna(0)





        top_k = 30
        top_produk = produk_share.sort_values(ascending=False).head(top_k).index.tolist()
        share_top = produk_share[top_produk].copy()
        share_lainnya = 1.0 - share_top.sum()
        share_top['__Lainnya__'] = max(0.0, share_lainnya)


        def allocate_per_product(total_forecast_vector, share_series):
            """Alokasikan total IDR harian → IDR per produk harian."""
            alloc = np.outer(total_forecast_vector, share_series.values)
            out_df = pd.DataFrame(alloc, columns=share_series.index)
            out_df.index = np.arange(1, len(total_forecast_vector)+1)  
            return out_df


        def top_produk_per_hari(pred_df, N=10):
            """Ambil nama Top-N produk (IDR) untuk tiap hari."""
            top_list = {}
            for day, row in pred_df.iterrows():
                row_riil = row.drop(labels='__Lainnya__', errors='ignore')
                top_list[day] = row_riil.sort_values(ascending=False).head(N).index.tolist()
            return top_list


        def top_produk_agregat_30hari(pred_df, N=20):
            """Top-N produk berdasarkan total 30 hari (IDR)."""
            cols = [c for c in pred_df.columns if c != '__Lainnya__']
            total_30 = pred_df[cols].sum(axis=0).sort_values(ascending=False)
            return total_30.head(N)


        print("[Check] Total forecast 30 hari:", float(np.sum(np.asarray(forecast_arima).reshape(-1))))
        print("[Check] Sum share_top:", float(share_top.sum()))
        if not np.isclose(share_top.sum(), 1.0):
            share_top = (share_top / max(share_top.sum(), 1e-9)).copy()





        def aggregate_to_weeks(pred_df, days_per_week=7):
            """Gabungkan prediksi harian menjadi mingguan (mean total IDR per minggu)."""
            n_weeks = len(pred_df) // days_per_week
            weekly_agg = []
            for i in range(n_weeks):
                chunk = pred_df.iloc[i * days_per_week:(i + 1) * days_per_week]
                weekly_agg.append(chunk.sum(axis=0))
            weekly_df = pd.DataFrame(weekly_agg)
            weekly_df.index = [f"Week {i+1}" for i in range(n_weeks)]
            return weekly_df

        def aggregate_to_months(pred_df, days_per_month=30):
            """Gabungkan prediksi harian menjadi bulanan."""
            n_months = len(pred_df) // days_per_month
            monthly_agg = []
            for i in range(n_months):
                chunk = pred_df.iloc[i * days_per_month:(i + 1) * days_per_month]
                monthly_agg.append(chunk.sum(axis=0))
            monthly_df = pd.DataFrame(monthly_agg)
            monthly_df.index = [f"Month {i+1}" for i in range(n_months)]
            return monthly_df


        def top_produk_per_periode(pred_df, N=10):
            """Ambil top-N produk per periode (minggu/bulan)."""
            top_dict = {}
            for idx, row in pred_df.iterrows():
                row_riil = row.drop(labels='_Lainnya_', errors='ignore')
                top_dict[idx] = row_riil.sort_values(ascending=False).head(N).index.tolist()
            return top_dict






        print("[Prediksi LSTM] ====")
        forecast_lstm_idr = np.asarray(predicted_sales).reshape(-1)
        pred_per_produk_lstm = allocate_per_product(forecast_lstm_idr, share_top)


        pred_7hari_lstm = pred_per_produk_lstm.head(7)
        top7harian_lstm = top_produk_per_hari(pred_7hari_lstm, N=5)


        weekly_lstm = aggregate_to_weeks(pred_per_produk_lstm.head(28), days_per_week=7)
        top4mingguan_lstm = top_produk_per_periode(weekly_lstm, N=5)


        monthly_lstm = aggregate_to_months(pred_per_produk_lstm.head(90), days_per_month=30)
        top3bulanan_lstm = top_produk_per_periode(monthly_lstm, N=5)


        print("\n[LSTM] Top produk harian 7 hari ke depan:")
        for day, prods in top7harian_lstm.items():
            print(f"Day {day}: {prods}")

        print("\n[LSTM] Top produk mingguan 4 minggu ke depan:")
        for week, prods in top4mingguan_lstm.items():
            print(f"{week}: {prods}")

        print("\n[LSTM] Top produk bulanan 3 bulan ke depan:")
        for month, prods in top3bulanan_lstm.items():
            print(f"{month}: {prods}")





        print("[Prediksi ARIMA] ====")
        forecast_arima_idr = np.asarray(forecast_arima_future).reshape(-1)
        pred_per_produk_arima = allocate_per_product(forecast_arima_idr, share_top)

        # ---- Harian (7 hari) ----
        pred_7hari_arima = pred_per_produk_arima.head(7)
        top7harian_arima = top_produk_per_hari(pred_7hari_arima, N=5)

        # ---- Mingguan (4 minggu) ----
        weekly_arima = aggregate_to_weeks(pred_per_produk_arima.head(28), days_per_week=7)
        top4mingguan_arima = top_produk_per_periode(weekly_arima, N=5)

        # ---- Bulanan (3 bulan) ----
        monthly_arima = aggregate_to_months(pred_per_produk_arima.head(90), days_per_month=30)
        top3bulanan_arima = top_produk_per_periode(monthly_arima, N=5)

        print("\n[ARIMA] Top produk harian 7 hari ke depan:")
        for day, prods in top7harian_arima.items():
            print(f"Day {day}: {prods}")

        print("\n[ARIMA] Top produk mingguan 4 minggu ke depan:")
        for week, prods in top4mingguan_arima.items():
            print(f"{week}: {prods}")

        print("\n[ARIMA] Top produk bulanan 3 bulan ke depan:")
        for month, prods in top3bulanan_arima.items():
            print(f"{month}: {prods}")


        import joblib
        import pandas as pd
        import numpy as np

        # 1. Ambil Tanggal Terakhir
        last_date = df['Tanggal Pembayaran'].max()

        # 2. Ambil Data 30 Hari Terakhir (untuk input LSTM prediksi masa depan)
        last_30_days_data = scaled_sales[-30:] 

        # 3. Ambil Data Share Produk (Top-K + Lainnya)
        # Pastikan Anda sudah menjalankan sel kode yang membuat 'share_top' di notebook
        product_share = share_top.to_dict()

        # 4. Ambil Data Aktual 10 Hari Terakhir (untuk endpoint compare)
        # Ambil dari df_arima (karena sudah diagregasi per hari dan diisi 0)
        df_arima = df
        df_arima['Tanggal Pembayaran'] = df_arima['Tanggal Pembayaran'].dt.date
        df_arima = df_arima.groupby('Tanggal Pembayaran')['Total Penjualan (IDR)'].sum()
        actual_data_last_10 = df_arima[-10:]
        print(actual_data_last_10)
        actual_data_dict = {
            "dates": actual_data_last_10.index,
            "values": actual_data_last_10.values.tolist()
        }

        # 5. Ambil Data Input LSTM untuk 10 Hari Terakhir tersebut
        # Kita butuh data 30 hari SEBELUM 10 hari terakhir ini untuk memprediksi ulang menggunakan LSTM
        # Posisi: dari -40 sampai -10
        lstm_input_for_compare = scaled_sales[-40:-10]

        product_prices = df.groupby('Nama Produk')['Harga Jual (IDR)'].mean().to_dict()

        # 6. Simpan Metadata Lengkap
        metadata = {
            "last_date": last_date,
            "last_sequence": last_30_days_data,
            "product_share": product_share,
            "product_prices": product_prices,
            "actual_data_last_10": actual_data_dict,
            "lstm_input_for_compare": lstm_input_for_compare
        }

        # 4. Simpan Metadata
        joblib.dump(metadata, 'models/model_metadata.pkl')

        # 5. Simpan Model
        joblib.dump(scaler_sales, 'models/scaler.pkl')
        model.save('models/lstm_model.h5')
        arima_fit.save('models/arima_model.pkl')

        print(f"Metadata disimpan. Tanggal terakhir data: {last_date}")

        result = {
            "arima_mae": arima_mae,
            "arima_rmse": arima_rmse,
            "arima_memori": round(peak_arima / (1024 ** 2), 2),
            "arima_waktu_train": end_arima-start_arima,
            "lstm_mae": test_mae,
            "lstm_rmse": test_rmse,
            "lstm_memori": round(peak_lstm / (1024 ** 2), 2),
            "lstm_waktu_train": end_lstm-start_lstm
        }

        prediction_metric = models.PredictionMetric(
            arima_mae = result["arima_mae"],
            arima_rmse = result["arima_rmse"],
            arima_waktu_train = result["arima_waktu_train"],
            arima_memori = result["arima_memori"],
            lstm_mae = result["lstm_mae"],
            lstm_rmse = result["lstm_rmse"],
            lstm_waktu_train = result["lstm_waktu_train"],
            lstm_memori = result["lstm_memori"],
            user_id = user_id,
        )

        db.add(prediction_metric)
        db.commit()
    except Exception as e:
        db.rollback()
        job.status = 'failed'
        db.commit()
        db.close()
        print("failed: ", e)
    finally:
        job.status = 'success'
        db.commit()
        db.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit(1)
    
    csv_path = sys.argv[1]
    user_id = sys.argv[2]
    run_prediction(csv_path, user_id)

# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# import os
# from dotenv import load_dotenv
# import sys

# # Get the current script's directory
# current_dir = os.path.dirname(os.path.abspath("model_terakhir.ipynb"))
# # Get the parent directory by going up one level
# parent_dir = os.path.dirname(current_dir)
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)
# import models

# load_dotenv()
# host=os.getenv("DB_HOST")
# port=os.getenv("DB_PORT")
# user=os.getenv("DB_USER")
# password=os.getenv("DB_PASS")
# database=os.getenv("DB_NAME")

# DB_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

# engine = create_engine(DB_URL, pool_pre_ping=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()


# db = SessionLocal()


# try:
#     df = pd.read_csv(csv_path)
# except Exception as e:
#     print(f"Failed reading file: {str(e)}")

# df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("[()]", "", regex=True)
# df = df.replace({np.nan: None, pd.NaT: None})
# df = df.dropna(how="all")

# df["tanggal_pembayaran"] = pd.to_datetime(df["tanggal_pembayaran"], errors="coerce")
# df["user_id"] = 1

# try:
#     db.query(models.Sale).delete()
#     records = df.to_dict(orient="records")
#     db.bulk_insert_mappings(models.Sale, records)
#     db.commit()
# except Exception as e:
#     db.rollback()
#     print(f"Database insertion failed: {str(e)}")


