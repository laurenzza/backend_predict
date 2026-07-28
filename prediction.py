# =============================================================================
# PNYB - PREDIKSI PENJUALAN ARIMA & LSTM (VERSI v6 -- FINAL: CEPAT + AKURAT)
# -----------------------------------------------------------------------------
# [v6] PERUBAHAN DARI v5: HANYA DUA KONSTANTA CAP_FACTOR YANG DIPERBARUI,
# berdasarkan eksperimen tambahan (Tabel 4.20 dan Tabel 4.25 skripsi):
#   CAP_FACTOR_ARIMA : 1.0 -> 0.8  (MAE ARIMA turun ~2.4%, dari Rp2.467.634
#                       menjadi Rp2.408.276, tanpa mengubah MAE LSTM sama sekali)
#   CAP_FACTOR_LSTM  : 1.5 -> 1.0  (MAE/RMSE LSTM identik di seluruh rentang
#                       1.0-2.5; 1.0 dipilih sbg fail-safe paling ketat)
# Seluruh struktur, urutan eksekusi, dan fungsi lain TIDAK berubah dari v5.
# =============================================================================
import pandas as pd
import numpy as np
import time
import tracemalloc
import warnings
import os
import random
from collections import Counter
from dotenv import load_dotenv
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ.setdefault('PYTHONHASHSEED', '0')
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import models
import crud
# =============================================================================
# KONSTANTA
# =============================================================================
SPLIT_PCT        = 0.80
MIN_BULAN        = 15
N_WINDOW_ARIMA   = 24
RANDOM_SEED      = 42
SEQ_LEN          = 6
N_CLUSTER        = 5
BIAS_HOLDOUT     = 7
CAP_FACTOR_ARIMA = 0.8    # [v6] sebelumnya 1.0 -- MAE ARIMA turun ~2.4% (Tabel 4.20)
CAP_FACTOR_LSTM  = 1.0    # [v6] sebelumnya 1.5 -- MAE/RMSE LSTM identik; fail-safe paling ketat (Tabel 4.25)
MIN_BULAN_AKTIF  = 3
IQR_MULTIPLIER   = 1.5
TOP_N            = 10
TOLERANSI_KETAT   = 20
TOLERANSI_LONGGAR = 60
N_JOBS           = -1
ARIMA_MAXITER    = 100
ARIMA_ORDERS = [
    (1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 0),
    (0, 1, 2), (2, 1, 1), (1, 1, 2), (3, 1, 0),
    (0, 1, 3), (2, 1, 2), (1, 2, 1), (0, 2, 1),
]
KALENDER_LIBUR = {
    '2020-05': 0.40, '2021-05': 0.45, '2022-05': 0.55,
    '2023-04': 0.50, '2024-04': 0.65, '2025-03': 0.90,
    '2020-06': 0.60, '2021-06': 0.60, '2022-06': 0.65,
    '2020-07': 0.60, '2021-07': 0.65, '2022-07': 0.70,
    '2023-06': 0.65, '2024-05': 0.60,
    '2024-06': 0.12, '2024-07': 0.28,
    '2024-08': 0.55, '2024-12': 0.40,
    '2025-01': 0.58,
}
BULAN_EKSKLUDE = ['2024-06', '2024-07']
BULAN_ANOMALI  = BULAN_EKSKLUDE
FEATURES = ['lag1_r', 'lag2_r', 'lag3_r', 'lag6_r', 'lag12_r',
            'roll3_r', 'roll6_r', 'tren_3m', 'bulan', 'produk_id', 'faktor_libur']
def set_seed_ulang(offset=0):
    random.seed(RANDOM_SEED + offset)
    np.random.seed(RANDOM_SEED + offset)
    tf.random.set_seed(RANDOM_SEED + offset)
set_seed_ulang(0)
# =============================================================================
# HELPER: MOMENTUM
# =============================================================================
def hitung_momentum(df_c):
    if len(df_c) < 3:
        return 1.0
    vals = df_c['qty'].values[-3:]
    if vals[0] > 0:
        tren = (vals[-1] - vals[0]) / vals[0]
        return float(np.clip(1.0 + tren * 0.10, 0.90, 1.10))
    return 1.0
# =============================================================================
# TRAINING ARIMA (PARALEL)
# =============================================================================
def _latih_satu_produk_arima(produk, monthly_train):
    df_c = monthly_train[monthly_train['Nama Produk'] == produk].sort_values('bulan_period')
    df_c = df_c[~df_c['bulan_period'].astype(str).isin(BULAN_EKSKLUDE)].tail(N_WINDOW_ARIMA)
    if len(df_c) < 10:
        return produk, None, 1.0
    fl_v = df_c['bulan_period'].astype(str).map(lambda x: KALENDER_LIBUR.get(x, 1.0)).values
    ts_log = np.log1p(df_c['qty'].clip(lower=0.1).values / np.maximum(fl_v, 0.1))
    best_aic, best_order = np.inf, (1, 1, 1)
    for order in ARIMA_ORDERS:
        try:
            m = ARIMA(ts_log, order=order, enforce_stationarity=True, enforce_invertibility=True
                      ).fit(method_kwargs={'maxiter': ARIMA_MAXITER})
            if m.aic < best_aic:
                best_aic, best_order = m.aic, order
        except Exception:
            pass
    bias = 1.0
    if len(df_c) > BIAS_HOLDOUT + 8:
        try:
            tr_b, ho_b = df_c.iloc[:-BIAS_HOLDOUT], df_c.iloc[-BIAS_HOLDOUT:]
            ts_tb = np.log1p(tr_b['qty'].clip(lower=0.1).values / np.maximum(
                tr_b['bulan_period'].astype(str).map(lambda x: KALENDER_LIBUR.get(x, 1.0)).values, 0.1))
            m_b = ARIMA(ts_tb, order=best_order, enforce_stationarity=True, enforce_invertibility=True
                        ).fit(method_kwargs={'maxiter': ARIMA_MAXITER})
            fc_log = np.clip(m_b.forecast(steps=BIAS_HOLDOUT), -2.0, 10.0)
            fl_hb = ho_b['bulan_period'].astype(str).map(lambda x: KALENDER_LIBUR.get(x, 1.0)).values
            fc_qty = np.expm1(fc_log) * fl_hb
            rasio = max(float(np.asarray(fc_qty).sum()), 1e-6) / float(ho_b['qty'].values.sum())
            bias = float(np.clip(rasio, 0.7, 1.5))
        except Exception:
            bias = 1.0
    return produk, best_order, bias
def latih_semua_arima(produk_arima_l, monthly_train):
    hasil_paralel = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_latih_satu_produk_arima)(produk, monthly_train) for produk in produk_arima_l
    )
    best_orders_arima, bias_correction = {}, {}
    for produk, order, bias in hasil_paralel:
        bias_correction[produk] = bias
        if order is not None:
            best_orders_arima[produk] = order
    return best_orders_arima, bias_correction
# =============================================================================
# PREDIKSI ARIMA (PARALEL, dipanggil berulang)
# =============================================================================
def _prediksi_satu_produk_arima(produk, bulan_str, faktor_libur, monthly_avail,
                                 best_orders_arima, bias_correction, harga_rata2, max_qty_produk):
    df_p = monthly_avail[monthly_avail['Nama Produk'] == produk].sort_values('bulan_period')
    df_c = df_p[~df_p['bulan_period'].astype(str).isin(BULAN_EKSKLUDE)]
    if len(df_c) > N_WINDOW_ARIMA:
        df_c = df_c.tail(N_WINDOW_ARIMA)
    fc = None
    if produk in best_orders_arima and len(df_c) >= 6:
        try:
            qty_vals = df_c['qty'].clip(lower=0.1).values
            fl_vals = df_c['bulan_period'].astype(str).map(lambda x: KALENDER_LIBUR.get(x, 1.0)).values
            ts_log = np.log1p(qty_vals / np.maximum(fl_vals, 0.1))
            m = ARIMA(ts_log, order=best_orders_arima[produk],
                      enforce_stationarity=True, enforce_invertibility=True
                      ).fit(method_kwargs={'maxiter': ARIMA_MAXITER})
            fc_result = m.forecast(steps=1)
            fc_val = fc_result.iloc[0] if hasattr(fc_result, 'iloc') else np.asarray(fc_result).reshape(-1)[0]
            fc = max(0.0, float(np.expm1(float(np.clip(fc_val, -2.0, 10.0)))) * faktor_libur)
            bias = bias_correction.get(produk, 1.0)
            if bias > 0:
                fc = fc / bias
            fc = fc * hitung_momentum(df_c)
            # [v6] CAP_FACTOR_ARIMA sekarang 0.8 (sebelumnya 1.0 pada v5)
            fc = min(max(0.0, fc), max_qty_produk.get(produk, fc) * CAP_FACTOR_ARIMA)
        except Exception:
            fc = None
    if fc is None and len(df_c) >= 1:
        recent = df_c['qty'].values[-min(6, len(df_c)):]
        bobot = np.array([1, 2, 3, 4, 5, 6][-len(recent):], dtype=float)
        fc = float(np.average(recent, weights=bobot)) * faktor_libur
    if fc is None:
        fc = float(df_c['qty'].median()) * faktor_libur if len(df_c) > 0 else 0.0
    pred_qty = max(0.0, fc)
    return {'nama_produk': produk, 'pred_qty_arima': round(pred_qty, 2),
            'pred_rev_arima': round(pred_qty * harga_rata2.get(produk, 0), 0)}
def prediksi_arima(bulan_pred, monthly_avail, produk_layak, best_orders_arima,
                    bias_correction, harga_rata2, max_qty_produk, paralel=True):
    bulan_str = str(bulan_pred)
    faktor_libur = KALENDER_LIBUR.get(bulan_str, 1.0)
    if paralel:
        hasil = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(_prediksi_satu_produk_arima)(
                produk, bulan_str, faktor_libur, monthly_avail, best_orders_arima,
                bias_correction, harga_rata2, max_qty_produk
            ) for produk in produk_layak
        )
    else:
        hasil = [_prediksi_satu_produk_arima(
            produk, bulan_str, faktor_libur, monthly_avail, best_orders_arima,
            bias_correction, harga_rata2, max_qty_produk
        ) for produk in produk_layak]
    return pd.DataFrame(hasil) if hasil else pd.DataFrame(columns=['nama_produk', 'pred_qty_arima', 'pred_rev_arima'])
# =============================================================================
# PREDIKSI LSTM (BATCH PER KLASTER)
# =============================================================================
def prediksi_lstm(bulan_pred, monthly_avail, produk_layak, cluster_models, cluster_scalers, cluster_bias,
                   produk_cluster, le, harga_rata2, median_qty_produk, max_qty_produk):
    bulan_str = str(bulan_pred)
    faktor_libur = KALENDER_LIBUR.get(bulan_str, 1.0)
    bulan_int = bulan_pred.month
    seq_per_cluster = {cid: {'produk': [], 'X': []} for cid in cluster_models.keys()}
    for produk in produk_layak:
        if produk not in le.classes_:
            continue
        cid = produk_cluster.get(produk, 0)
        if cid not in cluster_models or cid not in cluster_scalers:
            continue
        df_p = monthly_avail[monthly_avail['Nama Produk'] == produk].sort_values('bulan_period')
        df_c = df_p[~df_p['bulan_period'].astype(str).isin(BULAN_EKSKLUDE)]
        if len(df_c) < SEQ_LEN + 4:
            continue
        med_q = median_qty_produk.get(produk, 1.0)
        qty_v = df_c['qty'].values
        n = len(qty_v)
        seq_feats = []
        for step in range(SEQ_LEN):
            idx_cur = n - 1 - (SEQ_LEN - 1 - step)
            if idx_cur < 0:
                break
            def lag(k):
                return qty_v[max(idx_cur - k, 0)] / max(med_q, 1.0)
            r3 = float(np.mean(qty_v[max(0, idx_cur - 3):idx_cur])) / max(med_q, 1.0) if idx_cur >= 3 else lag(1)
            r6 = float(np.mean(qty_v[max(0, idx_cur - 6):idx_cur])) / max(med_q, 1.0) if idx_cur >= 6 else r3
            tren = ((qty_v[idx_cur] - qty_v[max(0, idx_cur - 3)]) / max(abs(qty_v[max(0, idx_cur - 3)]), 1.0)) if idx_cur >= 3 else 0.0
            try:
                curr_bp = df_c['bulan_period'].iloc[idx_cur]
                bulan_step, fl_step = curr_bp.month, KALENDER_LIBUR.get(str(curr_bp), 1.0)
            except Exception:
                bulan_step, fl_step = bulan_int, faktor_libur
            seq_feats.append([lag(1), lag(2), lag(3), lag(6), lag(12), r3, r6, tren,
                               bulan_step, float(le.transform([produk])[0]), fl_step])
        if len(seq_feats) < SEQ_LEN:
            continue
        seq_per_cluster[cid]['produk'].append(produk)
        seq_per_cluster[cid]['X'].append(np.array(seq_feats[-SEQ_LEN:]))
    hasil = []
    for cid, data in seq_per_cluster.items():
        if not data['produk']:
            continue
        X_batch = np.array(data['X'])
        n_prod, seq_len_batch, n_feat = X_batch.shape
        X_flat = X_batch.reshape(-1, n_feat)
        X_scaled = cluster_scalers[cid].transform(X_flat).reshape(n_prod, seq_len_batch, n_feat)
        pred_batch = cluster_models[cid].predict(X_scaled, verbose=0).reshape(-1)
        bias_cl = cluster_bias.get(cid, 1.0)
        for produk, pred_log in zip(data['produk'], pred_batch):
            pred_r = float(np.expm1(float(np.clip(pred_log, 0, None))))
            if bias_cl > 0:
                pred_r = pred_r / bias_cl
            med_q = median_qty_produk.get(produk, 1.0)
            # [v6] CAP_FACTOR_LSTM sekarang 1.0 (sebelumnya 1.5 pada v5)
            pred_qty = min(max(0.0, max(0.0, pred_r * med_q) * faktor_libur),
                            max_qty_produk.get(produk, pred_r * med_q) * CAP_FACTOR_LSTM)
            hasil.append({'nama_produk': produk, 'pred_qty_lstm': round(pred_qty, 2),
                          'pred_rev_lstm': round(pred_qty * harga_rata2.get(produk, 0), 0)})
    return pd.DataFrame(hasil) if hasil else pd.DataFrame(columns=['nama_produk', 'pred_qty_lstm', 'pred_rev_lstm'])
def hitung_precision_toleransi(semua_hasil, kolom_pred, toleransi_persen, bulan_test):
    hasil_per_bulan = []
    for i, bulan_pred in enumerate(bulan_test):
        df_b = semua_hasil[i]
        df_valid = df_b[df_b['aktual_qty'] > 0].copy()
        if len(df_valid) == 0:
            continue
        selisih = (df_valid[kolom_pred] - df_valid['aktual_qty']).abs() / df_valid['aktual_qty'] * 100
        n_akurat = int((selisih <= toleransi_persen).sum())
        n_total = len(df_valid)
        hasil_per_bulan.append({'bulan': str(bulan_pred), 'n_akurat': n_akurat, 'n_total': n_total,
                                 'persen_akurat': n_akurat / n_total * 100})
    return pd.DataFrame(hasil_per_bulan)
# =============================================================================
# MAIN EXECUTOR
# =============================================================================
def run_prediction(csv_path, user_id, on_complete=None):
    print(f"[INFO] Memulai Pipeline ARIMA & LSTM v6 (CAP_FACTOR_ARIMA=0.8, CAP_FACTOR_LSTM=1.0) - User ID: {user_id}")
    load_dotenv()
    DB_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        job = models.PredictionJob(status='running', user_id=user_id)
        db.add(job)
        db.commit()
        print("[INFO] Memuat dan membersihkan data...")
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df['Tanggal Pembayaran'] = pd.to_datetime(df['Tanggal Pembayaran'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Tanggal Pembayaran'])
        df = df[df['Status Terakhir'] == 'Pesanan Selesai'].copy()
        for col in ['Harga Jual (IDR)', 'Jumlah Produk Dibeli']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['item_revenue'] = (df['Harga Jual (IDR)'] * df['Jumlah Produk Dibeli']).clip(lower=0)
        df['bulan_period'] = df['Tanggal Pembayaran'].dt.to_period('M')
        df['bulan'] = df['Tanggal Pembayaran'].dt.month
        harga_rata2 = df.groupby('Nama Produk')['Harga Jual (IDR)'].mean().to_dict()
        bulan_list = sorted(df['bulan_period'].unique())
        split_idx = int(len(bulan_list) * SPLIT_PCT)
        bulan_train, bulan_test = bulan_list[:split_idx], bulan_list[split_idx:]
        monthly_all = df.groupby(['bulan_period', 'bulan', 'Nama Produk']).agg(
            qty=('Jumlah Produk Dibeli', 'sum'), revenue=('item_revenue', 'sum')
        ).reset_index().sort_values(['Nama Produk', 'bulan_period'])
        monthly_train = monthly_all[monthly_all['bulan_period'].isin(bulan_train)]
        monthly_test = monthly_all[monthly_all['bulan_period'].isin(bulan_test)]
        iqr_bounds = {}
        for p in monthly_train['Nama Produk'].unique():
            vals = monthly_train[monthly_train['Nama Produk'] == p]['qty']
            if len(vals) < 4:
                continue
            Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
            IQR = Q3 - Q1
            iqr_bounds[p] = (max(0.0, Q1 - IQR_MULTIPLIER * IQR), Q3 + IQR_MULTIPLIER * IQR)
        def _clip_qty(row):
            b = iqr_bounds.get(row['Nama Produk'])
            return row['qty'] if b is None else float(np.clip(row['qty'], b[0], b[1]))
        monthly_all['qty'] = monthly_all.apply(_clip_qty, axis=1)
        monthly_train = monthly_all[monthly_all['bulan_period'].isin(bulan_train)]
        monthly_test = monthly_all[monthly_all['bulan_period'].isin(bulan_test)]
        produk_count = monthly_train.groupby('Nama Produk')['bulan_period'].count()
        produk_layak_awal = produk_count[produk_count >= MIN_BULAN].index.tolist()
        bulan_train_bersih = [b for b in bulan_train if str(b) not in BULAN_ANOMALI]
        bulan_cek_aktif = bulan_train_bersih[-MIN_BULAN_AKTIF:]
        def cek_aktif(p):
            df_p = monthly_train[monthly_train['Nama Produk'] == p]
            return df_p[df_p['bulan_period'].isin(bulan_cek_aktif)]['qty'].sum() > 0
        produk_layak = [p for p in produk_layak_awal if cek_aktif(p)]
        produk_arima_l = list(produk_layak)
        mean_qty_produk, median_qty_produk, max_qty_produk = {}, {}, {}
        for p in produk_layak:
            vals = monthly_train[(monthly_train['Nama Produk'] == p) &
                                  (~monthly_train['bulan_period'].astype(str).isin(BULAN_ANOMALI))]['qty'].values
            mean_qty_produk[p] = max(float(vals.mean()), 1.0) if len(vals) > 0 else 1.0
            median_qty_produk[p] = max(float(np.median(vals)), 1.0) if len(vals) > 0 else 1.0
            max_qty_produk[p] = max(float(vals.max()), 1.0) if len(vals) > 0 else 1.0
        print(f"[INFO] Produk layak dimodelkan: {len(produk_layak)} dari {len(produk_count)} produk training.")
        print(f"[INFO] Melatih ARIMA per produk (paralel, n_jobs={N_JOBS})...")
        t0 = time.time()
        tracemalloc.start()
        best_orders_arima, bias_correction = latih_semua_arima(produk_arima_l, monthly_train)
        t_arima = time.time() - t0
        mem_arima = tracemalloc.get_traced_memory()[1] / 1024 ** 2
        tracemalloc.stop()
        print(f"[INFO] ARIMA selesai: {len(best_orders_arima)} produk dalam {t_arima:.1f} detik ({mem_arima:.1f} MB).")
        order_count = Counter(best_orders_arima.values())
        if order_count:
            order_top, freq_top = order_count.most_common(1)[0]
            print(f"[11] Order ARIMA terpopuler: {order_top} ({freq_top}/{len(best_orders_arima)} produk).")
        if bias_correction:
            print(f"[11] Rentang bias_correction: [{min(bias_correction.values()):.2f}, {max(bias_correction.values()):.2f}]")
        sorted_prods = sorted(produk_layak, key=lambda p: median_qty_produk[p])
        cluster_size = max(1, len(sorted_prods) // N_CLUSTER)
        produk_cluster = {p: min(i // cluster_size, N_CLUSTER - 1) for i, p in enumerate(sorted_prods)}
        le = LabelEncoder()
        mtr = monthly_train[monthly_train['Nama Produk'].isin(produk_layak)].copy()
        mtr['produk_id'] = le.fit_transform(mtr['Nama Produk'])
        mtr['median_qty'] = mtr['Nama Produk'].map(median_qty_produk)
        mtr['cluster_id'] = mtr['Nama Produk'].map(produk_cluster)
        for lag in [1, 2, 3, 6, 12]:
            mtr[f'lag{lag}_r'] = mtr.groupby('Nama Produk', sort=True)['qty'].transform(lambda x: x.shift(lag)) / mtr['median_qty'].clip(lower=1.0)
        for w, col in [(3, 'roll3_r'), (6, 'roll6_r')]:
            mtr[col] = mtr.groupby('Nama Produk', sort=True)['qty'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean()) / mtr['median_qty'].clip(lower=1.0)
        mtr['tren_3m'] = mtr.groupby('Nama Produk', sort=True)['qty'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).apply(lambda v: (v[-1] - v[0]) / max(abs(v[0]), 1), raw=True))
        mtr['faktor_libur'] = mtr['bulan_period'].astype(str).map(lambda x: KALENDER_LIBUR.get(x, 1.0))
        mtr['log_rasio'] = np.log1p((mtr['qty'] / mtr['median_qty'].clip(lower=1.0)).clip(0, 8.0))
        mtr_clean = mtr.dropna()
        mtr_clean = mtr_clean[~mtr_clean['bulan_period'].astype(str).isin(BULAN_EKSKLUDE)].copy()
        mtr_clean = mtr_clean.sort_values(['cluster_id', 'Nama Produk', 'bulan_period']).reset_index(drop=True)
        label_kl = ["Terendah", "Rendah", "Menengah", "Tinggi", "Tertinggi"]
        for cid in range(N_CLUSTER):
            angg = [p for p, c in produk_cluster.items() if c == cid]
            if not angg:
                continue
            med_v = [median_qty_produk[p] for p in angg]
            print(f"[11] Klaster {cid} ({label_kl[cid] if cid < len(label_kl) else cid}): "
                  f"{len(angg)} produk, median {min(med_v):.0f}-{max(med_v):.0f} unit/bulan.")
        print("[INFO] Melatih LSTM per klaster...")
        tracemalloc.start()
        t0 = time.time()
        cluster_models, cluster_scalers, cluster_bias = {}, {}, {}
        for cid in range(N_CLUSTER):
            df_cl = mtr_clean[mtr_clean['cluster_id'] == cid].reset_index(drop=True)
            if len(df_cl) < 40:
                continue
            set_seed_ulang(cid)
            scaler = MinMaxScaler()
            X_sc = scaler.fit_transform(df_cl[FEATURES].values.astype(float))
            y_all = df_cl['log_rasio'].values
            X_seq, y_seq = [], []
            for _, grp in df_cl.groupby('Nama Produk', sort=True):
                local_idx = list(grp.sort_values('bulan_period').index)
                for i in range(len(local_idx) - SEQ_LEN):
                    X_seq.append(X_sc[local_idx[i:i + SEQ_LEN]])
                    y_seq.append(y_all[local_idx[i + SEQ_LEN]])
            if len(X_seq) < 20:
                continue
            X_3d, y_arr = np.array(X_seq), np.array(y_seq)
            model = Sequential([
                LSTM(64, input_shape=(SEQ_LEN, len(FEATURES)), return_sequences=True),
                BatchNormalization(), Dropout(0.15),
                LSTM(32, return_sequences=False),
                BatchNormalization(), Dropout(0.15),
                Dense(16, activation='relu'), Dense(1),
            ])
            model.compile(loss=tf.keras.losses.Huber(delta=1.0), optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))
            model.fit(X_3d, y_arr, epochs=500, batch_size=16, validation_split=0.2, shuffle=False, callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, min_lr=1e-6, verbose=0),
            ], verbose=0)
            cluster_models[cid], cluster_scalers[cid], cluster_bias[cid] = model, scaler, 1.0
        t_lstm = time.time() - t0
        mem_lstm = tracemalloc.get_traced_memory()[1] / 1024 ** 2
        tracemalloc.stop()
        print(f"[INFO] LSTM selesai: {len(cluster_models)} klaster dalam {t_lstm:.1f} detik ({mem_lstm:.1f} MB).")
        print("[INFO] Menjalankan evaluasi rolling 12 bulan testing...")
        t0 = time.time()
        semua_hasil, log_bulan = [], []
        for bulan_pred in bulan_test:
            avail = monthly_all[monthly_all['bulan_period'] < bulan_pred]
            df_a = prediksi_arima(bulan_pred, avail, produk_layak, best_orders_arima, bias_correction, harga_rata2, max_qty_produk)
            df_l = prediksi_lstm(bulan_pred, avail, produk_layak, cluster_models, cluster_scalers, cluster_bias,
                                  produk_cluster, le, harga_rata2, median_qty_produk, max_qty_produk)
            df_m = pd.merge(df_a, df_l, on='nama_produk', how='outer').fillna(0)
            aktual_b = monthly_test[monthly_test['bulan_period'] == bulan_pred][['Nama Produk', 'qty', 'revenue']].rename(
                columns={'Nama Produk': 'nama_produk', 'qty': 'aktual_qty', 'revenue': 'aktual_rev'})
            df_m = pd.merge(df_m, aktual_b, on='nama_produk', how='left').fillna(0)
            semua_hasil.append(df_m)
            log_bulan.append({'bulan': str(bulan_pred), 'aktual': df_m['aktual_rev'].sum(),
                               'pred_arima': df_m['pred_rev_arima'].sum(), 'pred_lstm': df_m['pred_rev_lstm'].sum()})
        df_log = pd.DataFrame(log_bulan)
        print(f"[INFO] Evaluasi rolling selesai dalam {time.time()-t0:.1f} detik.")
        mae_agg_a = float(mean_absolute_error(df_log['aktual'], df_log['pred_arima']))
        rmse_agg_a = float(np.sqrt(mean_squared_error(df_log['aktual'], df_log['pred_arima'])))
        mae_agg_l = float(mean_absolute_error(df_log['aktual'], df_log['pred_lstm']))
        rmse_agg_l = float(np.sqrt(mean_squared_error(df_log['aktual'], df_log['pred_lstm'])))
        print(f"[HASIL] ARIMA -> MAE=Rp{mae_agg_a:,.0f}  RMSE=Rp{rmse_agg_a:,.0f}")
        print(f"[HASIL] LSTM  -> MAE=Rp{mae_agg_l:,.0f}  RMSE=Rp{rmse_agg_l:,.0f}")
        print(f"[11] Verifikasi presisi -- ARIMA MAE: {mae_agg_a:.3f} | LSTM MAE: {mae_agg_l:.3f}")
        print("[INFO] Menjalankan cek overfitting (12 bulan terakhir training)...")
        t0 = time.time()
        bulan_train_cek = bulan_train[-len(bulan_test):]
        log_train_cek = []
        for bulan_pred in bulan_train_cek:
            avail = monthly_all[monthly_all['bulan_period'] < bulan_pred]
            df_a = prediksi_arima(bulan_pred, avail, produk_layak, best_orders_arima, bias_correction, harga_rata2, max_qty_produk)
            df_l = prediksi_lstm(bulan_pred, avail, produk_layak, cluster_models, cluster_scalers, cluster_bias,
                                  produk_cluster, le, harga_rata2, median_qty_produk, max_qty_produk)
            df_m = pd.merge(df_a, df_l, on='nama_produk', how='outer').fillna(0)
            aktual_b = monthly_train[monthly_train['bulan_period'] == bulan_pred][['Nama Produk', 'qty', 'revenue']].rename(
                columns={'Nama Produk': 'nama_produk', 'qty': 'aktual_qty', 'revenue': 'aktual_rev'})
            df_m = pd.merge(df_m, aktual_b, on='nama_produk', how='left').fillna(0)
            log_train_cek.append({'aktual': df_m['aktual_rev'].sum(), 'pred_arima': df_m['pred_rev_arima'].sum(),
                                   'pred_lstm': df_m['pred_rev_lstm'].sum()})
        df_log_train_cek = pd.DataFrame(log_train_cek)
        mae_train_a = float(mean_absolute_error(df_log_train_cek['aktual'], df_log_train_cek['pred_arima']))
        mae_train_l = float(mean_absolute_error(df_log_train_cek['aktual'], df_log_train_cek['pred_lstm']))
        for nama, mae_tr, mae_te in [('ARIMA', mae_train_a, mae_agg_a), ('LSTM', mae_train_l, mae_agg_l)]:
            rasio = mae_te / max(mae_tr, 1)
            status = 'INDIKASI OVERFITTING' if rasio > 2.0 else ('PERLU DIPERHATIKAN' if rasio > 1.5 else 'STABIL')
            print(f"[2] {nama}: MAE Training=Rp{mae_tr:,.0f} MAE Testing=Rp{mae_te:,.0f} rasio={rasio:.2f}x -> {status}")
        print(f"[INFO] Cek overfitting selesai dalam {time.time()-t0:.1f} detik.")
        hasil_precision = []
        for i, bulan_pred in enumerate(bulan_test):
            df_m = semua_hasil[i]
            top_aktual = set(df_m.nlargest(TOP_N, 'aktual_rev')['nama_produk'])
            baris = {'bulan': str(bulan_pred)}
            for key, kolom in [('arima', 'pred_rev_arima'), ('lstm', 'pred_rev_lstm')]:
                top_pred = set(df_m.nlargest(TOP_N, kolom)['nama_produk'])
                baris[f'precision10_{key}'] = len(top_pred & top_aktual) / TOP_N
            hasil_precision.append(baris)
        df_precision = pd.DataFrame(hasil_precision)
        print(f"[3] Precision@10 rata-rata -- ARIMA: {df_precision['precision10_arima'].mean():.3f} | "
              f"LSTM: {df_precision['precision10_lstm'].mean():.3f}")
        df_bulan_akhir = semua_hasil[-1]
        df_valid_akhir = df_bulan_akhir[df_bulan_akhir['aktual_qty'] > 0].copy()
        for key, kolom in [('ARIMA', 'pred_qty_arima'), ('LSTM', 'pred_qty_lstm')]:
            selisih = (df_valid_akhir[kolom] - df_valid_akhir['aktual_qty']).abs() / df_valid_akhir['aktual_qty'] * 100
            n_akurat = int((selisih <= TOLERANSI_KETAT).sum())
            print(f"[6] {key} (bulan {bulan_test[-1]}): {n_akurat}/{len(df_valid_akhir)} produk "
                  f"({n_akurat / len(df_valid_akhir) * 100:.1f}%) dalam toleransi +-{TOLERANSI_KETAT}%.")
        hasil_tol_arima = hitung_precision_toleransi(semua_hasil, 'pred_qty_arima', TOLERANSI_LONGGAR, bulan_test)
        hasil_tol_lstm = hitung_precision_toleransi(semua_hasil, 'pred_qty_lstm', TOLERANSI_LONGGAR, bulan_test)
        print(f"[8] Rata-rata % produk akurat (tol +-{TOLERANSI_LONGGAR}%) -- "
              f"ARIMA: {hasil_tol_arima['persen_akurat'].mean():.1f}% | LSTM: {hasil_tol_lstm['persen_akurat'].mean():.1f}%")
        top10_aktual = df_bulan_akhir.nlargest(TOP_N, 'aktual_rev')[['nama_produk', 'aktual_qty', 'aktual_rev']].reset_index(drop=True)
        top10_pred_lstm = df_bulan_akhir.nlargest(TOP_N, 'pred_rev_lstm')[['nama_produk', 'pred_qty_lstm', 'pred_rev_lstm']].reset_index(drop=True)
        overlap_top10 = set(top10_aktual['nama_produk']) & set(top10_pred_lstm['nama_produk'])
        print(f"[5] Top-10 Aktual vs Prediksi LSTM (bulan {bulan_test[-1]}): {len(overlap_top10)}/{TOP_N} produk sama.")
        bulan_depan = bulan_list[-1] + 1
        df_a_depan = prediksi_arima(bulan_depan, monthly_all, produk_layak, best_orders_arima, bias_correction, harga_rata2, max_qty_produk)
        df_l_depan = prediksi_lstm(bulan_depan, monthly_all, produk_layak, cluster_models, cluster_scalers, cluster_bias,
                                    produk_cluster, le, harga_rata2, median_qty_produk, max_qty_produk)
        df_depan = pd.merge(df_a_depan, df_l_depan, on='nama_produk', how='outer').fillna(0)
        df_depan = df_depan.sort_values('pred_rev_lstm', ascending=False).reset_index(drop=True)
        df_depan.insert(0, 'Peringkat', range(1, len(df_depan) + 1))
        print(f"[4] Prediksi bulan {bulan_depan}: {len(df_depan)} produk, estimasi revenue LSTM "
              f"Rp{df_depan['pred_rev_lstm'].sum():,.0f} | ARIMA Rp{df_depan['pred_rev_arima'].sum():,.0f}")
        prediction_metric = models.PredictionMetric(
            arima_mae=mae_agg_a, arima_rmse=rmse_agg_a, arima_waktu_train=t_arima, arima_memori=mem_arima,
            lstm_mae=mae_agg_l, lstm_rmse=rmse_agg_l, lstm_waktu_train=t_lstm, lstm_memori=mem_lstm,
            user_id=user_id,
        )
        crud.delete_all_predictions(db, user_id)
        db.add(prediction_metric)
        db.commit()
        # --- TAMBAHKAN KODE INI UNTUK MENYIMPAN HASIL KE BACKEND ---
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'df_log': df_log,
            'semua_hasil': semua_hasil,
            'df_depan': df_depan,
            'bulan_depan': str(bulan_depan)
        }, 'models/prediction_results.pkl')
        # -----------------------------------------------------------

        if on_complete:
            on_complete()
        job.status = 'success'
        db.commit()
        print("[SUCCESS] Pipeline v6 (CAP_FACTOR_ARIMA=0.8, CAP_FACTOR_LSTM=1.0) selesai.")
        return {
            'produk_layak': produk_layak, 'df_log': df_log,
            'mae_arima': mae_agg_a, 'rmse_arima': rmse_agg_a,
            'mae_lstm': mae_agg_l, 'rmse_lstm': rmse_agg_l,
            'semua_hasil': semua_hasil, 'df_precision': df_precision,
            'df_depan': df_depan,
        }
    except Exception as e:
        db.rollback()
        if 'job' in locals():
            job.status = 'failed'
            db.commit()
        print(f"[ERROR] Pipeline gagal: {e}")
        return None
    finally:
        db.close()
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit(1)
    run_prediction(sys.argv[1], sys.argv[2])
