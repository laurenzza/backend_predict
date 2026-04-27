from fastapi import FastAPI, Depends, HTTPException, UploadFile, Form, File, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import pandas as pd
import numpy as np
from io import StringIO
import base64
from sqlalchemy.orm import Session
from database import get_db
from typing import Dict, List

import bcrypt
from jose import jwt, JWTError

import os
from dotenv import load_dotenv
import schemas, crud, models
from prediction import run_prediction

from datetime import datetime, timedelta, timezone

from statsmodels.tsa.arima.model import ARIMAResults
import tensorflow as tf
import joblib

app = FastAPI(title="Sales Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:8080", "http://localhost:8081"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# LOAD ENV VARIABLES
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
SALT = os.getenv("SALT").encode('utf-8')
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# --- Global Variables ---
lstm_model = None
arima_model = None
scaler = None
metadata = None

# Konfigurasi
TIME_STEP = 30
IS_LOG_ARIMA = True

@app.on_event("startup")
def load_artifacts():
    global lstm_model, arima_model, scaler, metadata
    
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    arima_model = ARIMAResults.load('models/arima_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    
    # print(f"Artifacts loaded. Training data ended on: {metadata['last_date']}")

# ROOT
@app.get("/api/", tags=["Root"])
async def index():
    return {"message": "Sales Predictor BE"}

# USER
@app.get("/api/user/me", tags=["User"], response_model=schemas.UserResponse)
async def read_user_me(conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    user = crud.get_user_by_email(conn, payload["email"])

    return schemas.UserResponse(
        user_id=user.user_id,
        email=user.email,
        nama_lengkap=user.nama_lengkap,
        nama_toko=user.nama_toko,
        role=user.role,
        csv_path=user.csv_path or ""
    )

@app.post("/api/user/register", tags=["User"], response_model=schemas.UserResponse)
async def create_user(user: schemas.UserCreate, conn: Session = Depends(get_db)):
    if crud.get_user_by_email(conn, user.email):
        raise HTTPException(409, "Email sudah digunakan")

    return crud.insert_user(conn, user)

@app.put("/api/user/edit_profile", tags=["User"], response_model=schemas.UserResponse)
async def edit_profile(user: schemas.UserEdit, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user.user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")

    edited_user = crud.update_user(conn, db_user, user)

    return schemas.UserResponse(
        user_id=edited_user.user_id,
        email=edited_user.email,
        nama_lengkap=edited_user.nama_lengkap,
        nama_toko=edited_user.nama_toko,
        role=edited_user.role,
        csv_path=edited_user.csv_path or ""
    )

@app.put("/api/user/change_password", tags=["User"], response_model=schemas.UserResponse)
async def change_password(user: schemas.UserChangePassword, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user.user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")
    
    if db_user.password != hash_password(user.password):
        return HTTPException(401, "Password tidak cocok")

    edited_user = crud.update_password(conn, db_user, hash_password(user.new_password))

    return schemas.UserResponse(
        user_id=edited_user.user_id,
        email=edited_user.email,
        nama_lengkap=edited_user.nama_lengkap,
        nama_toko=edited_user.nama_toko,
        role=edited_user.role,
        csv_path=edited_user.csv_path or ""
    )

@app.delete("/api/user/{user_id}/delete", tags=["User"])
async def delete_account(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    db_user = crud.get_user_by_id(conn, user_id)

    if not db_user:
        raise HTTPException(400, "User tidak ditemukan")
    
    result = crud.delete_user(conn, db_user)

    return result

def hash_path(path: str) -> str:
    byte_path = path.encode('utf-8')
    hashed = bcrypt.hashpw(byte_path, bcrypt.gensalt())
    safe_hash = base64.urlsafe_b64encode(hashed).decode('utf-8').rstrip("=")
    return safe_hash

# DATASET
# @app.post("/api/sales/upload", tags=["Dataset"], response_model=schemas.UserResponse)
# async def upload_sales(file: UploadFile = File(...), conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
#     payload = verify_token(token)
#     user = crud.get_user_by_email(conn, payload["email"])
#     prev_csv = os.path.join(UPLOAD_DIR, user.csv_path)

#     if not file.filename.endswith(".csv"):
#         raise HTTPException(400, detail="File format not supported")
    
#     os.makedirs(UPLOAD_DIR, exist_ok=True)

#     try:
#         csv_path = hash_path(f"user_id{user.user_id}_{file.filename}")
#         csv_path += ".csv"

#         with open(os.path.join(UPLOAD_DIR, csv_path), "wb") as f:
#             contents = await file.read()
#             f.write(contents)

#         user.csv_path = csv_path

#         conn.commit()
#         conn.refresh(user)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

#     try:
#         df = pd.read_csv(StringIO(contents.decode("utf-8")))
#     except Exception as e:
#         raise HTTPException(400, detail=f"Failed reading file: {str(e)}")
    
#     df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("[()]", "", regex=True)
#     df = df.replace({np.nan: None, pd.NaT: None})
#     df = df.dropna(how="all")

#     df["tanggal_pembayaran"] = pd.to_datetime(df["tanggal_pembayaran"], errors="coerce")
#     df["user_id"] = user.user_id

#     try:
#         crud.delete_sales(conn, user.user_id)
#         records = df.to_dict(orient="records")
#         conn.bulk_insert_mappings(models.Sale, records)
#         conn.commit()
#         if prev_csv != f"{UPLOAD_DIR}None" and prev_csv != UPLOAD_DIR and os.path.exists(prev_csv):
#             os.remove(prev_csv)
#     except Exception as e:
#         conn.rollback()
#         raise HTTPException(500, detail=f"Database insertion failed: {str(e)}")
    
#     return schemas.UserResponse(
#         user_id=user.user_id,
#         email=user.email,
#         nama_lengkap=user.nama_lengkap,
#         nama_toko=user.nama_toko,
#         role=user.role,
#         csv_path=user.csv_path
#     )

@app.post("/api/sales/upload", tags=["Dataset"], response_model=schemas.UserResponse)
async def upload_sales(
    file: UploadFile = File(...), 
    mode: str = Form(...),  # <--- TAMBAHAN: Menerima input mode ('replace' atau 'append')
    conn: Session = Depends(get_db), 
    token: str = Depends(oauth2_scheme)
):
    payload = verify_token(token)
    user = crud.get_user_by_email(conn, payload["email"])
    
    # Simpan path file lama
    prev_csv_path = None
    if user.csv_path:
        prev_csv_path = os.path.join(UPLOAD_DIR, user.csv_path)

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="File format not supported")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to read uploaded file: {str(e)}")

    # LOGIKA UTAMA (Sesuai Mode yang dipilih di Frontend)
    try:
        # Cek apakah file lama ada
        file_exists = prev_csv_path and os.path.exists(prev_csv_path)

        # KONDISI APPEND:
        # Hanya append jika mode 'append' DAN file fisiknya ada
        if mode == 'append' and file_exists:
            # --- MODE APPEND ---
            full_path = prev_csv_path
            
            try:
                # Cari baris baru pertama (akhir header) untuk membuang header data baru
                header_end_index = contents.index(b'\n')
                data_to_append = contents[header_end_index + 1:]
                
                if data_to_append:
                    with open(full_path, "ab") as f:
                        f.write(data_to_append)
            except ValueError:
                pass # Error jika file kosong/cuma header

        else:
            # --- MODE REPLACE / CREATE NEW ---
            # Jika mode 'replace' ATAU file belum ada -> Kita timpa/buat baru
            
            # Jika user minta replace tapi pakai nama file yang sama, kita generate hash baru biar bersih
            # atau overwrite hash lama juga boleh. Di sini kita buat hash baru.
            new_csv_name = hash_path(f"user_id{user.user_id}_{file.filename}") + ".csv"
            full_path = os.path.join(UPLOAD_DIR, new_csv_name)

            with open(full_path, "wb") as f:
                f.write(contents)
            
            # Update path user
            user.csv_path = new_csv_name
            
            # Jika ini REPLACE, kita harus pastikan Database juga bersih?
            # Note: Kode di bawah hanya insert. Jika mode 'replace', 
            # idealnya Anda menghapus data lama di DB juga sebelum insert baru.
            if mode == 'replace':
                # HAPUS DATA LAMA DI DB UNTUK USER INI (Opsional, tapi disarankan untuk 'Rewrite')
                crud.delete_sales(conn, user.user_id)
                
                # Jika file fisik lama berbeda nama dengan yang baru, hapus file lama
                if file_exists and prev_csv_path != full_path:
                    os.remove(prev_csv_path)

        conn.commit()
        conn.refresh(user)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # LOGIKA DATABASE INSERTION
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, detail=f"Failed reading file content: {str(e)}")
    
    # Cleaning Data
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("[()]", "", regex=True)
    df = df.replace({np.nan: None, pd.NaT: None})
    df = df.dropna(how="all")

    df["tanggal_pembayaran"] = pd.to_datetime(df["tanggal_pembayaran"], errors="coerce")
    df["user_id"] = user.user_id

    try:
        records = df.to_dict(orient="records")
        conn.bulk_insert_mappings(models.Sale, records)
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise HTTPException(500, detail=f"Database insertion failed: {str(e)}")
    
    return schemas.UserResponse(
        user_id=user.user_id,
        email=user.email,
        nama_lengkap=user.nama_lengkap,
        nama_toko=user.nama_toko,
        role=user.role,
        csv_path=user.csv_path
    )

@app.get("/api/sales/summary/{user_id}", tags=["Dataset"])
async def summarize_sales(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    sales = crud.get_sales_summary(conn, user_id)
    
    if sales == None:
        return None

    return schemas.DataSummaryResponse(
        total_transaksi=sales.total_transaksi,
        total_produk=sales.total_produk,
        periode_awal=sales.periode_awal,
        periode_akhir=sales.periode_akhir,
        total_status_selesai=sales.total_status_selesai,
        total_status_dibatalkan=sales.total_status_dibatalkan,
        total_status_dibatalkan_pembeli=sales.total_status_dibatalkan_pembeli,
        total_status_dibatalkan_penjual=sales.total_status_dibatalkan_penjual,
        total_status_dibatalkan_sistem=sales.total_status_dibatalkan_sistem,
        total_status_sedang_dikirim=sales.total_status_sedang_dikirim,
    )

# @app.get("/api/sales/average/{user_id}", tags=["Dataset"])
# async def average_sales(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
#     payload = verify_token(token)

#     sales = crud.get_sales_summary(conn, user_id)
    
#     if sales == None:
#         return None
    
#     total_idr = crud.get_total_sales_idr(conn, user_id)
#     span_days = sales.periode_akhir - sales.periode_awal
#     print(span_days.days)


#     daily = {
#         "avg_penjualan_idr_harian": total_idr/span_days,
#         "avg_transaksi_harian": sales.total_transaksi/span_days
#     }

#     return schemas.DataSummaryResponse(
#         total_transaksi=sales.total_transaksi,
#         total_produk=sales.total_produk,
#         periode_awal=sales.periode_awal,
#         periode_akhir=sales.periode_akhir,
#         total_status_selesai=sales.total_status_selesai,
#         total_status_dibatalkan=sales.total_status_dibatalkan,
#         total_status_dibatalkan_pembeli=sales.total_status_dibatalkan_pembeli,
#         total_status_dibatalkan_penjual=sales.total_status_dibatalkan_penjual,
#         total_status_dibatalkan_sistem=sales.total_status_dibatalkan_sistem,
#         total_status_sedang_dikirim=sales.total_status_sedang_dikirim,
#     )

@app.get("/api/sales/{user_id}", tags=["Dataset"])
async def my_sales(user_id: int, limit: int = 10, offset: int = 0, year: str = "all", status: str = "all", conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    sales_response = crud.get_sales(conn, user_id, limit, offset, year, status)

    sales: schemas.SalesDataResponse = []

    for row in sales_response:
        sales.append(schemas.SalesDataResponse(
            sale_id=row.sale_id,
            invoice=row.invoice,
            tanggal_pembayaran=row.tanggal_pembayaran,
            status_terakhir=row.status_terakhir,
            nama_produk=row.nama_produk,
            jumlah_produk_dibeli=row.jumlah_produk_dibeli,
            harga_jual_idr=row.harga_jual_idr,
            total_penjualan_idr=row.total_penjualan_idr
        ))

    rows_count = crud.count_fetch_filter_sales(conn, user_id, year, status)

    return {
        "dataset": sales,
        "rows": rows_count.rows
    }

@app.get("/api/sales/products/{user_id}", tags=["Dataset"], response_model=list[schemas.ProductResponse])
async def my_products(user_id: int, limit: int = 10, offset: int = 0, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_products(conn, user_id, limit, offset)

    products: schemas.ProductResponse = []

    number = offset + 1
    for row in result:
        products.append(schemas.ProductResponse(
            nomor_produk=number,
            nama_produk=row.nama_produk,
            total_transaksi=row.total_transaksi,
            total_penjualan=row.total_penjualan
        ))
        number += 1
    
    return products

# TOP PRODUCTS
@app.get("/api/top_products/{user_id}", tags=["Top Products"], response_model=list[schemas.TopProductsResponse])
async def my_top_products(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_top_products(conn, user_id)

    top_products: schemas.TopProductsResponse = []

    for row in result:
        top_products.append(schemas.TopProductsResponse(
            nama_produk=row.nama_produk,
            total_unit_terjual=row.total_unit_terjual,
            total_transaksi=row.total_transaksi,
            total_penjualan=row.total_penjualan,
        ))

    return top_products

@app.get("/api/top_products/summary/{user_id}", tags=["Top Products"], response_model=schemas.TopProductsSummaryResponse)
async def my_top_products_summary(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_top_products_summary(conn, user_id)

    return schemas.TopProductsSummaryResponse(
        total_penjualan_top=result.total_penjualan_top,
        total_unit_terjual_top=result.total_unit_terjual_top
    )

# STATISTICS
@app.get("/api/statistics/monthly_trend/{user_id}", tags=["Statistics"], response_model=list[schemas.SalesTrendResponse])
async def my_monthly_trend(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = list(crud.get_monthly_sales_trend(conn, user_id))

    trends: schemas.SalesTrendResponse = []

    is_first = True
    prev_total_penjualan = -1
    for row in reversed(result):
        difference = ((row.total_penjualan-prev_total_penjualan)/prev_total_penjualan)*100
        prev_total_penjualan = row.total_penjualan

        if is_first:
            is_first = False
            continue

        trends.append(schemas.SalesTrendResponse(
            bulan_pembayaran=row.bulan_pembayaran,
            tahun_pembayaran=row.tahun_pembayaran,
            total_penjualan=row.total_penjualan,
            pertumbuhan=difference
        ))

    return trends

@app.get("/api/statistics/transaction_analysis/{user_id}", tags=["Statistics"], response_model=schemas.TransactionAnalysisResponse)
async def my_transaction_analysis(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    result = crud.get_transaction_analysis(conn, user_id)

    return schemas.TransactionAnalysisResponse(
        avg_penjualan=result.avg_penjualan,
        max_penjualan=result.max_penjualan,
        min_penjualan=result.min_penjualan,
        median_penjualan=result.median_penjualan,
        std_penjualan=result.std_penjualan
    )

@app.get("/api/statistics/temporal_pattern/{user_id}", tags=["Statistics"], response_model=schemas.TemporalPatternResponse)
async def my_temporal_pattern(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    day = crud.get_temporal_day(conn, user_id)
    month = crud.get_temporal_month(conn, user_id)
    time_range = crud.get_temporal_time_range(conn, user_id)

    return schemas.TemporalPatternResponse(
        hari_transaksi=day.hari_transaksi,
        jumlah_transaksi_hari=day.jumlah_transaksi_hari,
        bulan_transaksi=month.bulan_transaksi,
        jumlah_transaksi_bulan=month.jumlah_transaksi_bulan,
        rentang_jam_transaksi=time_range.rentang_jam_transaksi,
        jumlah_transaksi_jam=time_range.jumlah_transaksi_jam,
    )

# PREDICTIONS
@app.post("/api/predictions/predict", tags=["Predictions"])
async def predict(args: schemas.PredictionArgs, background_tasks: BackgroundTasks, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    crud.delete_prediction_job(conn, args.user_id)
    crud.delete_all_predictions(conn, args.user_id)
    conn.commit()

    csv_path = f"uploads/sales/{args.csv_path}"
    background_tasks.add_task(run_prediction, csv_path, args.user_id)

    return {"status": "file executed successfully", "file": csv_path}

def get_job(user_id: int, conn: Session = Depends(get_db)):
    result = crud.get_prediction_job(conn, user_id)

    if result == None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction job not found."
        )

    return result

@app.get("/api/predictions/status/{user_id}", tags=["Predictions"])
async def my_prediction_status(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)
    return get_job(user_id, conn)

@app.get("/api/predictions/metrics/{user_id}", tags=["Predictions"])
async def my_prediction_metrics(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": ""
        }

    result = crud.get_prediction_metrics(conn, user_id)

    metrics = schemas.PredictionMetricsResponse(
        prediction_metric_id=result.prediction_metric_id,
        arima_mae=result.arima_mae,
        arima_rmse=result.arima_rmse,
        arima_waktu_train=result.arima_waktu_train,
        arima_memori=result.arima_memori,
        lstm_mae=result.lstm_mae,
        lstm_rmse=result.lstm_rmse,
        lstm_waktu_train=result.lstm_waktu_train,
        lstm_memori=result.lstm_memori,
    )

    return {
        "job_status": job.status,
        "data": metrics
    }

@app.get("/api/predictions/comparisons/{user_id}", tags=["Predictions"])
async def my_prediction_comparisons(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_prediction_comparisons(conn, user_id)

    prediction_comparisons = []

    for row in result:
        prediction_comparisons.append(schemas.PredictionComparisonsResponse(
            prediction_comparison_id=row.prediction_comparison_id,
            hari=row.hari,
            hasil_total_penjualan_aktual=row.hasil_total_penjualan_aktual,
            hasil_total_penjualan_arima=row.hasil_total_penjualan_arima,
            hasil_total_penjualan_lstm=row.hasil_total_penjualan_lstm
        ))

    return {
        "job_status": job.status,
        "data": prediction_comparisons
    }

@app.get("/api/predictions/total/{user_id}", tags=["Predictions"])
async def my_total_predictions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_total_predictions(conn, user_id)

    total_predictions = []

    for row in result:
        total_predictions.append(schemas.TotalPredictionResponse(
            total_prediction_id=row.total_prediction_id,
            hasil_tanggal=row.hasil_tanggal,
            hasil_total_penjualan_arima=row.hasil_total_penjualan_arima,
            hasil_total_penjualan_lstm=row.hasil_total_penjualan_lstm
        ))

    return {
        "job_status": job.status,
        "data": total_predictions
    }

@app.get("/api/predictions/daily/{user_id}", tags=["Predictions"])
async def my_daily_predicions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_daily_predictions(conn, user_id)

    products = []

    for row in result:
        products.append(schemas.DailyPredictionsResponse(
            daily_product_prediction_id=row.daily_product_prediction_id,
            hari=row.hari,
            hasil_nama_produk_arima=row.hasil_nama_produk_arima,
            hasil_nama_produk_lstm=row.hasil_nama_produk_lstm,
        ))

    return {
        "job_status": job.status,
        "data": products
    }

@app.get("/api/predictions/weekly/{user_id}", tags=["Predictions"])
async def my_weekly_predicions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_weekly_predictions(conn, user_id)

    products = []

    for row in result:
        products.append(schemas.WeeklyPredictionsResponse(
            weekly_product_prediction_id=row.weekly_product_prediction_id,
            minggu=row.minggu,
            hasil_nama_produk_arima=row.hasil_nama_produk_arima,
            hasil_nama_produk_lstm=row.hasil_nama_produk_lstm,
        ))

    return {
        "job_status": job.status,
        "data": products
    }

@app.get("/api/predictions/monthly/{user_id}", tags=["Predictions"])
async def my_monthly_predicions(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme), job = Depends(get_job)):
    payload = verify_token(token)

    if job.status == "running" or job.status == "failed":
        return {
            "job_status": job.status,
            "data": []
        }

    result = crud.get_monthly_predictions(conn, user_id)

    products = []

    for row in result:
        products.append(schemas.MonthlyPredictionsResponse(
            monthly_product_prediction_id=row.monthly_product_prediction_id,
            bulan=row.bulan,
            hasil_nama_produk_arima=row.hasil_nama_produk_arima,
            hasil_nama_produk_lstm=row.hasil_nama_produk_lstm,
        ))

    return {
        "job_status": job.status,
        "data": products
    }

# --- Helper Functions ---
def predict_lstm_recursive(days_ahead, initial_sequence):
    """
    Melakukan prediksi berulang (recursive) untuk LSTM
    """
    current_batch = initial_sequence.reshape(1, TIME_STEP, 1)
    predicted_val = 0
    
    for i in range(days_ahead):
        pred = lstm_model.predict(current_batch, verbose=0)
        predicted_val = pred[0][0]
        current_batch = np.append(current_batch[:, 1:, :], [[pred[0]]], axis=1)
        
    real_prediction = scaler.inverse_transform([[predicted_val]])
    return float(real_prediction[0][0])

def format_idr(value):
    return f"Rp{value:,.0f}".replace(",", ".")

# ==========================================
# ENDPOINT 1: Predict Single Day + Top Products
# ==========================================
@app.post("/api/predict/single_day")
def predict_single_day_products(payload: schemas.DateInput, token: str = Depends(oauth2_scheme)):
    # payload = verify_token(token)
    """
    Prediksi total penjualan pada 1 tanggal spesifik & 
    Top 5 Produk (IDR) + Estimasi Qty.
    """
    try:
        if not metadata:
            raise HTTPException(status_code=503, detail="Model belum siap (metadata not loaded).")

        last_train_date = pd.to_datetime(metadata['last_date']).date()
        target_date = payload.target_date
        
        # Validasi tanggal
        delta = (target_date - last_train_date).days
        if delta < 1:
             raise HTTPException(status_code=400, detail=f"Tanggal target harus setelah {last_train_date}.")

        # 1. Prediksi Total Penjualan (LSTM Recursive)
        lstm_total = predict_lstm_recursive(delta, metadata['last_sequence'])
        lstm_total = max(0.0, lstm_total) 

        # 2. Ambil Data dari Metadata
        product_share: Dict = metadata.get('product_share', {})
        product_prices: Dict = metadata.get('product_prices', {}) # <--- Load Harga Rata-rata
        
        # 3. Hitung Estimasi Per Produk
        all_product_allocations = []
        
        for product_name, share in product_share.items():
            if product_name == "__Lainnya__": 
                continue 
                
            # Hitung Omzet (IDR)
            estimated_sales = lstm_total * share
            
            # Hitung Qty (Jumlah)
            # Ambil harga satuan, default 0 jika error
            unit_price = product_prices.get(product_name, 0)
            # print(f"{product_name} {unit_price}")
            
            estimated_qty = 0
            if unit_price > 0:
                # Rumus: Omzet dibagi Harga Satuan
                # estimated_qty = int(round(estimated_sales / unit_price))
                estimated_qty = round(estimated_sales / unit_price)
                estimated_qty = max(estimated_qty, 1)
            
            all_product_allocations.append({
                "product_name": product_name,
                "estimated_sales": estimated_sales,
                "estimated_qty": estimated_qty, # Simpan qty sementara
                "share_percent": share * 100
            })
        
        # 4. Sortir (Tertinggi ke Terendah)
        all_product_allocations.sort(key=lambda x: x['estimated_sales'], reverse=True)
        
        # 5. Ambil Top 5
        top_5_products = all_product_allocations
        # top_5_products = all_product_allocations[:5]

        # 6. Format Output (JANGAN UBAH KEY LAMA)
        formatted_top_5 = []
        for item in top_5_products:
            formatted_top_5.append({
                # --- VARIABEL LAMA (JANGAN DIUBAH) ---
                "product_name": item['product_name'],
                "estimated_sales_idr": format_idr(item['estimated_sales']),
                "raw_value": item['estimated_sales'],
                
                # --- VARIABEL BARU (TAMBAHAN) ---
                "estimated_qty": item['estimated_qty'] 
            })
        
        return {
            "date": target_date,
            "days_ahead": delta,
            "prediction_summary": {
                "total_sales_forecast": format_idr(lstm_total),
                "model_used": "LSTM Recursive"
            },
            "top_5_products": formatted_top_5
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/api/predict/single_day")
# def predict_single_day_products(payload: schemas.DateInput, token: str = Depends(oauth2_scheme)):
#     # payload = verify_token(token)
#     """
#     Prediksi total penjualan pada 1 tanggal spesifik & 
#     secara otomatis menentukan Top 5 Produk terlaris hari itu.
#     """
#     try:
#         if not metadata:
#             raise HTTPException(status_code=503, detail="Model belum siap (metadata not loaded).")

#         last_train_date = pd.to_datetime(metadata['last_date']).date()
#         target_date = payload.target_date
        
#         # Hitung selisih hari
#         delta = (target_date - last_train_date).days
        
#         if delta < 1:
#              raise HTTPException(status_code=400, detail=f"Tanggal target harus setelah {last_train_date}.")

#         # 1. Prediksi Total Penjualan (LSTM Recursive)
#         # Menggunakan sequence terakhir dari metadata sebagai start point
#         lstm_total = predict_lstm_recursive(delta, metadata['last_sequence'])
#         lstm_total = max(0.0, lstm_total) # Pastikan tidak negatif

#         # 2. Ambil Share Produk dari Metadata
#         # Dictionary ini berisi { "Nama Produk": 0.05, ... }
#         product_share: Dict = metadata.get('product_share', {})
        
#         # 3. Hitung Estimasi Penjualan untuk SETIAP produk
#         all_product_allocations = []
        
#         for product_name, share in product_share.items():
#             # Opsional: Skip kategori 'Lainnya' jika hanya ingin produk spesifik
#             if product_name == "__Lainnya__": 
#                 continue 
                
#             # Rumus: Total Prediksi Hari Itu * Persentase Share Produk
#             estimated_sales = lstm_total * share
            
#             all_product_allocations.append({
#                 "product_name": product_name,
#                 "estimated_sales": estimated_sales,
#                 "share_percent": share * 100
#             })
        
#         # 4. Sortir dari Penjualan Tertinggi ke Terendah
#         all_product_allocations.sort(key=lambda x: x['estimated_sales'], reverse=True)
        
#         # 5. Ambil Top 5
#         top_5_products = all_product_allocations[:5]

#         # 6. Format Output
#         formatted_top_5 = []
#         for item in top_5_products:
#             formatted_top_5.append({
#                 "product_name": item['product_name'],
#                 "estimated_sales_idr": format_idr(item['estimated_sales']),
#                 "raw_value": item['estimated_sales']
#             })
        
#         return {
#             "date": target_date,
#             "days_ahead": delta,
#             "prediction_summary": {
#                 "total_sales_forecast": format_idr(lstm_total),
#                 "model_used": "LSTM Recursive"
#             },
#             "top_5_products": formatted_top_5
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINT 2: Compare Last 10 Days (Actual vs Predicted)
# ==========================================
@app.get("/api/compare/last_10_days")
def compare_models_accuracy():
    """
    Membandingkan data aktual 10 hari terakhir (dari dataset training/testing)
    dengan hasil prediksi ARIMA dan LSTM pada periode yang sama.
    Berguna untuk evaluasi performa model secara real-time.
    """
    try:
        actual_data = metadata['actual_data_last_10']
        dates = actual_data['dates']
        actual_values = actual_data['values']
        
        results = []
        
        # Data awal untuk LSTM (30 hari sebelum periode 10 hari ini)
        current_lstm_batch = metadata['lstm_input_for_compare'].reshape(1, TIME_STEP, 1)

        # ARIMA Forecast (In-Sample / Out-of-Sample)
        # Kita perlu trick sedikit: ARIMA model di statsmodels menyimpan history.
        # Kita bisa memanggil predict() pada index yang sesuai.
        # Namun, cara paling aman dan simpel untuk API ini adalah melakukan forecast 
        # seolah-olah data berakhir sebelum 10 hari ini (jika model belum ditraining ulang).
        # TAPI, karena model 'arima_sales_model.pkl' adalah hasil fit PNUH,
        # maka 'predict' pada 10 hari terakhir adalah in-sample prediction.
        
        # Ambil panjang data total yang diketahui model
        n_obs = len(arima_model.fittedvalues)
        # Prediksi 10 hari terakhir (index n_obs-10 sampai n_obs-1)
        arima_preds_log = arima_model.predict(start=n_obs-10, end=n_obs-1)
        
        if IS_LOG_ARIMA:
            arima_preds = np.expm1(arima_preds_log)
        else:
            arima_preds = arima_preds_log

        # Loop 10 hari
        for i in range(10):
            # 1. Actual
            act_val = actual_values[i]
            
            # 2. ARIMA Prediction
            # Perlu handling jika arima_preds berupa series/array
            arima_val = float(arima_preds.iloc[i]) if hasattr(arima_preds, 'iloc') else float(arima_preds[i])
            arima_val = max(0.0, arima_val)

            # 3. LSTM Prediction (Recursive)
            # Prediksi hari ini
            lstm_pred_scaled = lstm_model.predict(current_lstm_batch, verbose=0)[0][0]
            
            # Kembalikan ke real value
            lstm_val = float(scaler.inverse_transform([[lstm_pred_scaled]])[0][0])
            lstm_val = max(0.0, lstm_val)

            # Update batch LSTM: 
            # PENTING: Untuk perbandingan yang fair, biasanya kita punya 2 metode:
            # a) One-step-ahead: Update batch menggunakan data AKTUAL hari ini (Teacher Forcing).
            # b) Multi-step: Update batch menggunakan data PREDIKSI hari ini.
            # Di sini kita gunakan (b) Multi-step agar konsisten dengan cara prediksi masa depan.
            current_lstm_batch = np.append(current_lstm_batch[:, 1:, :], [[[lstm_pred_scaled]]], axis=1)

            # Hitung Error (Absolute Percentage Error - APE) untuk hari ini
            # Hindari pembagian dengan 0
            safe_act = act_val if act_val > 0 else 1
            arima_ape = abs(act_val - arima_val) / safe_act * 100
            lstm_ape = abs(act_val - lstm_val) / safe_act * 100

            results.append({
                "date": dates[i],
                "day": i+1,
                "actual": act_val,
                "arima_pred": arima_val,
                "lstm_pred": lstm_val,
                "diff_arima": act_val - arima_val,
                "diff_lstm": act_val - lstm_val,
                # "error_percent_arima": f"{arima_ape:.2f}%",
                # "error_percent_lstm": f"{lstm_ape:.2f}%"
            })

        return {
            "description": "Perbandingan Data Aktual vs Prediksi Model (10 Hari Terakhir)",
            "data": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictions/predict/seven_days")
def predict_sales_7_days(payload: schemas.DateInput):
    """
    Prediksi penjualan selama 7 hari berturut-turut dimulai dari tanggal target.
    """
    try:
        # 1. Hitung selisih hari dari data training terakhir
        last_train_date = pd.to_datetime(metadata['last_date']).date()
        target_date = payload.target_date
        
        # Delta adalah jarak dari data terakhir ke hari pertama prediksi
        delta = (target_date - last_train_date).days
        
        if delta < 1:
            raise HTTPException(
                status_code=400, 
                detail=f"Tanggal target harus setelah {last_train_date}."
            )
        
        # Berapa total langkah yang harus diprediksi?
        # Jika delta=1 (besok), kita butuh 10 hari (1 s/d 10) -> total 10 langkah
        # Jika delta=5, kita butuh hari ke 5 s/d 14 -> total 5 + 9 = 14 langkah
        total_steps = delta + 6

        # ============================
        # --- A. Prediksi ARIMA ---
        # ============================
        
        # Forecast sampai langkah terakhir yang dibutuhkan
        arima_forecast = arima_model.forecast(steps=total_steps)
        
        # Kembalikan ke skala asli jika menggunakan Log
        if IS_LOG_ARIMA:
            arima_forecast = np.expm1(arima_forecast)
        
        # Ambil 10 nilai terakhir (sesuai range 10 hari yang diminta)
        arima_10_days = arima_forecast.iloc[-10:].values
        
        # Cegah nilai negatif
        arima_10_days = np.maximum(arima_10_days, 0.0)

        # ============================
        # --- B. Prediksi LSTM ---
        # ============================
        
        # Ambil data inisial (30 hari terakhir dari training)
        current_batch = metadata['last_sequence'].reshape(1, TIME_STEP, 1)
        lstm_predictions_scaled = []

        # Loop sebanyak total_steps untuk mendapatkan sequence sampai akhir
        for i in range(total_steps):
            # Prediksi 1 langkah
            pred = lstm_model.predict(current_batch, verbose=0)
            
            # Simpan hasil (masih dalam skala 0-1)
            lstm_predictions_scaled.append(pred[0][0])
            
            # Update batch untuk langkah berikutnya
            current_batch = np.append(current_batch[:, 1:, :], [[pred[0]]], axis=1)
        
        # Ambil 10 hasil terakhir
        lstm_10_days_scaled = np.array(lstm_predictions_scaled[-10:]).reshape(-1, 1)
        
        # Kembalikan ke skala asli (Rupiah)
        lstm_10_days_real = scaler.inverse_transform(lstm_10_days_scaled)
        
        # Flatten array dan cegah negatif
        lstm_10_days_final = np.maximum(lstm_10_days_real.flatten(), 0.0)

        # ============================
        # --- C. Format Output ---
        # ============================
        
        results = []
        for i in range(10):
            # Hitung tanggal untuk hari ke-i
            current_date = target_date + timedelta(days=i)
            
            # Ambil nilai raw
            raw_arima = arima_10_days[i]
            raw_lstm = lstm_10_days_final[i]

            # --- SANITIZATION (PERBAIKAN ERROR JSON) ---
            # Cek apakah NaN atau Infinity, jika ya ubah jadi 0.0
            
            # Sanitasi ARIMA
            if np.isnan(raw_arima) or np.isinf(raw_arima):
                val_arima = 0.0
            else:
                val_arima = float(raw_arima)

            # Sanitasi LSTM
            if np.isnan(raw_lstm) or np.isinf(raw_lstm):
                val_lstm = 0.0
            else:
                val_lstm = float(raw_lstm)
            
            # --- END SANITIZATION ---

            results.append({
                "date": current_date,
                "day_index": i + 1,
                "ARIMA": {
                    "value": val_arima,
                    "formatted": f"Rp{val_arima:,.0f}".replace(",", ".")
                },
                "LSTM": {
                    "value": val_lstm,
                    "formatted": f"Rp{val_lstm:,.0f}".replace(",", ".")
                }
            })

        return {
            "start_date": target_date,
            "end_date": target_date + timedelta(days=9),
            "predictions": results
        }

    except Exception as e:
        # Print error di terminal agar mudah debug
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AUTHENTICATION
def hash_password(password):
    byte_password = password.encode('utf-8')
    hashed = bcrypt.hashpw(byte_password, SALT).decode('utf-8')
    return hashed

def authenticate(credentials: schemas.UserLogin, conn: Session):
    user = crud.get_user_by_email(conn, credentials.email)
    if user:
        if user.status_aktivasi == "PENDING":
            raise HTTPException(401, "Akun belum teraktivasi!")

        return hash_password(credentials.password) == user.password
    else:
        return False

def create_token(email: str):
    expiration_time = datetime.now(timezone.utc) + timedelta(days=30)
    token = jwt.encode({"email": email, "exp": expiration_time.timestamp()}, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        if not payload["email"]:
            raise HTTPException(401, "Invalid Access Token")

        return payload
    except JWTError:
        raise HTTPException(401, "Invalid Access Token")

@app.post("/api/token", tags=["Auth"], response_model=schemas.Token)
async def token(form_data: OAuth2PasswordRequestForm = Depends(), conn: Session = Depends(get_db)):
    credentials = schemas.UserLogin(
        email=form_data.username,
        password=form_data.password
    )

    if not authenticate(credentials, conn):
        raise HTTPException(401, "Invalid Credentials!")
    
    access_token = create_token(credentials.email)

    return schemas.Token(
        access_token=access_token,
        token_type="bearer"
    )

@app.get("/api/auth/pending-users/{user_id}", tags=["Auth"], response_model=list[schemas.UserAktivasi])
async def pending_users(user_id: int, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    if crud.get_user_by_id(conn, user_id).role != "OWNER":
        raise HTTPException(401, "Unauthorized access!")
    
    result = crud.get_pending_users(conn)
    users: schemas.UserAktivasi = []

    for user in result:
        users.append(
            schemas.UserAktivasi(
                user_id=user.user_id,
                email=user.email,
                nama_lengkap=user.nama_lengkap,
                role=user.role,
                status_aktivasi=user.status_aktivasi
            )
        )

    return users

@app.put("/api/auth/update-activation-status/{current_user_id}", tags=["Auth"])
async def pending_users(current_user_id: int, body: schemas.UserAktivasiEdit, conn: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    if crud.get_user_by_id(conn, current_user_id).role != "OWNER":
        raise HTTPException(401, "Unauthorized access!")
    
    target_user = crud.get_user_by_id(conn, body.user_id)

    if not body.approve:
        crud.delete_user(conn, target_user)
        return {"status": "success", "message": "berhasil menolak aktivasi akun"}

    result = crud.activate_user(conn, target_user)

    return {"status": "success", "message": "berhasil aktivasi akun"}