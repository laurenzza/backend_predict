from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Double, func
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement="auto")
    email = Column(String(255), nullable=False, unique=True, index=True)
    password = Column(String(255), nullable=False)
    nama_lengkap = Column(String(255), nullable=False)
    nama_toko = Column(String(255), nullable=False, default="")
    role = Column(String(50), nullable=False)
    status_aktivasi = Column(String(50), nullable=False, default="PENDING")
    csv_path = Column(String(255), nullable=False, default="")

class Sale(Base):
    __tablename__ = "sales"

    sale_id = Column(Integer, primary_key=True, autoincrement="auto")
    invoice = Column(String, nullable=True, index=True)
    tanggal_pembayaran = Column(DateTime, nullable=True, index=True)
    status_terakhir = Column(String(255), nullable=True, index=True)
    nama_produk = Column(String(255), nullable=True, index=True)
    jumlah_produk_dibeli = Column(Integer, nullable=True)
    harga_jual_idr = Column(Integer, nullable=True)
    total_penjualan_idr = Column(Integer, nullable=True)

    user_id = Column(Integer, nullable=False, index=True)

class PredictionMetric(Base):
    __tablename__ = "prediction_metrics"

    prediction_metric_id = Column(Integer, primary_key=True, autoincrement="auto")
    arima_mae = Column(Double, nullable=False)
    arima_rmse = Column(Double, nullable=False)
    arima_waktu_train = Column(Double, nullable=False)
    arima_memori = Column(Double, nullable=False)
    lstm_mae = Column(Double, nullable=False)
    lstm_rmse = Column(Double, nullable=False)
    lstm_waktu_train = Column(Double, nullable=False)
    lstm_memori = Column(Double, nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

class PredictionComparison(Base):
    __tablename__ = "prediction_comparisons"

    prediction_comparison_id = Column(Integer, primary_key=True, autoincrement="auto")
    hari = Column(Integer, nullable=False)
    hasil_total_penjualan_aktual = Column(Double, nullable=False)
    hasil_total_penjualan_arima = Column(Double, nullable=False)
    hasil_total_penjualan_lstm = Column(Double, nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

class TotalPrediction(Base):
    __tablename__ = "total_predictions"

    total_prediction_id = Column(Integer, primary_key=True, autoincrement="auto")
    hasil_tanggal = Column(DateTime, nullable=False)
    hasil_total_penjualan_arima = Column(Double, nullable=False)
    hasil_total_penjualan_lstm = Column(Double, nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

class DailyProductPrediction(Base):
    __tablename__ = "daily_product_predictions"

    daily_product_prediction_id = Column(Integer, primary_key=True, autoincrement="auto")
    hari = Column(Integer, nullable=False)
    hasil_nama_produk_arima = Column(String(255), nullable=False)
    hasil_nama_produk_lstm = Column(String(255), nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

class WeeklyProductPrediction(Base):
    __tablename__ = "weekly_product_predictions"

    weekly_product_prediction_id = Column(Integer, primary_key=True, autoincrement="auto")
    minggu = Column(Integer, nullable=False)
    hasil_nama_produk_arima = Column(String(255), nullable=False)
    hasil_nama_produk_lstm = Column(String(255), nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

class MonthlyProductPrediction(Base):
    __tablename__ = "monthly_product_predictions"

    monthly_product_prediction_id = Column(Integer, primary_key=True, autoincrement="auto")
    bulan = Column(Integer, nullable=False)
    hasil_nama_produk_arima = Column(String(255), nullable=False)
    hasil_nama_produk_lstm = Column(String(255), nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)

class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    prediction_job_id = Column(Integer, primary_key=True, autoincrement="auto")
    status = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=True, server_default=func.now())

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)