from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date

# USER
class UserBase(BaseModel):
    email: str

class UserLogin(UserBase):
    password: str

class UserCreate(UserLogin):
    nama_lengkap: str
    role: str

class UserChangePassword(UserLogin):
    user_id: int
    new_password: str

class UserEdit(UserBase):
    user_id: int
    nama_lengkap: str
    nama_toko: str
    role: str

class UserAktivasiEdit(BaseModel):
    user_id: int
    approve: bool

class UserResponse(UserBase):
    user_id: int
    nama_lengkap: str
    nama_toko: str
    role: str
    csv_path: Optional[str]

    class Config:
        orm_mode = True

class UserAktivasi(UserBase):
    nama_lengkap: str
    role: str
    status_aktivasi: str
    user_id: int

    class Config:
        orm_mode = True

# DATASET
class SalesDataResponse(BaseModel):
    sale_id: int
    invoice: Optional[str] = None
    tanggal_pembayaran: Optional[datetime] = None
    status_terakhir: Optional[str] = None
    nama_produk: Optional[str] = None
    jumlah_produk_dibeli: Optional[int] = None
    harga_jual_idr: Optional[int] = None
    total_penjualan_idr: Optional[int] = None

    class Config:
        orm_mode = True

class DataSummaryResponse(BaseModel):
    total_transaksi: int
    total_produk: int
    periode_awal: datetime
    periode_akhir: datetime
    total_status_selesai: int
    total_status_dibatalkan: int
    total_status_dibatalkan_pembeli: int
    total_status_dibatalkan_penjual: int
    total_status_dibatalkan_sistem: int
    total_status_sedang_dikirim: int

    class Config:
        orm_mode = True

class ProductResponse(BaseModel):
    nomor_produk: int
    nama_produk: str
    total_transaksi: int
    total_penjualan: int

# TOP PRODUCTS
class TopProductsResponse(BaseModel):
    nama_produk: str
    total_unit_terjual: int
    total_transaksi: int
    total_penjualan: int

    class Config:
        orm_mode = True

class TopProductsSummaryResponse(BaseModel):
    total_penjualan_top: int
    total_unit_terjual_top: int

    class Config:
        orm_mode = True

# SALES TREND
class SalesTrendResponse(BaseModel):
    bulan_pembayaran: int
    tahun_pembayaran: int
    total_penjualan: int
    pertumbuhan: float

    class Config:
        orm_mode = True

class TransactionAnalysisResponse(BaseModel):
    avg_penjualan: float
    max_penjualan: int
    min_penjualan: int
    median_penjualan: int
    std_penjualan: float

    class Config:
        orm_mode = True

class TemporalPatternResponse(BaseModel):
    hari_transaksi: int
    jumlah_transaksi_hari: int
    bulan_transaksi: int
    jumlah_transaksi_bulan: int
    rentang_jam_transaksi: str
    jumlah_transaksi_jam: int

    class Config:
        orm_mode = True

# PREDICTIONS
class DateInput(BaseModel):
    target_date: date  # Format: YYYY-MM-DD

class PredictionArgs(BaseModel):
    csv_path: str
    user_id: int

class JobStatus(BaseModel):
    job_status: str

    class Config:
        orm_mode = True

class PredictionMetricsResponse(BaseModel):
    prediction_metric_id: int
    arima_mae: float
    arima_rmse: float
    arima_waktu_train: float
    arima_memori: float
    lstm_mae: float
    lstm_rmse: float
    lstm_waktu_train: float
    lstm_memori: float

    class Config:
        orm_mode = True

class PredictionComparisonsResponse(BaseModel):
    prediction_comparison_id: int
    hari: int
    hasil_total_penjualan_aktual: float
    hasil_total_penjualan_arima: float
    hasil_total_penjualan_lstm: float

    class Config:
        orm_mode = True

class TotalPredictionResponse(BaseModel):
    total_prediction_id: int
    hasil_tanggal: datetime
    hasil_total_penjualan_arima: float
    hasil_total_penjualan_lstm: float

    class Config:
        orm_mode = True

class ProductPredictionBase(BaseModel):
    hasil_nama_produk_arima: str
    hasil_nama_produk_lstm: str

class DailyPredictionsResponse(ProductPredictionBase):
    daily_product_prediction_id: int
    hari: int

    class Config:
        orm_mode = True

class WeeklyPredictionsResponse(ProductPredictionBase):
    weekly_product_prediction_id: int
    minggu: int

    class Config:
        orm_mode = True

class MonthlyPredictionsResponse(ProductPredictionBase):
    monthly_product_prediction_id: int
    bulan: int

    class Config:
        orm_mode = True

# TOKEN
class Token(BaseModel):
    access_token: str
    token_type: str

    class Config:
        orm_mode = True