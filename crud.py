from sqlalchemy import func, case, extract, not_, or_, select, desc
from sqlalchemy.orm import Session
import models, schemas
from dotenv import load_dotenv
import bcrypt, os

# USERS
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def hash_password(password):
    load_dotenv()
    SALT = os.getenv("SALT").encode('utf-8')
    byte_password = password.encode('utf-8')
    hashed = bcrypt.hashpw(byte_password, SALT).decode('utf-8')
    return hashed

def insert_user(db: Session, user: schemas.UserCreate):
    hashed_pwd = hash_password(user.password)
    db_user = models.User(
        email=user.email,
        password=hashed_pwd,
        nama_lengkap=user.nama_lengkap,
        role=user.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

def update_user(db: Session, db_user: models.User, user: schemas.UserResponse):
    db_user.email = user.email
    db_user.nama_lengkap = user.nama_lengkap
    db_user.nama_toko = user.nama_toko

    db.commit()
    db.refresh(db_user)

    return db_user

def update_password(db: Session, db_user: models.User, hashed_password: str):
    db_user.password = hashed_password

    db.commit()
    db.refresh(db_user)

    return db_user

def delete_user(db: Session, db_user: models.User):

    db.delete(db_user)
    db.commit()

    return {"message": "success"}

# AUTH
def get_pending_users(db: Session):
    return db.query(models.User).filter(models.User.status_aktivasi == "PENDING").all()

def activate_user(db: Session, db_user: models.User):

    db_user.status_aktivasi = "ACTIVATED"

    db.commit()
    db.refresh(db_user)

    return db_user

# DATASET
def get_sales(db: Session, user_id: int, limit: int, offset: int, year: str, status: str):
    if year == "all" and status == "all":
        return db.query(models.Sale).limit(limit).offset(offset)
    
    if year == "all" and status != "others":
        return db.query(models.Sale).filter(models.Sale.status_terakhir == status).limit(limit).offset(offset)
    
    if year == "all" and status == "others":
        return db.query(models.Sale).filter(or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).limit(limit).offset(offset)
    
    if status == "all":
        return db.query(models.Sale).filter(extract("year", models.Sale.tanggal_pembayaran) == year).limit(limit).offset(offset)
    
    if status == "others":
        return db.query(models.Sale).filter(extract("year", models.Sale.tanggal_pembayaran) == year, or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).limit(limit).offset(offset)

    return db.query(models.Sale).filter(extract("year", models.Sale.tanggal_pembayaran) == year, models.Sale.status_terakhir == status).limit(limit).offset(offset)

def delete_sales(db: Session, user_id: int):
    db.query(models.Sale).delete()

def get_sales_summary(db: Session, user_id: int):
    result = db.query(
        func.count(models.Sale.sale_id).label("total_transaksi"),
        func.count(func.distinct(models.Sale.nama_produk)).label("total_produk"),
        func.sum(case((models.Sale.status_terakhir == "Pesanan Selesai", 1), else_=0)).label("total_status_selesai"),
        func.sum(case((models.Sale.status_terakhir.like("Dibatalkan%"), 1), else_=0)).label("total_status_dibatalkan"),
        func.sum(case((models.Sale.status_terakhir == "Dibatalkan Pembeli", 1), else_=0)).label("total_status_dibatalkan_pembeli"),
        func.sum(case((models.Sale.status_terakhir == "Dibatalkan Penjual", 1), else_=0)).label("total_status_dibatalkan_penjual"),
        func.sum(case((models.Sale.status_terakhir == "Dibatalkan Sistem", 1), else_=0)).label("total_status_dibatalkan_sistem"),
        func.sum(case((models.Sale.status_terakhir == "Sedang Dikirim", 1), else_=0)).label("total_status_sedang_dikirim"),
        func.min(models.Sale.tanggal_pembayaran).label("periode_awal"),
        func.max(models.Sale.tanggal_pembayaran).label("periode_akhir")
    ).one()

    if result.total_transaksi == 0 and result.total_produk == 0:
        return None

    return result

def get_total_sales_idr(db: Session, user_id: int):
    return db.query(
        func.sum(models.Sale.total_penjualan_idr)
    ).group_by(func.date(models.Sale.tanggal_pembayaran)).all()

def get_products(db: Session, user_id: int, limit: int, offset: int):
    return db.query(
        models.Sale.nama_produk.label("nama_produk"),
        func.count(models.Sale.sale_id).label("total_transaksi"),
        func.sum(models.Sale.total_penjualan_idr).label("total_penjualan"),
    ).group_by("nama_produk").order_by("nama_produk").limit(limit).offset(offset)

def count_fetch_filter_sales(db: Session, user_id: int, year: str, status: str):
    if year == "all" and status == "all":
        return db.query(func.count(models.Sale.sale_id).label("rows")).one()
    
    if year == "all" and status != "others":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(models.Sale.status_terakhir == status).one()
    
    if year == "all" and status == "others":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).one()
    
    if status == "all":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(extract("year", models.Sale.tanggal_pembayaran) == year).one()
    
    if status == "others":
        return db.query(func.count(models.Sale.sale_id).label("rows")).filter(extract("year", models.Sale.tanggal_pembayaran) == year, or_(not_(models.Sale.status_terakhir.in_(["Pesanan Selesai", "Dibatalkan Penjual", "Dibatalkan Pembeli", "Dibatalkan Sistem", "Sedang Dikirim"])), models.Sale.status_terakhir.is_(None))).one()

    return db.query(func.count(models.Sale.sale_id).label("rows")).filter(extract("year", models.Sale.tanggal_pembayaran) == year, models.Sale.status_terakhir == status).one()

# TOP PRODUCTS
def get_top_products_summary(db: Session, user_id: int):
    subquery = (
        select(
            models.Sale.nama_produk,
            func.sum(models.Sale.total_penjualan_idr).label("sum_penjualan"),
            func.sum(models.Sale.jumlah_produk_dibeli).label("sum_dibeli"),
        )
        .group_by(models.Sale.nama_produk)
        .order_by(desc("sum_penjualan"))
        .limit(10)
    ).subquery("top_10")

    summary_query = select(
        func.sum(subquery.c.sum_penjualan).label("total_penjualan_top"),
        func.sum(subquery.c.sum_dibeli).label("total_unit_terjual_top"),
    )

    return db.execute(summary_query).fetchone()

def get_top_products(db: Session, user_id: int):
    return db.query(
        models.Sale.nama_produk.label("nama_produk"),
        func.count(models.Sale.sale_id).label("total_transaksi"),
        func.sum(models.Sale.total_penjualan_idr).label("total_penjualan"),
        func.sum(models.Sale.jumlah_produk_dibeli).label("total_unit_terjual"),
    ).group_by("nama_produk").order_by(desc("total_penjualan")).limit(10)

# STATISTICS
def get_monthly_sales_trend(db: Session, user_id: int):
    return db.query(
        func.month(models.Sale.tanggal_pembayaran).label("bulan_pembayaran"),
        func.year(models.Sale.tanggal_pembayaran).label("tahun_pembayaran"),
        func.sum(models.Sale.total_penjualan_idr).label("total_penjualan")
    ).group_by("tahun_pembayaran", "bulan_pembayaran").order_by(desc("tahun_pembayaran"), desc("bulan_pembayaran")).limit(7)

def get_transaction_analysis(db: Session, user_id: int):
    subq = (
        select(
            models.Sale.total_penjualan_idr,
            func.percent_rank().over(order_by=models.Sale.total_penjualan_idr).label("p"),
        )
        .subquery()
    )

    median_subq = (
        select(subq.c.total_penjualan_idr)
        .where(subq.c.p <= 0.5)
        .order_by(subq.c.p.desc())
        .limit(1)
        .scalar_subquery()
    )

    query = (
        select(
            func.avg(models.Sale.total_penjualan_idr).label("avg_penjualan"),
            func.max(models.Sale.total_penjualan_idr).label("max_penjualan"),
            func.min(models.Sale.total_penjualan_idr).label("min_penjualan"),
            func.stddev_samp(models.Sale.total_penjualan_idr).label("std_penjualan"),
            median_subq.label("median_penjualan"),
        )
    )

    return db.execute(query).mappings().first()

def get_temporal_day(db: Session, user_id: int):
    return db.query(
        func.dayofweek(models.Sale.tanggal_pembayaran).label("hari_transaksi"),
        func.count(models.Sale.sale_id).label("jumlah_transaksi_hari")
    ).group_by("hari_transaksi").order_by(desc("jumlah_transaksi_hari")).limit(1).one()

def get_temporal_month(db: Session, user_id: int):
    return db.query(
        func.month(models.Sale.tanggal_pembayaran).label("bulan_transaksi"),
        func.count(models.Sale.sale_id).label("jumlah_transaksi_bulan")
    ).group_by("bulan_transaksi").order_by(desc("jumlah_transaksi_bulan")).limit(1).one()

def get_temporal_time_range(db: Session, user_id: int):
    sub = (
        select(
            models.Sale.sale_id,
            func.floor(func.hour(models.Sale.tanggal_pembayaran) / 2).label("bucket")
        )
        .subquery()
    )

    query = (
        select(
            func.concat(
                func.lpad(sub.c.bucket * 2, 2, "0"),
                ":00 - ",
                func.lpad(sub.c.bucket * 2 + 2, 2, "0"),
                ":00"
            ).label("rentang_jam_transaksi"),
            func.count(sub.c.sale_id).label("jumlah_transaksi_jam")
        )
        .group_by(sub.c.bucket)
        .order_by(func.count(sub.c.sale_id).desc())
    )

    return db.execute(query).mappings().first()

# PREDICTIONS
def get_prediction_job(db: Session, user_id: int):
    return db.query(models.PredictionJob).first()

def get_total_predictions(db: Session, user_id: int):
    return db.query(models.TotalPrediction).all()

def get_prediction_comparisons(db: Session, user_id: int):
    return db.query(models.PredictionComparison).all()

def get_prediction_metrics(db: Session, user_id: int):
    return db.query(models.PredictionMetric).first()

def get_daily_predictions(db: Session, user_id: int):
    return db.query(models.DailyProductPrediction).all()

def get_weekly_predictions(db: Session, user_id: int):
    return db.query(models.WeeklyProductPrediction).all()

def get_monthly_predictions(db: Session, user_id: int):
    return db.query(models.MonthlyProductPrediction).all()

def delete_prediction_job(db: Session, user_id: int):
    db.query(models.PredictionJob).delete()

def delete_all_predictions(db: Session, user_id: int):
    db.query(models.PredictionMetric).delete()
    db.query(models.PredictionComparison).delete()
    db.query(models.TotalPrediction).delete()
    db.query(models.DailyProductPrediction).delete()
    db.query(models.WeeklyProductPrediction).delete()
    db.query(models.MonthlyProductPrediction).delete()