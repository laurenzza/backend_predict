from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()
host=os.getenv("DB_HOST")
port=os.getenv("DB_PORT")
user=os.getenv("DB_USER")
password=os.getenv("DB_PASS")
database=os.getenv("DB_NAME")
ca_path=os.getenv("CA_PATH")

DB_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

connect_args = {}
if ca_path:
    connect_args = {
        "ssl": {
            "ca": ca_path
        }
    }

# 3. Masukkan connect_args ke dalam create_engine
engine = create_engine(DB_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()