import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import String, create_engine, text
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

try:
    from geoalchemy2 import Geometry
except Exception:
    Geometry = None


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "waste_model.pt"

# Vercel filesystem is read-only
# Create uploads folder only in local environment
if os.environ.get("VERCEL") != "1":
    if os.environ.get("VERCEL") != "1":
   if os.environ.get("VERCEL") != "1":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24).hex())
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024

    ADMIN_EMAIL = os.environ.get(
        "ADMIN_EMAIL",
        "admin@geosmart.local"
    )

    ADMIN_PASSWORD = os.environ.get(
        "ADMIN_PASSWORD",
        "admin123"
    )

    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{(BASE_DIR / 'geosmart.db').as_posix()}",
    )

    HOTSPOT_CLUSTER_EPS_METERS = float(
        os.environ.get("HOTSPOT_CLUSTER_EPS_METERS", "250")
    )

    HOTSPOT_MIN_POINTS = int(
        os.environ.get("HOTSPOT_MIN_POINTS", "1")
    )

    VERIFIED_THRESHOLD = float(
        os.environ.get("VERIFIED_THRESHOLD", "0.5")
    )

    RESPONSE_SPEED_KMPH = float(
        os.environ.get("RESPONSE_SPEED_KMPH", "24")
    )


ISSUE_TYPES = {
    "collection_delay": "Delayed collection",
    "overflow": "Overflowing bin",
    "illegal_dumping": "Illegal dumping",
    "hazardous_waste": "Hazardous waste leak",
    "no_segregation": "Segregation breach",
    "other": "Other",
}


ISSUE_STREAMS = {
    "collection_delay": "Mixed municipal waste",
    "overflow": "Organic-heavy overflow",
    "illegal_dumping": "Mixed construction and household waste",
    "hazardous_waste": "Hazardous waste",
    "no_segregation": "Mixed recyclables and organics",
    "other": "Mixed waste",
}


ISSUE_BASE_SCORE = {
    "collection_delay": 42,
    "overflow": 58,
    "illegal_dumping": 62,
    "hazardous_waste": 82,
    "no_segregation": 46,
    "other": 40,
}


PRIORITY_BONUS = {
    "normal": 0,
    "high": 10,
    "critical": 18,
}


WARD_CENTROIDS = {
    "Ward 09": (12.9304, 77.6038),
    "Ward 11": (12.9348, 77.6112),
    "Ward 14": (12.9369, 77.6219),
    "Ward 16": (12.9432, 77.6298),
    "Ward 18": (12.9478, 77.6144),
    "Ward 21": (12.9273, 77.6292),
}


DEPOT_COORDS = {
    "South Depot": (12.9287, 77.6070),
    "Central Transfer Hub": (12.9412, 77.6176),
}


ROUTE_GRAPH = {
    "South Depot": [
        "Ward 09",
        "Ward 11",
        "Ward 21",
        "Central Transfer Hub"
    ],

    "Central Transfer Hub": [
        "South Depot",
        "Ward 11",
        "Ward 14",
        "Ward 16",
        "Ward 18"
    ],

    "Ward 09": [
        "South Depot",
        "Ward 11"
    ],

    "Ward 11": [
        "South Depot",
        "Central Transfer Hub",
        "Ward 09",
        "Ward 14",
        "Ward 18"
    ],

    "Ward 14": [
        "Central Transfer Hub",
        "Ward 11",
        "Ward 16",
        "Ward 21"
    ],

    "Ward 16": [
        "Central Transfer Hub",
        "Ward 14",
        "Ward 18"
    ],

    "Ward 18": [
        "Central Transfer Hub",
        "Ward 11",
        "Ward 16"
    ],

    "Ward 21": [
        "South Depot",
        "Ward 14"
    ],
}


NODE_COORDS = {
    **DEPOT_COORDS,
    **WARD_CENTROIDS
}


ENGINE = create_engine(
    Config.DATABASE_URL,
    future=True,
    pool_pre_ping=True
)


SessionLocal = scoped_session(
    sessionmaker(
        bind=ENGINE,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False
    )
)


Base = declarative_base()


def is_postgres():
    return ENGINE.dialect.name == "postgresql"


def geometry_column_type():
    if Geometry is None or not is_postgres():
        return String(255)

    return Geometry(
        "POINT",
        srid=4326,
        spatial_index=True
    )


@contextmanager
def session_scope():
    session = SessionLocal()

    try:
        yield session
        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


def init_database():

    from models.complaint import Complaint
    from models.hotspot import Hotspot

    if is_postgres():
        with ENGINE.begin() as connection:
            connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS postgis")
            )

    Base.metadata.create_all(bind=ENGINE)