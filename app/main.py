import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.services.xgboost_service import xgboost_feed_service


def _resolve_timezone(tz_name: str):
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=7))


class TimezoneFormatter(logging.Formatter):
    """Formatter đồng bộ múi giờ (mặc định Asia/Ho_Chi_Minh, GMT+7) với Backend NestJS."""

    def __init__(self, fmt: str, datefmt: str, tz_name: str):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._tz = _resolve_timezone(tz_name)

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self._tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


def _configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        TimezoneFormatter(
            fmt="%(asctime)s | %(levelname)-5s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            tz_name=settings.TIMEZONE,
        )
    )
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        handlers=[handler],
        force=True,
    )


_configure_logging()
logger = logging.getLogger("shrimpmate.ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load XGBoost feed model once into memory.
    """
    logger.info("=" * 60)
    logger.info("Starting ShrimpMate AI Feeding Service (Pure AI Feeding Mode)...")
    logger.info(f"Python: {sys.version.split()[0]}")

    # Load XGBoost Feed Prediction model
    logger.info("Loading XGBoost Feed Demand Prediction model...")
    try:
        xgboost_feed_service.load_model()
        logger.info("Mô hình AI dự báo lượng thức ăn đã sẵn sàng phục vụ.")
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {e}")

    logger.info("=" * 60)

    yield

    logger.info("Shutting down ShrimpMate AI Service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Microservice AI dự báo lượng thức ăn thông minh cho ao nuôi tôm.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to measure latency of every request and attach header."""
    import time
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Register API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["General"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /ai/predict-feed-demand",
            "GET /health",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    )
