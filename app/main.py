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
    """Bộ định dạng log đồng bộ múi giờ (mặc định Asia/Ho_Chi_Minh, GMT+7) với Backend NestJS."""

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
    Trình quản lý vòng đời ứng dụng (Lifespan) để nạp mô hình XGBoost một lần duy nhất vào bộ nhớ.
    """
    logger.info("=" * 60)
    logger.info("Đang khởi động ShrimpMate AI Feeding Service (Chế độ dự báo thức ăn thuần AI)...")
    logger.info(f"Phiên bản Python: {sys.version.split()[0]}")

    # Nạp mô hình XGBoost dự báo nhu cầu thức ăn
    logger.info("Đang nạp mô hình XGBoost dự báo nhu cầu thức ăn...")
    try:
        xgboost_feed_service.load_model()
        logger.info("Mô hình AI dự báo lượng thức ăn đã sẵn sàng phục vụ.")
    except Exception as e:
        logger.error(f"Lỗi khi nạp mô hình XGBoost: {e}")

    logger.info("=" * 60)

    yield

    logger.info("Đang tắt ShrimpMate AI Service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Microservice AI dự báo lượng thức ăn thông minh cho ao nuôi tôm.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cấu hình CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware đo thời gian xử lý của mỗi yêu cầu và gắn vào header phản hồi."""
    import time
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Đăng ký các router API
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Hệ thống"])
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
