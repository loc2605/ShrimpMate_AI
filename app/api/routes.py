import time
import logging
from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.system import (
    HealthResponse,
    ModelsLoadedStatus,
)
from app.schemas.feed_prediction import (
    FeedPredictionRequest,
    FeedPredictionResult,
)
from app.services.xgboost_service import xgboost_feed_service

logger = logging.getLogger("shrimpmate.ai")

router = APIRouter()


@router.post(
    "/ai/predict-feed-demand",
    response_model=FeedPredictionResult,
    summary="Dự báo lượng thức ăn thông minh (XGBoost 7 đặc trưng môi trường & tăng trưởng)",
    tags=["Dự báo thức ăn AI"],
)
async def predict_feed_demand(request: FeedPredictionRequest) -> FeedPredictionResult:
    """
    Dự báo lượng thức ăn tối ưu cho cữ ăn tiếp theo bằng mô hình XGBoost:
    - DOC (Ngày tuổi nuôi)
    - Trọng lượng bình quân mẫu (ABW - g/con)
    - DO min (Nồng độ oxy hòa tan tối thiểu mg/L)
    - Delta TAN (Chênh lệch đạm Amoniac mg/L)
    - Lượng mưa sáng sớm (mm)
    - Lượng thức ăn cữ trước (kg)
    - Lượng nước thay trong ngày (m3)
    """
    start_time = time.perf_counter()
    pond_label = request.pondId[:8] if request.pondId else "N/A"
    try:
        result = await run_in_threadpool(xgboost_feed_service.predict, req=request)
    except Exception as exc:
        logger.error(f"[HỆ-THỐNG-AI] [Ao: {pond_label}] Lỗi dự báo lượng thức ăn XGBoost: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi dự báo lượng thức ăn AI: {str(exc)}",
        ) from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        f"[HỆ-THỐNG-AI] [Ao: {pond_label}] -> DOC: {request.doc} | Trọng lượng TB: {request.avgWeightG:.2f}g | "
        f"DO_min: {request.doMin:.2f}mg/L | Dự báo: {result.predictedFeedAmountKg:.2f}kg | "
        f"Độ trễ: {elapsed_ms:.1f}ms"
    )
    return result


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái dịch vụ và mô hình AI",
    tags=["Hệ thống"],
)
async def health_check() -> HealthResponse:
    """Kiểm tra sức khỏe dịch vụ và trạng thái nạp mô hình AI."""
    return HealthResponse(
        status="healthy" if xgboost_feed_service.is_loaded else "degraded",
        service=settings.PROJECT_NAME,
        version=xgboost_feed_service.model_version,
        modelName="XGBoost-Regressor",
        modelsLoaded=ModelsLoadedStatus(
            xgboost=xgboost_feed_service.is_loaded,
        ),
    )
