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
    tags=["Feeding AI"],
)
async def predict_feed_demand(request: FeedPredictionRequest) -> FeedPredictionResult:
    """
    Dự báo lượng thức ăn tối ưu cho cữ ăn tiếp theo bằng mô hình XGBoost:
    - DOC (Ngày tuổi nuôi)
    - Average Body Weight (Trọng lượng bình quân mẫu g/con)
    - DO min (Nồng độ oxy hòa tan tối thiểu mg/L)
    - Delta TAN (Chênh lệch đạm Amoniac mg/L)
    - Morning rain (Lượng mưa sáng sớm mm)
    - Previous feed (Lượng thức ăn cữ trước kg)
    - Water change (Lượng nước thay m3)
    """
    start_time = time.perf_counter()
    pond_label = request.pondId[:8] if request.pondId else "N/A"
    try:
        result = await run_in_threadpool(xgboost_feed_service.predict, req=request)
    except Exception as exc:
        logger.error(f"[AI-ENGINE] [Pond: {pond_label}] XGBoost feed prediction error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi dự báo lượng thức ăn AI: {str(exc)}",
        ) from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        f"[AI-ENGINE] [Pond: {pond_label}] -> DOC: {request.doc} | ABW: {request.avgWeightG:.2f}g | "
        f"DO_min: {request.doMin:.2f}mg/L | Predict: {result.predictedFeedAmountKg:.2f}kg | "
        f"Latency: {elapsed_ms:.1f}ms"
    )
    return result


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái dịch vụ và mô hình AI",
    tags=["System"],
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
