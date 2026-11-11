from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Mã định danh đặc trưng (thuộc 7 đặc trưng XGBoost)")
    name: str = Field(..., description="Tên tiếng Việt hiển thị người dùng")
    value: str = Field(..., description="Giá trị đo đạc/sinh học đầu vào thực tế")
    impact: Literal["INCREASE", "DECREASE", "NEUTRAL"] = Field(..., description="Chiều hướng tác động theo TreeSHAP")
    deltaKg: float = Field(..., description="Mức đóng góp thực tế vào kết quả dự báo theo XGBoost TreeSHAP (kg)")
    description: str = Field(..., description="Giải thích tác động nông học của đặc trưng")


class FeedPredictionRequest(BaseModel):
    pondId: Optional[str] = Field(default=None, description="Mã định danh ao nuôi (dùng để đồng bộ log với Backend)")
    doc: int = Field(..., description="Tuổi tôm (Days of Culture - ngày)", ge=1, le=180)
    avgWeightG: float = Field(..., description="Trọng lượng tôm bình quân (g/con)", ge=0.1, le=60.0)
    doMin: float = Field(..., description="Nồng độ Oxy hòa tan tối thiểu DO_min (mg/L)", ge=0.0, le=15.0)
    deltaTan: float = Field(..., description="Chênh lệch tổng đạm Amoniac Delta_TAN giữa 2 lần cho ăn (mg/L)", ge=-5.0, le=10.0)
    morningRainMm: float = Field(default=0.0, description="Lượng mưa ghi nhận lúc 06:00 (mm)", ge=0.0)
    prevFeedAmountKg: float = Field(..., description="Lượng thức ăn đã tiêu thụ cữ liền trước (kg)", ge=0.0)
    waterChangeM3: float = Field(default=0.0, description="Lượng nước thay trong ngày (m3)", ge=0.0)


class FeedPredictionResult(BaseModel):
    predictedFeedAmountKg: float = Field(..., description="Lượng thức ăn khuyến nghị cho cữ ăn tiếp theo (kg/cữ)")
    baseValueKg: float = Field(..., description="Giá trị kỳ vọng cơ sở (TreeSHAP bias) của mô hình XGBoost (kg)")
    modelName: str = Field(default="XGBoost-Regressor", description="Tên mô hình AI dự báo")
    modelVersion: str = Field(default="v2.0.0", description="Phiên bản mô hình AI")
    featureImportance: Dict[str, float] = Field(default_factory=dict, description="Độ quan trọng đóng góp của 7 đặc trưng")
    featureContributions: List[FeatureContribution] = Field(default_factory=list, description="Đóng góp thực tế từng đặc trưng tính bằng XGBoost TreeSHAP (XAI)")
    explanation: str = Field(..., description="Giải thích khuyến nghị AI dựa trên 7 đặc trưng đầu vào và kết quả TreeSHAP")
