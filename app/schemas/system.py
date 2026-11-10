from pydantic import BaseModel, Field


class ModelsLoadedStatus(BaseModel):
    xgboost: bool = False


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Trạng thái tổng quan dịch vụ AI")
    service: str = Field(default="ShrimpMate AI Feeding Service", description="Tên dịch vụ")
    version: str = Field(default="2.0.0", description="Phiên bản dịch vụ AI")
    modelName: str = Field(default="XGBoost-Regressor", description="Tên mô hình ML chính")
    modelsLoaded: ModelsLoadedStatus

