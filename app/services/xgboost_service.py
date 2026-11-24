import os
import json
import logging
from datetime import datetime
import numpy as np
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.schemas.feed_prediction import (
    FeatureContribution,
    FeedPredictionRequest,
    FeedPredictionResult,
)

logger = logging.getLogger("shrimpmate.ai.xgboost")

FEATURE_NAMES = [
    "doc",
    "avg_weight_g",
    "do_min",
    "delta_tan",
    "morning_rain_mm",
    "prev_feed_kg",
    "water_change_m3",
]


class XgboostFeedService:
    def __init__(self):
        self.model_path = os.path.join(settings.MODEL_DIR, "xgboost_feed_model.json")
        self.meta_file = os.path.join(settings.MODEL_DIR, "model_meta.json")
        self.feature_names = FEATURE_NAMES
        self.model_version = "v2.0.0"
        self.last_retrained_at: Optional[str] = None
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.is_loaded = False
        self._feature_importances: Dict[str, float] = {}

    def _load_meta(self):
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.model_version = meta.get("version", self.model_version)
                    self.last_retrained_at = meta.get("lastRetrainedAt")
            except Exception as e:
                logger.warning(f"Không thể đọc file meta {self.meta_file}: {e}")

    def _save_meta(self):
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        meta = {
            "version": self.model_version,
            "lastRetrainedAt": self.last_retrained_at,
            "updatedAt": datetime.now().isoformat(),
        }
        try:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Lỗi khi lưu meta mô hình: {e}")

    def load_model(self):
        """
        Nạp mô hình XGBoost đã lưu từ weights, hoặc tự động khởi tạo và huấn luyện
        theo quy luật sinh học tôm thẻ chân trắng và điều kiện quan trắc môi trường.
        """
        self._load_meta()
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                # Xác thực mô hình nạp vào khớp với 7 đặc trưng
                if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ == len(self.feature_names):
                    self.is_loaded = True
                    self._compute_importances()
                    logger.info(f"Đã nạp thành công mô hình XGBoost {self.model_version} ({len(self.feature_names)} đặc trưng) từ {self.model_path}")
                    return
                else:
                    logger.info("Mô hình cũ không khớp số lượng đặc trưng mới (7 đặc trưng). Huấn luyện lại...")
            except Exception as exc:
                logger.warning(f"Không thể đọc file {self.model_path}: {exc}. Tiến hành huấn luyện lại.")

        logger.info("Khởi tạo và huấn luyện mô hình XGBoost dự báo lượng thức ăn...")
        self._train_baseline_model()
        self.is_loaded = True
        self._compute_importances()

    def _compute_importances(self):
        try:
            scores = self.model.feature_importances_
            total = sum(scores) if sum(scores) > 0 else 1.0
            self._feature_importances = {
                feat: float(round(score / total, 4))
                for feat, score in zip(self.feature_names, scores)
            }
        except Exception:
            self._feature_importances = {feat: round(1.0 / len(self.feature_names), 4) for feat in self.feature_names}

    def _train_baseline_model(self):
        """
        Huấn luyện mô hình XGBoost dựa trên quy luật sinh học tôm thẻ chân trắng (Litopenaeus vannamei),
        cân nhắc các biến môi trường nước, thời tiết và lượng ăn cữ trước.
        """
        np.random.seed(42)
        n_samples = 4000

        # 1. Phân phối biến đầu vào
        doc = np.random.randint(10, 105, size=n_samples)
        # Khối lượng tăng theo ngày tuổi DOC + độ lệch tự nhiên
        avg_weight_g = 0.2 + 0.28 * doc + np.random.normal(0, 1.2, size=n_samples)
        avg_weight_g = np.clip(avg_weight_g, 1.0, 35.0)

        # Mật độ tôm và tỷ lệ sống ước tính theo DOC (thả 120 con/m2 trong ao 1000m2)
        survival_rate = np.clip(0.95 - (doc / 120.0) * 0.25 + np.random.normal(0, 0.03, size=n_samples), 0.5, 0.95)
        pond_density = 120.0 * survival_rate
        biomass_kg = (pond_density * 1000.0 * avg_weight_g) / 1000.0

        # Môi trường: Oxy hòa tan DO tối thiểu (mg/L)
        do_min = np.random.normal(5.2, 0.9, size=n_samples)
        do_min = np.clip(do_min, 2.5, 7.5)

        # Chênh lệch đạm Amoniac Delta TAN (mg/L)
        delta_tan = np.random.exponential(scale=0.2, size=n_samples)
        delta_tan = np.clip(delta_tan, -0.1, 1.5)

        # Thời tiết mưa sáng sớm (xác suất mưa 20%)
        is_raining = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        morning_rain_mm = is_raining * np.random.uniform(5.0, 45.0, size=n_samples)

        # Lượng nước thay (m3)
        water_change_m3 = np.random.choice([0.0, 20.0, 50.0, 100.0], size=n_samples, p=[0.6, 0.2, 0.15, 0.05])

        # 2. Tính toán lượng thức ăn mục tiêu theo quy chuẩn nông học
        daily_rate_percent = 7.0 / (avg_weight_g ** 0.28)
        feed_daily_kg = biomass_kg * (daily_rate_percent / 100.0)
        feed_per_meal_base = feed_daily_kg / 4.0  # Mỗi ngày chia 4 cữ ăn

        # Lượng thức ăn cữ trước
        prev_feed_kg = feed_per_meal_base * np.random.uniform(0.9, 1.1, size=n_samples)

        # Ảnh hưởng từ môi trường & thời tiết:
        # Mưa lớn >= 10mm -> sức ăn giảm 30%
        rain_factor = np.where(morning_rain_mm >= 10.0, 0.70, 1.0)

        # DO thấp -> giảm mạnh hoặc ngưng ăn
        do_factor = np.ones(n_samples)
        do_factor[do_min < 4.0] = 0.5
        do_factor[do_min < 3.5] = 0.0

        # Khí độc TAN cao -> giảm ăn
        tan_factor = np.where(delta_tan > 0.5, 0.75, 1.0)

        # Thay nước hợp lý kích thích ăn nhẹ
        water_factor = np.where(water_change_m3 > 0, 1.05, 1.0)

        # Lượng thức ăn mục tiêu
        target_feed = feed_per_meal_base * rain_factor * do_factor * tan_factor * water_factor
        target_feed += np.random.normal(0, 0.15, size=n_samples)
        target_feed = np.clip(target_feed, 0.0, None)

        X = np.column_stack([
            doc,
            avg_weight_g,
            do_min,
            delta_tan,
            morning_rain_mm,
            prev_feed_kg,
            water_change_m3,
        ])
        y = target_feed

        split_idx = int(n_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds = self.model.predict(X_val)
        mae = float(np.mean(np.abs(val_preds - y_val)))
        r2 = float(1 - np.sum((val_preds - y_val)**2) / np.sum((y_val - np.mean(y_val))**2))
        logger.info(f"Huấn luyện XGBoost hoàn tất: MAE = {mae:.3f} kg, R2 = {r2:.3f}")

        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        self.model.save_model(self.model_path)
        logger.info(f"Đã lưu mô hình XGBoost tại: {self.model_path}")

    def predict(self, req: FeedPredictionRequest) -> FeedPredictionResult:
        if not self.is_loaded:
            self.load_model()

        # 1. Chuẩn bị vector 7 đặc trưng chuẩn
        feature_vector = np.array([[
            req.doc,
            req.avgWeightG,
            req.doMin,
            req.deltaTan,
            req.morningRainMm,
            req.prevFeedAmountKg,
            req.waterChangeM3,
        ]], dtype=np.float32)

        # 2. Dự báo bằng mô hình XGBoost
        raw_prediction = float(self.model.predict(feature_vector)[0])
        predicted_kg = max(0.0, round(raw_prediction, 2))

        # 3. Tính đóng góp thực tế của từng đặc trưng bằng XGBoost TreeSHAP (pred_contribs=True)
        dmatrix = xgb.DMatrix(feature_vector)
        shap_row = self.model.get_booster().predict(dmatrix, pred_contribs=True)[0]
        shap_values = [float(v) for v in shap_row[:len(self.feature_names)]]
        base_value_kg = round(float(shap_row[len(self.feature_names)]), 2)

        # 4. Sinh chuỗi giải thích & mảng đóng góp XAI chuẩn từ 7 đặc trưng và giá trị TreeSHAP thực tế
        explanation, feature_contributions = self._build_explanation_and_contributions(
            req, predicted_kg, base_value_kg, shap_values
        )

        return FeedPredictionResult(
            predictedFeedAmountKg=predicted_kg,
            baseValueKg=base_value_kg,
            modelName="XGBoost-Regressor",
            modelVersion=self.model_version,
            featureImportance=self._feature_importances,
            featureContributions=feature_contributions,
            explanation=explanation,
        )

    def _build_explanation_and_contributions(
        self,
        req: FeedPredictionRequest,
        predicted_kg: float,
        base_value_kg: float,
        shap_values: List[float],
    ) -> Tuple[str, List[FeatureContribution]]:
        feature_specs = [
            ("doc", "Ngày tuổi nuôi (DOC)", f"{req.doc} ngày"),
            ("avg_weight_g", "Trọng lượng bình quân (ABW)", f"{req.avgWeightG:.2f} g/con"),
            ("do_min", "Oxy hòa tan tối thiểu (DO min)", f"{req.doMin:.2f} mg/L"),
            ("delta_tan", "Biến thiên đạm Amoniac (ΔTAN)", f"{req.deltaTan:+.3f} mg/L"),
            ("morning_rain_mm", "Lượng mưa sáng sớm", f"{req.morningRainMm:.1f} mm"),
            ("prev_feed_kg", "Lượng ăn cữ liền trước", f"{req.prevFeedAmountKg:.2f} kg"),
            ("water_change_m3", "Lượng nước thay trong ngày", f"{req.waterChangeM3:.0f} m³"),
        ]

        contributions: List[FeatureContribution] = []
        for (feat_key, feat_label, value_str), raw_shap in zip(feature_specs, shap_values):
            delta_kg = round(raw_shap, 2)
            if delta_kg > 0.01:
                impact = "INCREASE"
                desc = f"{feat_label} ({value_str}) làm tăng dự báo khẩu phần +{delta_kg:.2f} kg theo mô hình XGBoost (TreeSHAP)."
            elif delta_kg < -0.01:
                impact = "DECREASE"
                desc = f"{feat_label} ({value_str}) làm giảm dự báo khẩu phần {delta_kg:.2f} kg theo mô hình XGBoost (TreeSHAP)."
            else:
                impact = "NEUTRAL"
                delta_kg = 0.0
                desc = f"{feat_label} ({value_str}) tác động không đáng kể (~0.00 kg) lên khẩu phần cữ này."

            contributions.append(
                FeatureContribution(
                    feature=feat_key,
                    name=feat_label,
                    value=value_str,
                    impact=impact,
                    deltaKg=delta_kg,
                    description=desc,
                )
            )

        # Tóm tắt giải thích dựa hoàn toàn trên 7 đặc trưng thực tế và giá trị TreeSHAP
        reasons: List[str] = []
        if req.doMin < 3.5:
            reasons.append(
                f"CẢNH BÁO NGUY CẤP: Oxy hòa tan ({req.doMin:.2f} mg/L) dưới ngưỡng an toàn (<3.5 mg/L)."
            )

        reasons.append(
            f"Tôm DOC {req.doc} ngày (ABW {req.avgWeightG:.2f} g/con), lượng ăn cữ trước {req.prevFeedAmountKg:.2f} kg."
        )

        # Nêu các yếu tố môi trường nổi bật đang tác động mạnh nhất (theo |deltaKg|)
        top_drivers = sorted(
            [c for c in contributions if c.impact != "NEUTRAL"],
            key=lambda c: abs(c.deltaKg),
            reverse=True,
        )[:3]
        if top_drivers:
            driver_summary = ", ".join(
                f"{c.name} ({c.deltaKg:+.2f} kg)" for c in top_drivers
            )
            reasons.append(f"Các yếu tố ảnh hưởng mạnh nhất (TreeSHAP từ mốc cơ sở {base_value_kg:.2f} kg): {driver_summary}.")

        reasons.append(f"Khuyến nghị XGBoost cấp phát: {predicted_kg:.2f} kg cho cữ ăn này.")
        return (" ".join(reasons), contributions)


# Thể hiện duy nhất (Singleton)
xgboost_feed_service = XgboostFeedService()
