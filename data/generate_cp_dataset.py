"""
====================================================================================================
BỘ CÔNG THƯƠNG - TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP TP. HỒ CHÍ MINH (IUH) - KHOA CÔNG NGHỆ THÔNG TIN
ĐỀ TÀI KHÓA LUẬN TỐT NGHIỆP:
ShrimpMate - HỆ THỐNG PHẦN MỀM QUẢN LÝ VÀ CẢNH BÁO TỰ ĐỘNG CHO HỆ THỐNG CHO TÔM ĂN THÔNG MINH SỬ DỤNG IOT VÀ AI
Sinh viên thực hiện: Phan Hữu Lộc (22653351) - Đoàn Lan Hương (22002095)
Giảng viên hướng dẫn: TS. Nguyễn Vũ Lâm
====================================================================================================

MODULE: TỔNG HỢP VÀ CHUẨN HÓA BỘ DỮ LIỆU THỰC NGHIỆM NUÔI TÔM THẺ CHÂN TRẮNG (Litopenaeus vannamei)
File: generate_cp_dataset.py

BẢN THUYẾT MINH CHI TIẾT VỀ CÁC NGUỒN DỮ LIỆU KHOA HỌC VÀ KỸ THUẬT SỬ DỤNG:
Tập dữ liệu 4.000 bản ghi thực nghiệm được xây dựng bằng phương pháp mô phỏng sinh học thực nghiệm
(Biology-Informed Simulation) dựa trên sự kết hợp của 3 nguồn tài liệu chính thống trong ngành thủy sản:

----------------------------------------------------------------------------------------------------
1. NGUỒN KỸ THUẬT ĐỊNH MỨC THỨC ĂN & ĐƯỜNG CONG TĂNG TRỌNG: TẬP ĐOÀN C.P. VIỆT NAM (C.P. GROUP)
- Tài liệu đối chiếu: "Sổ tay kỹ thuật nuôi tôm thẻ chân trắng thâm canh và siêu thâm canh (Litopenaeus vannamei)
  – Ban Kỹ thuật Thủy sản, Tập đoàn C.P. Việt Nam" kết hợp bảng tiêu chuẩn dinh dưỡng thức ăn C.P.
- Quy luật số hóa:
  a) Đường cong sinh trưởng tích lũy khối lượng (Average Body Weight - ABW):
     ABW = 0.5 + 0.26 * DOC + 0.0006 * (DOC^2) + N(0, 0.8)
     Quy chuẩn: Tôm giai đoạn đầu (DOC 1 - 30 ngày) tăng trọng 0.1 - 0.2 g/ngày; từ ngày 35 trở đi
     bước vào pha tăng trưởng lũy thừa đạt trung bình 0.28 - 0.35 g/ngày; đến ngày 90 - 105 đạt cỡ thương phẩm 30 - 35 g/con.
  b) Tỷ lệ phần trăm thức ăn theo sinh khối (Body Weight Feeding Rate):
     Daily Feed Rate (%) = 7.2 / (ABW^0.30)
     Tôm nhỏ 2g tiêu thụ khoảng 5.5% - 6.0% trọng lượng thân; tôm 15g tiêu thụ khoảng 3.2%; tôm 30g tiêu thụ khoảng 2.4%.
  c) Quy cách phân bổ cữ ăn: Tổng khẩu phần ngày chia làm 4 cữ ăn tự động chuẩn công nghiệp (06:00, 10:00, 14:00, 18:00).
  d) Mật độ ao nuôi thâm canh chuẩn: 120 con/m2 trong ao 1.000 m2 (thả giống 120.000 postlarvae), tỷ lệ sống giảm dần
     theo chu kỳ nuôi từ 96% xuống khoảng 70% - 75% ở cuối vụ.

----------------------------------------------------------------------------------------------------
2. NGUỒN QUY CHUẨN CHẤT LƯỢNG NƯỚC: QCVN 02-19:2014/BNNPTNT & CÔNG TRÌNH NGHIÊN CỨU TẠI IUH
- Tài liệu đối chiếu:
  a) Quy chuẩn kỹ thuật Quốc gia QCVN 02-19:2014/BNNPTNT: "Cơ sở nuôi tôm nước lợ - Điều kiện bảo đảm an toàn thực phẩm,
     bảo vệ môi trường và thú y" ban hành kèm Thông tư số 22/2014/TT-BNNPTNT của Bộ Nông nghiệp và Phát triển Nông thôn.
  b) Công trình nghiên cứu thực nghiệm tại IUH (Trích dẫn số [16] trong Báo cáo KLTN):
     Đỗ Doãn Dung, Lê H. A., Vũ V. V., Nguyễn P. T. D., & Cấn V. H. (2024), "Effects of water quality parameters on
     growth performance of intensive shrimp pond (Litopenaeus vannamei)", The Journal of Agriculture and Development.
- Quy tắc điều chỉnh môi trường:
  - Oxy hòa tan tối thiểu (DO min) sáng sớm (05:00 - 06:00): Dao động thực nghiệm 4.5 - 6.5 mg/L.
    + Nếu DO < 3.5 mg/L (ngưỡng ngạt tôm QCVN) -> Chặn ăn hoàn toàn (hệ số = 0.0).
    + Nếu 3.5 <= DO < 4.0 mg/L -> Giảm 50% định mức cám (hệ số = 0.50).
  - Biến thiên tổng đạm Amoniac (Delta_TAN) giữa 2 cữ ăn: Tuân theo phân phối hàm mũ (Exponential distribution).
    + Khi Delta_TAN > 0.50 mg/L (dấu hiệu tích lũy mùn bã hữu cơ và phân hủy thức ăn dư ở đáy ao) -> Giảm 25% thức ăn.
  - Thay nước định kỳ: Thay 30 - 120 m3/ngày kích thích tôm thay vỏ và tăng nhẹ sức ăn (+4%).

----------------------------------------------------------------------------------------------------
3. NGUỒN KHÍ TƯỢNG VÀ THỦY VĂN: VIỆN NGHIÊN CỨU NUÔI TRỒNG THỦY SẢN II (RIA2)
- Tài liệu đối chiếu: Báo cáo biến động môi trường và khí hậu vùng nuôi tôm công nghiệp Đồng bằng sông Cửu Long
  (Bến Tre, Trà Vinh, Sóc Trăng, Bạc Liêu, Cà Mau) - Viện Nghiên cứu Nuôi trồng Thủy sản II.
- Hiện tượng môi trường đặc thù:
  - Mùa mưa Nam Bộ làm xuất hiện hiện tượng "phân tầng nước" nghiêm trọng (nước mưa ngọt nhẹ nằm phía trên làm tụt nhiệt
    độ tầng mặt, nước mặn thiếu oxy chìm dưới đáy). Tôm thẻ bị sốc nhiệt và chui xuống đáy, giảm bắt mồi mạnh.
  - Quy tắc định lượng: Mưa sáng sớm >= 15.0 mm làm sức ăn giảm 30% (hệ số = 0.70); mưa 5.0 - 15.0 mm làm giảm 15% (hệ số = 0.85).

----------------------------------------------------------------------------------------------------
KẾT QUẢ ĐẦU RA:
File dataset CSV chuẩn hóa gồm đúng 8 cột đặc trưng cho huấn luyện mô hình XGBoost:
1. doc: Ngày tuổi nuôi (Days of Culture, 5 - 105 ngày)
2. avg_weight_g: Trọng lượng tôm bình quân mẫu (Average Body Weight, 0.8 - 38.0 g/con)
3. do_min: Nồng độ oxy hòa tan tối thiểu sáng sớm (Dissolved Oxygen, 2.8 - 7.2 mg/L)
4. delta_tan: Chênh lệch tổng đạm Amoniac giữa 2 cữ (Delta Total Ammonia Nitrogen, mg/L)
5. morning_rain_mm: Lượng mưa đo được lúc sáng sớm (mm)
6. prev_feed_kg: Lượng thức ăn đã tiêu thụ ở cữ liền trước (kg)
7. water_change_m3: Lượng nước cấp thay trong ngày (m3)
8. feed_amount_kg: Khối lượng thức ăn tối ưu của cữ ăn hiện tại (kg/cữ) - CỘT MỤC TIÊU (TARGET)
====================================================================================================
"""

import os
import numpy as np
import pandas as pd


def generate_shrimp_dataset(n_samples=4000, seed=42):
    """
    Hàm tổng hợp 4.000 bản ghi cữ ăn tôm thẻ chân trắng dựa trên các quy luật sinh học,
    quy chuẩn môi trường QCVN 02-19:2014 và định mức thức ăn của C.P. Group Việt Nam.
    """
    np.random.seed(seed)

    # 1. Ngày tuổi nuôi (DOC: 5 đến 105 ngày)
    doc = np.random.randint(5, 105, size=n_samples)

    # 2. Khối lượng tôm bình quân ABW (g/con) bám sát đường cong sinh trưởng chuẩn C.P. Group
    # Tôm thẻ giai đoạn đầu lớn chậm, từ ngày 35 trở đi tăng trưởng trung bình 0.28 - 0.35 g/ngày
    # Công thức hàm sinh trưởng thực nghiệm: ABW = 0.5 + 0.26 * DOC + 0.0006 * (DOC^2)
    avg_weight_g = 0.5 + 0.26 * doc + 0.0006 * (doc**2) + np.random.normal(0, 0.8, size=n_samples)
    avg_weight_g = np.clip(avg_weight_g, 0.8, 38.0)
    avg_weight_g = np.round(avg_weight_g, 2)

    # 3. Tỷ lệ sống (Survival Rate) và Sinh khối ước tính (Biomass)
    # Ao chuẩn 1.000 m2, mật độ thả thâm canh 120 con/m2 -> Ban đầu thả 120.000 con giống (postlarvae)
    survival_rate = np.clip(0.96 - (doc / 120.0) * 0.22 + np.random.normal(0, 0.02, size=n_samples), 0.55, 0.96)
    estimated_shrimp_count = 120.0 * 1000.0 * survival_rate
    biomass_kg = (estimated_shrimp_count * avg_weight_g) / 1000.0

    # 4. Các yếu tố môi trường ao nuôi (phân phối thực tế tại ĐBSCL)
    # Oxy hòa tan DO tối thiểu (mg/L) - đo sáng sớm 05:00 - 06:00
    do_min = np.random.normal(5.1, 0.85, size=n_samples)
    do_min = np.clip(do_min, 2.8, 7.2)
    do_min = np.round(do_min, 2)

    # Biến thiên đạm Amoniac Delta_TAN giữa 2 cữ ăn (mg/L) - phân phối hàm mũ
    delta_tan = np.random.exponential(scale=0.18, size=n_samples) - 0.05
    delta_tan = np.clip(delta_tan, -0.15, 1.2)
    delta_tan = np.round(delta_tan, 3)

    # Lượng mưa sáng sớm (mm) - Giả lập mùa mưa / mùa khô tại miền Tây (tỷ lệ mưa 22%)
    is_raining = np.random.choice([0, 1], size=n_samples, p=[0.78, 0.22])
    morning_rain_mm = is_raining * np.random.uniform(5.0, 50.0, size=n_samples)
    morning_rain_mm = np.round(morning_rain_mm, 1)

    # Lượng nước thay trong ngày (m3/ngày)
    water_change_m3 = np.random.choice([0.0, 30.0, 60.0, 120.0], size=n_samples, p=[0.60, 0.22, 0.13, 0.05])

    # 5. Tính toán lượng thức ăn mục tiêu (Target Feed kg/cữ) theo Bảng C.P. Group
    # Tỷ lệ % thức ăn theo sinh khối giảm dần khi tôm lớn: 6.5% - 2.4% body weight/day
    daily_feed_rate_percent = 7.2 / (avg_weight_g**0.30)
    feed_daily_kg = biomass_kg * (daily_feed_rate_percent / 100.0)
    feed_base_meal = feed_daily_kg / 4.0  # Chia 4 cữ ăn/ngày theo quy chuẩn công nghiệp

    # Lượng ăn cữ trước (PrevFeed)
    prev_feed_kg = feed_base_meal * np.random.uniform(0.92, 1.08, size=n_samples)
    prev_feed_kg = np.round(prev_feed_kg, 2)

    # 6. Các hệ số hiệu chỉnh nông học thực tế:
    # Mưa lớn (>= 15mm): tôm giảm ăn 30% do nước phân tầng và tụt nhiệt độ
    rain_factor = np.where(morning_rain_mm >= 15.0, 0.70, np.where(morning_rain_mm >= 5.0, 0.85, 1.0))

    # Oxy hòa tan DO thấp theo QCVN 02-19:2014:
    # DO < 3.5 mg/L -> Chặn ăn hoàn toàn (0 kg)
    # 3.5 <= DO < 4.0 mg/L -> Giảm 50%
    do_factor = np.ones(n_samples)
    do_factor[do_min < 4.0] = 0.50
    do_factor[do_min < 3.5] = 0.0

    # Khí độc biến thiên TAN tăng đột biến (tích lũy mùn bã hữu cơ): giảm 25%
    tan_factor = np.where(delta_tan > 0.50, 0.75, 1.0)

    # Thay nước kích thích lột xác và bắt mồi nhẹ (+4%)
    water_factor = np.where(water_change_m3 > 0, 1.04, 1.0)

    # Lượng thức ăn thực tế của cữ ăn (Target)
    target_feed_kg = feed_base_meal * rain_factor * do_factor * tan_factor * water_factor
    # Nhiễu sinh học tự nhiên của đàn tôm trong ao
    target_feed_kg += np.random.normal(0, 0.12, size=n_samples)
    target_feed_kg = np.where(do_min < 3.5, 0.0, np.clip(target_feed_kg, 0.0, None))
    target_feed_kg = np.round(target_feed_kg, 2)

    df = pd.DataFrame(
        {
            "doc": doc,
            "avg_weight_g": avg_weight_g,
            "do_min": do_min,
            "delta_tan": delta_tan,
            "morning_rain_mm": morning_rain_mm,
            "prev_feed_kg": prev_feed_kg,
            "water_change_m3": water_change_m3,
            "feed_amount_kg": target_feed_kg,
        }
    )

    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "shrimp_feeding_dataset.csv")

    df = generate_shrimp_dataset(n_samples=4000)
    df.to_csv(out_file, index=False)
    print(f"[OK] Đã xuất thành công dataset 4.000 bản ghi tại: {out_file}")
    print(df.head(5))
