# Airbnb Price Prediction - End-to-End ML Project 🏠💰
# Noi dung tu nhanh CONFLICT-TEST
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
##JavaScriptCourse
## 📌 Giới thiệu (Introduction)

Dự án này là một ứng dụng **End-to-End Machine Learning** hoàn chỉnh giúp dự đoán giá thuê căn hộ Airbnb dựa trên các thuộc tính như vị trí, loại phòng, tiện ích và số lượng người ở. Ứng dụng được xây dựng trên bộ dữ liệu thực tế từ **Inside Airbnb** (74,000+ bản ghi), sử dụng các kỹ thuật xử lý dữ liệu nâng cao và mô hình thuật toán tối ưu.

### ✨ Điểm nổi bật (Highlights)

* **Mô hình chính xác cao**: Sử dụng **Gradient Boosting Regressor** với R2 Score ~0.76 và RMSE ~0.35.
* **Giao diện Web thân thiện**: Viết bằng **Flask**, hỗ trợ Tiếng Việt, có chế độ "Simple Mode" cho người dùng phổ thông.
* **Tính năng thông minh**:
  * 🕒 **Lịch sử dự đoán**: Tự động lưu và hiển thị các lần tra cứu gần nhất (SQLite).
  * 🔍 **So sánh thực tế**: Gợi ý 5 căn hộ tương tự đang hoạt động để người dùng tham chiếu giá.

---

## 🚀 Cài đặt và Chạy dự án (Installation)

### 1. Clone dự án

```bash
git clone https://github.com/L4mL4mmm/AirBnBApps.git
cd AirBnBApps
```

### 2. Cài đặt thư viện

Khuyên dùng môi trường ảo (Virtual Environment):

```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
# Kích hoạt (Windows)
.\venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng

```bash
python app.py
```

Sau đó mở trình duyệt và truy cập: **<http://127.0.0.1:8080>**

---

## 📊 Quy trình Xử lý & Mô hình (Model Pipeline)

Hệ thống tuân theo quy trình MLOps chuẩn:

1. **Data Ingestion**: Tải dữ liệu, chia tập Train (60%) - Val (20%) - Test (20%).
2. **Data Transformation**:
    * Xử lý Missing Value (Imputation).
    * Loại bỏ Outliers (Cắt bỏ 1% giá cao nhất).
    * Mã hóa (Ordinal Encoding & Standard Scaling).
3. **Model Training**:
    * Thử nghiệm: Linear Regression, Random Forest, Gradient Boosting.
    * Tối ưu hóa: Sử dụng **GridSearchCV**.
    * **Kết quả tốt nhất**: Gradient Boosting (R2: 0.76).

---

## 📂 Cấu trúc thư mục (Folder Structure)

```
AirBnBApps/
├── Artifacts/           # Chứa Model, Preprocessor và Data
├── logs/                # Nhật ký chạy hệ thống
├── src/
│   └── Airbnb/
│       ├── components/  # Các module xử lý chính (Ingestion, Transformation, Trainer)
│       ├── pipelines/   # Pipeline huấn luyện và dự đoán
│       └── utils/       # Các hàm tiện ích (Save/Load object)
├── templates/           # Giao diện HTML
├── app.py               # File chạy chính (Flask App)
├── requirements.txt     # Danh sách thư viện
└── README.md            # Tài liệu hướng dẫn
```

## 👨‍💻 Tác giả (Author)

* **L4mL4mmm** -

---
*Dự án phục vụ môn học Project 2.*
