# 🏠 Machine Learning: House Price Prediction
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

Dự án này sử dụng các kỹ thuật học máy (Machine Learning) để dự đoán giá nhà dựa trên các đặc điểm về vị trí, diện tích và cơ sở vật chất. Đây là sự kết hợp giữa kiến thức thống kê ứng dụng (**MAS291**) và quy trình xây dựng mô hình thực tế.

---

## 📊 1. Quy trình xử lý dữ liệu (EDA & Preprocessing)
Áp dụng các nguyên lý thống kê để làm sạch và hiểu dữ liệu:

* **Phân loại biến (Data Classification):** Xác định các biến định danh (Nominal) như khu vực, hướng nhà và biến liên tục (Continuous) như diện tích, giá.
* **Thống kê mô tả (Descriptive Statistics):** Sử dụng biểu đồ Boxplot để xác định Outliers theo quy tắc $1.5 \times IQR$.
* **Tương quan (Correlation):** Tính toán hệ số $r$ để loại bỏ các đặc trưng không ảnh hưởng đến giá nhà (Feature Selection).
* **Chuẩn hóa (Scaling):** Áp dụng $Z-score$ chuẩn hóa ($Z = \frac{X - \mu}{\sigma}$) để đưa các biến về cùng một thang đo, giúp Gradient Descent hội tụ nhanh hơn.

---

## 🤖 2. Xây dựng mô hình (Modeling)
Dự án tập trung vào các thuật toán hồi quy:

1.  **Linear Regression:** Mô hình cơ sở (Baseline) sử dụng phương pháp Bình phương tối thiểu (OLS) để tối thiểu hóa $SSE$.
2.  **Định lý Bayes & Xác suất:** Áp dụng để hiểu về phân phối xác suất của sai số và các biến độc lập.
3.  **Định lý giới hạn trung tâm (CLT):** Đảm bảo tính ổn định của dự báo khi kích thước mẫu $n \ge 30$.



---

## 📈 3. Đánh giá kết quả (Evaluation)
Đánh giá độ chính xác và độ tin cậy của mô hình:

* **Hệ số xác định ($R^2$):** Đo lường mức độ giải thích của mô hình đối với sự biến động của giá nhà.
* **Khoảng tin cậy (Confidence Interval):** Đưa ra dự báo giá nhà trong một khoảng giá trị thay vì một con số duy nhất để giảm thiểu rủi ro.
* **P-value:** Kiểm tra ý nghĩa thống kê để xác nhận các biến như "Số phòng ngủ" hay "Diện tích" thực sự có tác động đến giá.

---

## 📂 Cấu trúc thư mục
```text
.
├── data/               # Chứa file csv (Raw & Cleaned)
├── notebooks/          # Jupyter notebooks cho EDA và Model Testing
├── src/                # Mã nguồn chính (train.py, preprocess.py)
├── models/             # Lưu trữ các model đã huấn luyện (.pkl)
└── README.md
