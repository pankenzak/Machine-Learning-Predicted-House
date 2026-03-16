"# MAS291" 
"# Machine-Learning-Predicted-House" 
# 📈 Applied Statistics (MAS291) in Machine Learning Workflow

Tài liệu này tóm tắt cách áp dụng các kiến thức trong giáo trình MAS291 vào các giai đoạn cốt lõi của một dự án Machine Learning (ML), từ tiền xử lý dữ liệu đến đánh giá mô hình.

---

## 1. Exploratory Data Analysis (EDA) & Preprocessing
*Giai đoạn "lắng nghe dữ liệu" và làm sạch trước khi huấn luyện mô hình.*

* **Data Classification (Chương 1):** Phân loại biến đầu vào thành **Nominal** (danh nghĩa), **Ordinal** (thứ bậc) hoặc **Continuous** (liên tục) để lựa chọn phương pháp mã hóa (Encoding) như One-Hot Encoding hoặc Label Encoding phù hợp.
* **Descriptive Statistics (Chương 6):**
    * Sử dụng **Mean** (trung bình) và **Median** (trung vị) để hiểu về trọng tâm và độ lệch của phân phối dữ liệu.
    * Áp dụng **Boxplot** và quy tắc $1.5 \times IQR$ để phát hiện **Outliers** (giá trị ngoại lai) — những điểm dữ liệu bất thường có thể gây nhiễu và làm sai lệch mô hình.
* **Correlation (Chương 5):** Sử dụng hệ số tương quan ($r$) để đo lường mức độ "đi cùng nhau" của các biến, hỗ trợ giai đoạn **Feature Selection** nhằm tìm ra các biến ảnh hưởng mạnh nhất đến mục tiêu (ví dụ: giá nhà).
* **Standardization (Chương 4):** Kỹ thuật Feature Scaling sử dụng công thức chuẩn hóa $Z = \frac{X - \mu}{\sigma}$ để đưa các đặc trưng về cùng một đơn vị, giúp các thuật toán tối ưu (như Gradient Descent) hội tụ nhanh hơn và ổn định hơn.

---

## 2. Modeling Stage
*Sử dụng dữ liệu mẫu (Sample) để suy diễn và tìm ra quy luật của quần thể (Population).*

* **Linear Regression (Chương 10 & 11):** Thuật toán ML cơ bản nhất dựa trên phương pháp **Bình phương tối thiểu (Least Squares)** để tìm ra đường thẳng dự báo có sai số ($SSE$) nhỏ nhất so với các điểm dữ liệu thực tế.
* **Probability & Bayes' Theorem (Chương 2):** Định lý Bayes cung cấp nền tảng toán học cho các mô hình phân loại xác suất như **Naive Bayes**.
* **Probability Distributions (Chương 3 & 4):** Các giả định về phân phối (như giá nhà thường tuân theo **Normal Distribution**) giúp mô hình hóa sự ngẫu nhiên và cải thiện khả năng dự báo.
* **Central Limit Theorem (Chương 7):** Giải thích lý do vì sao khi kích thước mẫu đủ lớn ($n \ge 30$), sai số của mô hình thường hội tụ về phân phối chuẩn, giúp kết quả dự báo trở nên ổn định và đáng tin cậy hơn.

---

## 3. Evaluation & Output Analysis
*Kiểm chứng độ tin cậy của các con số đầu ra từ mô hình ML.*

* **Interval Estimation (Chương 8):** Thay vì chỉ cung cấp một con số dự báo duy nhất (Point Estimate), ta có thể đưa ra **Khoảng tin cậy (Confidence Interval)** với độ tin cậy (ví dụ 95%) để cung cấp cái nhìn thực tế hơn về rủi ro và sai số.
* **Hypothesis Testing (Chương 9):**
    * Sử dụng **P-value** để kiểm tra ý nghĩa thống kê của các biến độc lập, xác định xem một đặc trưng thực sự có tác động đến kết quả hay chỉ là do ngẫu nhiên.
    * So sánh hiệu năng giữa các thuật toán (ví dụ: so sánh độ chính xác của hai mô hình khác nhau) để kết luận sự vượt trội có mang tính hệ thống hay không.
* **Goodness of Fit (Chương 10.3):** Sử dụng hệ số xác định $R^2$ (**Coefficient of Determination**) để đo lường tỷ lệ phần trăm sự biến động của mục tiêu được giải thích bởi mô hình, từ đó đánh giá chất lượng tổng thể của mô hình.
