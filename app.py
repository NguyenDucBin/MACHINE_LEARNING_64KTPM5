import streamlit as st
import numpy as np
from joblib import load

# Tải các mô hình đã lưu
bagging_model = load('linear_model.joblib')
ridge_bagging_model = load('ridge_model.joblib')
mlp_bagging_model = load('neural_net_model.joblib')

# Tạo giao diện ứng dụng
st.title("Dự đoán giá cổ phiếu Vinamilk")

# Yêu cầu người dùng nhập dữ liệu
st.write("Nhập các yếu tố ảnh hưởng đến giá cổ phiếu:")
year = st.number_input("Năm:", value=2023, format='%d')
open_price = st.number_input("Giá mở cửa:", value=0.0)
close_price = st.number_input("Giá đóng cửa:", value=0.0)
high = st.number_input("Giá cao nhất:", value=0.0)
low = st.number_input("Giá thấp nhất:", value=0.0)

# Chuyển đổi dữ liệu đầu vào thành mảng numpy
user_input = np.array([[open_price, close_price, high, low, year]])

# Chọn mô hình dự đoán
model_option = st.selectbox("Chọn mô hình dự đoán:", 
                            ('Linear Regression', 'Ridge Regression', 'Neural Network'))

# Dự đoán
if st.button("Dự đoán"):
    if model_option == 'Linear Regression':
        prediction = bagging_model.predict(user_input)
    elif model_option == 'Ridge Regression':
        prediction = ridge_bagging_model.predict(user_input)
    elif model_option == 'Neural Network':
        prediction = mlp_bagging_model.predict(user_input)
    
    # Hiển thị kết quả dự đoán
    st.write(f"Giá cổ phiếu Vinamilk dự đoán: ${prediction[0]:,.2f}")
