import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 设置页面配置
st.set_page_config(page_title="AKD Prediction Model", layout="wide")

# 尝试加载模型
try:
    model = joblib.load('0511重置版saki_lr_model1.pkl')
    scaler = joblib.load('0511重置版saki_scaler.pkl')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# 定义特征参数
feature_ranges = {
    'ACEI/ARB': {"type": "categorical", "options": [0, 1]},
    'APS III': {"type": "numerical", "min": 0, "max": 215, "default": 0, "unit": "points"},
    'CRRT': {"type": "categorical", "options": [0, 1]},
    'Cerebrovascular Disease': {"type": "categorical", "options": [0, 1]},
    'LODS': {"type": "numerical", "min": 0, "max": 22, "default": 0, "unit": "points"},
    'Los_inf._AB': {"type": "numerical", "min": 0, "max": 7, "default": 0, "unit": "hours"},
    'MBP': {"type": "numerical", "min": 0, "max": 200, "default": 60, "unit": "mmHg"},
    'Mechanical Ventilation': {"type": "categorical", "options": [0, 1]},
    'Paraplegia': {"type": "categorical", "options": [0, 1]},
    'Resp Rate': {"type": "numerical", "min": 0, "max": 50, "default": 18, "unit": "breaths/min"},
    'Scr Baseline': {"type": "numerical", "min": 0, "max": 5000, "default": 60, "unit": "mmol/L"},
    'SpO2': {"type": "numerical", "min": 0, "max": 100, "default": 100, "unit": "%"},
    'Vasoactive Agent': {"type": "categorical", "options": [0, 1]},
    'Weight': {"type": "numerical", "min": 30, "max": 200, "default": 60, "unit": "kg"}
}

# 页面布局
st.title("AKD Prediction Model")
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Parameters")
    feature_values = []
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['unit']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=feature,
                help=f"Range: {properties['min']}-{properties['max']} {properties['unit']}"
            )
        elif properties["type"] == "categorical":
            value = st.selectbox(
                label=feature,
                options=properties["options"],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key=feature
            )
        feature_values.append(value)

with col2:
    st.header("Prediction Results")
    if st.button("Predict AKD Risk", help="Click to calculate AKD risk probability"):
        try:
            # 转换为模型输入格式
            features = np.array([feature_values])
            scaled_features = scaler.transform(features)
            
            # 模型预测
            predicted_proba = model.predict_proba(scaled_features)[0][1]
            probability = predicted_proba * 100
            
            # 显示结果
            st.subheader("Prediction Result")
            st.markdown(f'<p class="big-font">AKD Risk Probability: <b>{probability:.2f}%</b></p>', 
                       unsafe_allow_html=True)
            
            # 添加进度条可视化
            st.progress(int(probability))
            
            # 添加解释性文本
            if probability > 50:
                st.warning("High risk of AKD - Consider close monitoring and intervention")
            else:
                st.success("Lower risk of AKD - Continue standard monitoring")
            
            # 可以添加特征重要性解释（如果有）
            # st.subheader("Key Contributing Factors")
            # 这里可以添加特征重要性分析
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# 可以添加页脚
st.markdown("---")
st.caption("Note: This prediction tool is for clinical reference only and should not replace clinical judgment.")
