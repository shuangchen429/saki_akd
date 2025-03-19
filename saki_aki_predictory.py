import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载逻辑回归模型
model = joblib.load('pipeline.pkl')

# 定义特征参数（包含单位和范围）
feature_ranges = {
    # 分类特征 (直接显示0/1)
    'ACEI/ARB': {"type": "categorical", "options": [0, 1]},
    'APS III': {"type": "numerical", "min": 0, "max": 215, "default": 0, "unit": "points"},
    'Age': {"type": "numerical", "min": 18, "max": 120, "default": 50, "unit": "years"},
    'Baseexcess': {"type": "numerical", "min": -25, "max": 30, "default": 0, "unit": "mmol/L"},
    'Bun': {"type": "numerical", "min": 1, "max": 100, "default": 20, "unit": "mg/dL"},
    'CRRT': {"type": "categorical", "options": [0, 1]},
    'Cerebrovascular_disease': {"type": "categorical", "options": [0, 1]},
    'Glucose': {"type": "numerical", "min": 1.5, "max": 50.0, "default": 5.5, "unit": "mmol/L"},
    'LODS': {"type": "numerical", "min": 0, "max": 22, "default": 0, "unit": "points"},
    'Los_inf._AB': {"type": "numerical", "min": 0, "max": 7, "default": 0, "unit": "days"},
    'OASIS': {"type": "numerical", "min": 0, "max": 299, "default": 0, "unit": "points"},
    'Pco2': {"type": "numerical", "min": 10, "max": 150, "default": 40, "unit": "mmHg"},
    'Po2': {"type": "numerical", "min": 20, "max": 700, "default": 100, "unit": "mmHg"},
    'Resp_rate': {"type": "numerical", "min": 0, "max": 50, "default": 18, "unit": "breaths/min"},
    'Scr_baseline': {"type": "numerical", "min": 0, "max": 5000, "default": 60, "unit": "mmol/L"},
    'Sodium': {"type": "numerical", "min": 110, "max": 170, "default": 140, "unit": "mmol/L"},
    'Temperature': {"type": "numerical", "min": 32.0, "max": 42.0, "default": 36.6, "unit": "°C"},
    'Vasoactive_agent': {"type": "categorical", "options": [0, 1]},
    'WBC': {"type": "numerical", "min": 0.0, "max": 50.0, "default": 8.0, "unit": "×10^9/L"},
    'Weight': {"type": "numerical", "min": 30, "max": 200, "default": 60, "unit": "kg"}
}

# 设置页面标题
st.title("AKD Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']}){properties['unit']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)


# 转换为模型输入格式
features = np.array([feature_values])

if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]


    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKD is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")
