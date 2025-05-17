import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载逻辑回归模型
model = joblib.load('0511重置版saki_lr_model1.pkl')
scaler = joblib.load('0511重置版saki_scaler.pkl')

# 定义特征参数（包含单位和范围）
feature_ranges = {
    # 分类特征 (直接显示0/1)'ACEI/ARB', 'APS III' ,'CRRT', 'Cerebrovascular Disease', 'LODS',
                'Los_inf._AB', 'MBP', 'Mechanical Ventilation', 'Paraplegia', 'Resp Rate',
                'Scr Baseline' ,'SpO2', 'Vasoactive Agent', 'Weight'
    
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

# 设置页面标题
st.title("AKD Prediction Model")

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
features1 = np.array([feature_values])
features= scaler.transform(features1)
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]


    # 提取预测的类别概率
    probability = predicted_proba[1] * 100

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
