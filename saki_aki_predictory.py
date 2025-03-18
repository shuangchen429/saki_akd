import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "X_9": {"type": "categorical", "options": [0, 1]},
    "X_39": {"type": "numerical", "min": 0.0, "max": 14417.0, "default": 5000.0},
    "X_32": {"type": "numerical", "min": 2.0, "max": 6.6, "default": 4.0},
    "X_34": {"type": "numerical", "min": 93.0, "max": 180.0, "default": 120.0},
    "X_30": {"type": "numerical", "min": 1.0, "max": 480.0, "default": 240.0},
    "X_28": {"type": "numerical", "min": 16.0, "max": 37.44, "default": 25.0},
    "X_46": {"type": "numerical", "min": 0.0, "max": 77.0, "default": 40.0},
    "X_31": {"type": "numerical", "min": 1.0, "max": 5.0, "default": 3.0},
    "X_3": {"type": "numerical", "min": 0.0, "max": 170.0, "default": 85.0},
    "X_36": {"type": "numerical", "min": 17.0, "max": 7768.0, "default": 2000.0},
    "X_33": {"type": "numerical", "min": 10.0, "max": 45.6, "default": 30.0},
    "X_44": {"type": "numerical", "min": 1.0, "max": 10.0, "default": 5.0},
    "X_16": {"type": "categorical", "options": [0, 1]},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
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

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
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

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
