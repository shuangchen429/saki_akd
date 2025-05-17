import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shap
import tempfile

# 设置页面配置
st.set_page_config(page_title="AKD Prediction Model", layout="wide")

# 标题栏容器
header_container = st.container()
with header_container:
    cols = st.columns([0.2, 0.8])
    with cols[0]:
        logo = Image.open("东华医院图标.png")
        st.image(logo, use_column_width=True)
    with cols[1]:
        st.title("AKD Prediction Model")
        st.markdown("""
            <div style='border-left: 5px solid #1A5276; padding-left: 15px;'>
            <h4 style='color: #2E86C1;'>Clinical Decision Support System</h4>
            <p style='font-size:16px;'>Emergency Department, Dongguan Tungwah Hospital</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")  # 添加分割线


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
    'Los_inf._AB': {"type": "numerical", "min": 0, "max": 168, "default": 0, "unit": "hours"},
    'MBP': {"type": "numerical", "min": 0, "max": 300, "default": 60, "unit": "mmHg"},
    'Mechanical Ventilation': {"type": "categorical", "options": [0, 1]},
    'Paraplegia': {"type": "categorical", "options": [0, 1]},
    'Resp Rate': {"type": "numerical", "min": 0, "max": 80, "default": 20, "unit": "breaths/min"},
    'Scr Baseline': {"type": "numerical", "min": 0, "max": 2000, "default": 60, "unit": "mmol/L"},
    'SpO2': {"type": "numerical", "min": 0, "max": 100, "default": 100, "unit": "%"},
    'Vasoactive Agent': {"type": "categorical", "options": [0, 1]},
    'Weight': {"type": "numerical", "min": 0, "max": 500, "default": 60, "unit": "kg"}
}

# 定义模型训练时的特征顺序
feature_order = [
    'ACEI/ARB', 'APS III', 'CRRT', 'Cerebrovascular Disease', 'LODS', 'Los_inf._AB',
    'MBP', 'Mechanical Ventilation', 'Paraplegia', 'Resp Rate', 'Scr Baseline', 'SpO2',
    'Vasoactive Agent', 'Weight'
]

# 页面布局
st.title("AKD Prediction Model")
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    .prediction-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
        margin-top: 20px;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Parameters")
    input_values = {}  # 使用字典临时存储特征值
    
    # 分组显示参数
    groups = {
        "Demographics": ['Weight', 'Scr Baseline'],
        "Clinical Conditions": ['Cerebrovascular Disease', 'Paraplegia'],
        "Vitals & Measurements": ['MBP', 'Resp Rate', 'SpO2', 'Los_inf._AB'],
        "Treatment & Support": ['ACEI/ARB', 'CRRT', 'Mechanical Ventilation', 'Vasoactive Agent'],
        "Scoring Systems": ['APS III', 'LODS']
    }
    
    for group_name, features in groups.items():
        with st.expander(group_name):
            for feature in features:
                properties = feature_ranges[feature]
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
                input_values[feature] = value  # 按特征名存储到字典
    
    # 按feature_order顺序生成特征列表
    feature_values = [input_values[feature] for feature in feature_order]

with col2:
    st.header("Prediction Results")
    st.markdown("""
        <div class="prediction-box">
            <p>This section will display the AKD risk probability after you click the predict button.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 修改后的预测按钮部分
if st.button("Predict AKD Risk", help="Click to calculate AKD risk probability"):
    try:
        # ==================== 特征处理 ==================== 
        # 确保按模型训练顺序获取特征值
        ordered_feature_values = [input_values[feature] for feature in feature_order]
        
        # 转换为DataFrame并标准化
        input_df = pd.DataFrame([ordered_feature_values], columns=feature_order)
        scaled_features = scaler.transform(input_df)

        # ==================== 模型预测 ==================== 
        predicted_proba = model.predict_proba(scaled_features)[0][1]
        probability = predicted_proba * 100

        # ==================== 显示结果 ==================== 
        st.subheader("Prediction Result")
        st.markdown(f'<p class="big-font">AKD Risk Probability: <b>{probability:.2f}%</b></p>', 
                   unsafe_allow_html=True)
        st.progress(int(probability))

        # ==================== SHAP分析 ==================== 
        st.subheader("Model Interpretation")
        
        # 初始化SHAP解释器
        explainer = shap.LinearExplainer(model, scaler.transform(pd.DataFrame(columns=feature_order).fillna(0))
        
        # 计算SHAP值
        shap_values = explainer.shap_values(scaled_features)
        
        # 生成可视化
        plt.figure(figsize=(12, 6))
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values,
            features=input_df,
            feature_names=feature_order,
            matplotlib=True,
            show=False
        )
        
        # 保存并显示图像
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
            plt.close()
            st.image(tmpfile.name, caption="SHAP Feature Impact Analysis")

    except Exception as e:
        st.error(f"系统错误: {str(e)}")
        st.stop()  # 严重错误时停止执行

# 可以添加页脚
st.markdown("---")
st.caption("Los_inf._AB: Time from infection discovery to antibiotic use.")
st.caption("APS III: Acute Physiology Score III.")
st.caption("LODS: Logistic Organ Dysfunction System score.")
st.caption("Note: This prediction tool is for clinical reference only and should not replace clinical judgment.")
