import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# 添加全局CSS样式
st.markdown("""
<style>
/* 全局样式 */
body {
    background-color: #f8f9fa;
    font-family: 'Arial', sans-serif;
}

/* 风险等级显示优化 */
.risk-level {
    font-size: 24px;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}

.risk-level.low {
    background-color: #d4edda;
    color: #155724;
}

.risk-level.moderate {
    background-color: #fff3cd;
    color: #856404;
}

.risk-level.high {
    background-color: #ffeeba;
    color: #856404;
}

.risk-level.very-high {
    background-color: #f8d7da;
    color: #721c24;
}

/* 卡片式布局 */
.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

/* 输入区域样式 */
.input-container {
    margin-bottom: 20px;
}

/* 按钮样式 */
.stButton>button {
    background-color: #1A5276;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}

/* 进度条样式 */
.stProgress > div > div > div > div {
    background-color: #2E86C1;
}

/* 图表样式 */
.plot-container {
    height: 300px;
    margin: 20px 0;
}

/* 表格样式 */
.stDataFrame {
    margin-top: 20px;
}

/* 侧边栏样式 */
.stSidebar {
    background-color: #f0f2f6;
    padding: 20px;
}

/* 标题样式 */
h1, h2, h3 {
    color: #1A5276;
}

/* 说明文本样式 */
.stMarkdown {
    color: #2E86C1;
}

/* 错误信息样式 */
.stError {
    background-color: #f8d7da;
    color: #721c24;
    padding: 15px;
    border-radius: 5px;
}

/* 成功信息样式 */
.stSuccess {
    background-color: #d4edda;
    color: #155724;
    padding: 15px;
    border-radius: 5px;
}

/* 警告信息样式 */
.stWarning {
    background-color: #fff3cd;
    color: #856404;
    padding: 15px;
    border-radius: 5px;
}

/* 滑块样式 */
.stSlider > div > div > div > div {
    background-color: #2E86C1;
}

/* 下拉菜单样式 */
.stSelectbox > div > div > div {
    background-color: white;
    color: #1A5276;
}

/* 日期选择器样式 */
.stDateInput > div > div > div {
    background-color: white;
    color: #1A5276;
}

/* 代码块样式 */
.stCode {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}

/* 链接样式 */
a {
    color: #1A5276;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #2E86C1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #1A5276;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .card {
        padding: 15px;
    }
    
    .risk-level {
        font-size: 20px;
    }
}
</style>
""", unsafe_allowhtml=True)

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
    'MBP': {"type": "numerical", "min": 0, "max": 300, "default": 75, "unit": "mmHg"},
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
    
    if st.button("Predict AKD Risk", help="Click to calculate AKD risk probability"):
        try:
            # 根据模型训练时的特征顺序重新排序输入特征
            ordered_feature_values = [feature_values[feature_order.index(feature)] for feature in feature_order]
            
            # 转换为模型输入格式
            features = np.array([ordered_feature_values])
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
            
            # 添加风险等级分类
            if probability >= 75:
                risk_level = "Very High"
                color = "red"
            elif probability >= 50:
                risk_level = "High"
                color = "orange"
            elif probability >= 25:
                risk_level = "Moderate"
                color = "yellow"
            else:
                risk_level = "Low"
                color = "green"
                
            st.markdown(f'<p style="color:{color};">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
            
            # 添加特征重要性解释（如果有）
            # st.subheader("Key Contributing Factors")
            # 这里可以添加特征重要性分析
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# 可以添加页脚
st.markdown("---")
st.caption("Los_inf._AB: Time from infection discovery to antibiotic use.")
st.caption("APS III: Acute Physiology Score III.")
st.caption("LODS: Logistic Organ Dysfunction System score.")
st.caption("Note: This prediction tool is for clinical reference only and should not replace clinical judgment.")
