import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载逻辑回归模型
model = joblib.load('saki_lr_model.pkl')

# 定义特征参数（包含单位和范围）
feature_config = {
    # 分类特征 (直接显示0/1)
    'ACEI/ARB': {"type": "categorical", "options": [0, 1]},
    'CRRT': {"type": "categorical", "options": [0, 1]},
    'Cerebrovascular_disease': {"type": "categorical", "options": [0, 1]},
    'Vasoactive_agent': {"type": "categorical", "options": [0, 1]},
    
    # 数值型特征
    'APS III': {"type": "numerical", "min": 0, "max": 215, "default": 50, "unit": "points"},
    'Age': {"type": "numerical", "min": 18, "max": 120, "default": 60, "unit": "years"},
    'Baseexcess': {"type": "numerical", "min": -25, "max": 30, "default": 0, "unit": "mmol/L"},
    'Bun': {"type": "numerical", "min": 1, "max": 100, "default": 20, "unit": "mg/dL"},
    'Glucose': {"type": "numerical", "min": 1.5, "max": 50.0, "default": 5.5, "unit": "mmol/L"},
    'LODS': {"type": "numerical", "min": 0, "max": 22, "default": 5, "unit": "points"},
    'Los_inf._AB': {"type": "numerical", "min": 0, "max": 30, "default": 7, "unit": "days"},
    'OASIS': {"type": "numerical", "min": 0, "max": 299, "default": 150, "unit": "points"},
    'Pco2': {"type": "numerical", "min": 10, "max": 150, "default": 40, "unit": "mmHg"},
    'Po2': {"type": "numerical", "min": 20, "max": 700, "default": 100, "unit": "mmHg"},
    'Resp_rate': {"type": "numerical", "min": 0, "max": 50, "default": 18, "unit": "breaths/min"},
    'Scr_baseline': {"type": "numerical", "min": 0.3, "max": 10.0, "default": 1.0, "unit": "mg/dL"},
    'Sodium': {"type": "numerical", "min": 120, "max": 160, "default": 140, "unit": "mmol/L"},
    'Temperature': {"type": "numerical", "min": 32.0, "max": 42.0, "default": 36.6, "unit": "°C"},
    'WBC': {"type": "numerical", "min": 0.0, "max": 50.0, "default": 8.0, "unit": "×10^9/L"},
    'Weight': {"type": "numerical", "min": 30, "max": 200, "default": 70, "unit": "kg"}
}

# 设置页面标题
st.title("SA-AKI_AKD Prediction Model with SHAP Visualization")

# 生成输入组件
st.header("Patient Clinical Parameters")
inputs = {}
cols = st.columns(3)  # 创建3列布局

for i, (feature, config) in enumerate(feature_config.items()):
    with cols[i % 3]:  # 循环使用3列
        if config["type"] == "categorical":
            inputs[feature] = st.selectbox(
                label=f"{feature} (0=NO, 1=YES)",  # 添加解释说明
                options=config["options"]
            )
        else:
            inputs[feature] = st.number_input(
                label=f"{feature} ({config['unit']})",
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=0.1 if isinstance(config["default"], float) else 1
            )

# 转换为模型输入格式（自动保持数据类型）
input_df = pd.DataFrame([inputs], columns=feature_config.keys())

# 预测和可视化
if st.button("Predict"):
    # 进行预测
    try:
        proba = model.predict_proba(input_df)[0][1]  # 获取正类概率
        # 显示预测结果
        st.markdown(f"## Prediction Result: **{proba*100:.1f}%** probability of SA-AKI/SA-AKD")
        
        # 计算SHAP值
        explainer = shap.LinearExplainer(model, input_df)
        shap_values = explainer.shap_values(input_df)
        
        # 绘制SHAP解释图
        st.subheader("Feature Impact Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 添加解释说明
        st.markdown("""
        **SHAP Value Interpretation Guide:**
        - Positive SHAP value ➔ Increases prediction probability
        - Negative SHAP value ➔ Decreases prediction probability
        - Value magnitude indicates impact strength
        - Calculated in log-odds units
        """)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# 侧边栏添加说明
st.sidebar.markdown("""
**Clinical Parameters Guide:**
- **0/1 二分类变量:**  
  0 = 未使用/未发生  
  1 = 使用/已发生
- **CRRT:** 连续性肾脏替代治疗
- **Vasoactive_agent:** 血管活性药物使用
- **APS III:** 急性生理评分第三版
- **OASIS:** 氧合状态指数
""")
