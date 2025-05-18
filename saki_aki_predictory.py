import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shap

# --------------------------
# 配置与初始化
# --------------------------
st.set_page_config(
    page_title="AKD Prediction Model",
    layout="wide",
    page_icon="🏥"
)

# --------------------------
# 自定义样式
# --------------------------
st.markdown("""
    <style>
    .risk-high { background-color: #ffcccc !important; border-left: 5px solid #ff0000; }
    .risk-medium { background-color: #fff3cd !important; border-left: 5px solid #ffc107; }
    .risk-low { background-color: #d4edda !important; border-left: 5px solid #28a745; }
    .parameter-group { border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .feature-importance { font-size: 0.9em; color: #6c757d; }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# 模型与数据加载
# --------------------------
@st.cache_resource
def load_assets():
    """加载模型和scaler"""
    try:
        model = joblib.load('0511重置版saki_lr_model1.pkl')
        scaler = joblib.load('0511重置版saki_scaler.pkl')
        explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.LinearExplainer(model, scaler.transform(feature_samples))
        return model, scaler, explainer
    except Exception as e:
        st.error(f"资源加载失败: {str(e)}")
        st.stop()

model, scaler, explainer = load_assets()

# --------------------------
# 特征配置
# --------------------------
FEATURE_CONFIG = {
    'ACEI/ARB': {
        "type": "binary",
        "label": "ACEI/ARB使用",
        "options": {0: "否", 1: "是"},
        "help": "患者是否正在使用ACEI/ARB类药物"
    },
    'APS III': {
        "type": "numeric",
        "label": "APS III评分",
        "min": 0,
        "max": 215,
        "default": 40,
        "unit": "分",
        "clinical_range": (20, 50)
    },
    # 其他特征配置...
}

FEATURE_ORDER = ['ACEI/ARB', 'APS III', ..., 'Weight']

# --------------------------
# 组件生成函数
# --------------------------
def create_input_widget(feature):
    """根据特征配置生成输入组件"""
    config = FEATURE_CONFIG[feature]
    
    with st.container():
        if config["type"] == "binary":
            return st.radio(
                label=config["label"],
                options=list(config["options"].keys()),
                format_func=lambda x: config["options"][x],
                help=config.get("help", "")
            )
        elif config["type"] == "numeric":
            value = st.number_input(
                label=f"{config['label']} ({config['unit']})",
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=1.0,
                help=get_clinical_guidance(config)
            )
            validate_input(value, config)
            return value

def get_clinical_guidance(config):
    """生成临床参考范围提示"""
    if "clinical_range" in config:
        return f"临床参考范围: {config['clinical_range'][0]}-{config['clinical_range'][1]} {config['unit']}"
    return ""

def validate_input(value, config):
    """输入值验证"""
    if "clinical_range" in config:
        if not (config["clinical_range"][0] <= value <= config["clinical_range"][1]):
            st.warning(f"⚠️ 异常值: 临床常见范围为{config['clinical_range'][0]}-{config['clinical_range'][1]}")

# --------------------------
# 可解释性分析
# --------------------------
def explain_prediction(features):
    """生成SHAP解释"""
    shap_values = explainer.shap_values(features)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features, feature_names=FEATURE_ORDER, show=False)
    plt.tight_layout()
    return fig

# --------------------------
# 主界面
# --------------------------
def main():
    # 页眉
    with st.container():
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            st.image(Image.open("东华医院图标.png"), width=120)
        with cols[1]:
            st.title("急性肾脏病（AKD）预测模型")
            st.markdown("**东莞东华医院急诊科临床决策支持系统**")
    
    st.markdown("---")
    
    # 主内容区
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.header("患者参数")
        inputs = {}
        
        # 分组输入
        with st.expander("基本信息", expanded=True):
            inputs['Weight'] = create_input_widget('Weight')
            inputs['Age'] = create_input_widget('Age')
        
        with st.expander("生命体征"):
            inputs['MBP'] = create_input_widget('MBP')
            inputs['Resp Rate'] = create_input_widget('Resp Rate')
        
        # 其他分组...
    
    with col2:
        st.header("风险评估")
        
        if st.button("开始预测", type="primary"):
            try:
                # 特征处理
                features = np.array([inputs[feat] for feat in FEATURE_ORDER]).reshape(1, -1)
                scaled_features = scaler.transform(features)
                
                # 预测
                proba = model.predict_proba(scaled_features)[0][1]
                risk_class = "high" if proba >= 0.5 else "low"
                
                # 结果展示
                with st.container():
                    st.markdown(f"""
                        <div class='risk-{risk_class}' style='padding: 20px; border-radius: 8px;'>
                            <h3>预测结果</h3>
                            <p style='font-size: 2em; margin: 0.5em 0;'>{proba*100:.1f}%</p>
                            <p>AKD发生风险：<strong>{'高危' if risk_class == 'high' else '低危'}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 可解释性分析
                    st.subheader("风险因素分析")
                    fig = explain_prediction(scaled_features)
                    st.pyplot(fig)
                    
                    # 临床建议
                    st.subheader("临床建议")
                    if risk_class == "high":
                        st.markdown("""
                            - 立即进行肾功能评估
                            - 考虑启动肾脏保护措施
                            - 每小时监测尿量
                            - 复查血肌酐和电解质
                        """)
                    else:
                        st.markdown("""
                            - 维持当前治疗方案
                            - 每4小时监测生命体征
                            - 关注液体平衡
                        """)
            
            except Exception as e:
                st.error(f"预测失败: {str(e)}")

# --------------------------
# 运行应用
# --------------------------
if __name__ == "__main__":
    main()
