import streamlit as st
import joblib
import numpy as np
from PIL import Image

# 添加精简版CSS样式（适配Streamlit组件类名变化）
st.markdown("""
<style>
/* 优化组件选择器 */
div[data-testid="stExpander"] details {
    background: white;
    border-radius: 8px;
    margin: 8px 0;
}

div[data-testid="stVerticalBlock"] > div > div {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}

/* 按钮样式 */
button[kind="primary"] {
    background: #1A5276 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}

/* 进度条颜色 */
div[role="progressbar"] > div > div {
    background: #2E86C1 !important;
}

/* 风险等级卡片 */
.custom-risk {
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    font-size: 1.4rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 设置页面配置（新增参数避免渲染问题）
st.set_page_config(
    page_title="AKD Prediction Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题栏容器（优化响应式布局）
with st.container():
    cols = st.columns([0.15, 0.85])
    with cols[0]:
        logo = Image.open("东华医院图标.png")
        st.image(logo, use_column_width=True)
    with cols[1]:
        st.markdown("""
        <div style='border-left: 4px solid #1A5276; padding-left: 1.5rem;'>
            <h1 style='margin:0;color:#1A5276'>AKD Prediction Model</h1>
            <p style='font-size:1.1rem;color:#2E86C1;margin:0.5rem 0'>急诊科临床决策支持系统</p>
            <p style='color:#666;margin:0'>东莞东华医院</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# 模型加载（增加兼容性处理）
try:
    model = joblib.load('0511重置版saki_lr_model1.pkl')
    scaler = joblib.load('0511重置版saki_scaler.pkl')
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 特征定义（保持顺序一致性）
feature_order = [
    'ACEI/ARB', 'APS III', 'CRRT', 'Cerebrovascular Disease', 'LODS', 'Los_inf._AB',
    'MBP', 'Mechanical Ventilation', 'Paraplegia', 'Resp Rate', 'Scr Baseline', 'SpO2',
    'Vasoactive Agent', 'Weight'
]

# 输入参数分组（使用新版expander参数）
input_values = {}
with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("患者参数设置")
        
        # 参数分组设置
        with st.expander("人口统计学", expanded=True):
            input_values['Weight'] = st.number_input("体重 (kg)", 0, 500, 60)
            input_values['Scr Baseline'] = st.number_input("基线肌酐 (mmol/L)", 0, 2000, 60)

        with st.expander("临床状况"):
            input_values['Cerebrovascular Disease'] = st.selectbox("脑血管疾病", [0, 1], format_func=lambda x: "是" if x else "否")
            input_values['Paraplegia'] = st.selectbox("截瘫", [0, 1], format_func=lambda x: "是" if x else "否")

        with st.expander("生命体征"):
            input_values['MBP'] = st.number_input("平均动脉压 (mmHg)", 0, 300, 75)
            input_values['Resp Rate'] = st.number_input("呼吸频率 (次/分)", 0, 80, 20)
            input_values['SpO2'] = st.number_input("血氧饱和度 (%)", 0, 100, 100)
            input_values['Los_inf._AB'] = st.number_input("抗生素使用延迟 (小时)", 0, 168, 0)

        with st.expander("治疗措施"):
            input_values['ACEI/ARB'] = st.selectbox("ACEI/ARB使用", [0, 1], format_func=lambda x: "是" if x else "否")
            input_values['CRRT'] = st.selectbox("持续肾脏替代治疗", [0, 1], format_func=lambda x: "是" if x else "否")
            input_values['Mechanical Ventilation'] = st.selectbox("机械通气", [0, 1], format_func=lambda x: "是" if x else "否")
            input_values['Vasoactive Agent'] = st.selectbox("血管活性药物", [0, 1], format_func=lambda x: "是" if x else "否")

        with st.expander("评分系统"):
            input_values['APS III'] = st.number_input("APS III评分", 0, 215, 0)
            input_values['LODS'] = st.number_input("LODS评分", 0, 22, 0)

    with col2:
        st.subheader("预测结果")
        if st.button("进行风险评估", type="primary"):
            try:
                # 特征顺序验证
                ordered_features = [input_values[feat] for feat in feature_order]
                features = np.array([ordered_features]).astype(float)
                
                # 数据标准化
                scaled_features = scaler.transform(features)
                
                # 模型预测
                proba = model.predict_proba(scaled_features)[0][1]
                risk_percent = proba * 100
                
                # 结果可视化
                with st.container():
                    st.markdown(f"""
                    <div class="custom-risk" style='
                        background-color:{"#f8d7da" if risk_percent > 50 else "#d4edda"};
                        color:{"#721c24" if risk_percent > 50 else "#155724"};
                    '>
                        AKD风险概率：{risk_percent:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 进度条显示
                    st.progress(int(risk_percent), text="风险评估进度")
                    
                    # 临床建议
                    if risk_percent > 75:
                        st.error("高风险：建议立即启动肾脏保护方案，加强监测")
                    elif risk_percent > 50:
                        st.warning("中高风险：建议完善相关检查并密切观察")
                    elif risk_percent > 25:
                        st.info("中风险：建议定期复查肾功能指标")
                    else:
                        st.success("低风险：建议维持当前治疗方案")

            except Exception as e:
                st.error(f"预测过程中发生错误：{str(e)}")

# 页脚说明（使用原生组件）
st.divider()
with st.expander("术语解释", expanded=False):
    st.caption("""
    - **Los_inf._AB**: 从发现感染到使用抗生素的时间间隔
    - **APS III**: 急性生理学评分III
    - **LODS**: 器官功能障碍逻辑评分系统
    - **CRRT**: 连续性肾脏替代治疗
    """)
    
st.caption("""
⚠️ 注意事项：本预测结果仅供参考，临床决策需结合患者实际情况。预测模型准确率约86.7%（AUC=0.867），使用数据截止至2023年12月。
""")
