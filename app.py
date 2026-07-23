import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path


# =========================
# 1. 页面基础设置
# =========================

st.set_page_config(
    page_title="FraudShield 金融反诈智能辅助系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# 2. 路径设置
# =========================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

PREDICTION_PATH = OUTPUT_DIR / "prediction_result.csv"
GLOBAL_IMPORTANCE_PATH = OUTPUT_DIR / "global_feature_importance.csv"
BUSINESS_CASE_PATH = OUTPUT_DIR / "risk_cases_business_explained.csv"
EXPLAINED_CASE_PATH = OUTPUT_DIR / "risk_cases_explained.csv"


# =========================
# 3. 美化 CSS
# =========================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 8%, rgba(80, 140, 255, 0.10), transparent 28%),
            radial-gradient(circle at 92% 12%, rgba(255, 100, 130, 0.08), transparent 26%),
            linear-gradient(180deg, rgba(250, 252, 255, 1) 0%, rgba(245, 247, 251, 1) 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, #172033 100%);
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 26px;
        background: linear-gradient(135deg, #111827 0%, #1E3A8A 48%, #2563EB 100%);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .hero p {
        font-size: 1.02rem;
        line-height: 1.8;
        color: rgba(255, 255, 255, 0.88);
        max-width: 920px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        color: white;
        font-size: 0.86rem;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.20);
    }

    .soft-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        padding: 20px 22px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        height: 100%;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .metric-label {
        color: #64748B;
        font-size: 0.86rem;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #0F172A;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }

    .metric-help {
        color: #64748B;
        font-size: 0.82rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0.4rem 0 0.8rem 0;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 0.95rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }

    .risk-high {
        color: #991B1B;
        background: #FEE2E2;
        border: 1px solid #FECACA;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .risk-mid {
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .risk-low {
        color: #075985;
        background: #E0F2FE;
        border: 1px solid #BAE6FD;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .risk-normal {
        color: #166534;
        background: #DCFCE7;
        border: 1px solid #BBF7D0;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .case-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.6rem;
    }

    .case-line {
        color: #334155;
        font-size: 0.94rem;
        line-height: 1.8;
    }

    .explain-box {
        background: #FFFFFF;
        border-left: 5px solid #2563EB;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        color: #1E293B;
        line-height: 1.9;
        font-size: 0.95rem;
    }

    .warning-phone {
        max-width: 460px;
        background: #111827;
        padding: 16px;
        border-radius: 30px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
        margin: 0 auto;
    }

    .phone-screen {
        background: #F8FAFC;
        border-radius: 22px;
        padding: 18px;
        min-height: 340px;
    }

    .phone-header {
        color: #0F172A;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .message-bubble {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 14px 16px;
        color: #334155;
        line-height: 1.8;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    .flow-wrap {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin-top: 10px;
    }

    .flow-step {
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }

    .flow-icon {
        font-size: 1.5rem;
        margin-bottom: 6px;
    }

    .flow-title {
        font-weight: 800;
        color: #0F172A;
        font-size: 0.94rem;
        margin-bottom: 4px;
    }

    .flow-desc {
        color: #64748B;
        font-size: 0.80rem;
        line-height: 1.5;
    }

    @media (max-width: 900px) {
        .flow-wrap {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 4. 工具函数
# =========================

def money(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


def risk_badge(level):
    if level == "高风险":
        return '<span class="risk-high">高风险</span>'
    if level == "中风险":
        return '<span class="risk-mid">中风险</span>'
    if level == "低风险":
        return '<span class="risk-low">低风险</span>'
    return '<span class="risk-normal">正常</span>'


def get_text(row, business_col, fallback_col):
    if business_col in row.index and pd.notna(row[business_col]):
        return str(row[business_col])
    if fallback_col in row.index and pd.notna(row[fallback_col]):
        return str(row[fallback_col])
    return "暂无说明。"


@st.cache_data
def load_data():
    if not PREDICTION_PATH.exists():
        st.error(f"没有找到预测结果文件：{PREDICTION_PATH}")
        st.stop()

    pred_df = pd.read_csv(PREDICTION_PATH)

    if BUSINESS_CASE_PATH.exists():
        risk_df = pd.read_csv(BUSINESS_CASE_PATH)
    elif EXPLAINED_CASE_PATH.exists():
        risk_df = pd.read_csv(EXPLAINED_CASE_PATH)
    else:
        risk_df = pred_df[pred_df["risk_level"].isin(["中风险", "高风险"])].copy()

    if GLOBAL_IMPORTANCE_PATH.exists():
        global_df = pd.read_csv(GLOBAL_IMPORTANCE_PATH)
    else:
        global_df = pd.DataFrame()

    return pred_df, risk_df, global_df


prediction_df, risk_cases_df, global_importance_df = load_data()


# =========================
# 5. 侧边栏
# =========================

st.sidebar.markdown("## 🛡️ FraudShield")
st.sidebar.markdown("金融交易反诈智能辅助系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能导航",
    [
        "项目首页",
        "风险总览",
        "风险案例",
        "模型解释",
        "客户提醒",
        "成果文件"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 当前数据")
st.sidebar.markdown(f"- 预测交易数：**{len(prediction_df):,}**")
st.sidebar.markdown(f"- 风险案例数：**{len(risk_cases_df):,}**")

if "risk_level" in prediction_df.columns:
    high_count = int((prediction_df["risk_level"] == "高风险").sum())
    mid_count = int((prediction_df["risk_level"] == "中风险").sum())
    st.sidebar.markdown(f"- 高风险：**{high_count:,}**")
    st.sidebar.markdown(f"- 中风险：**{mid_count:,}**")


# =========================
# 6. 项目首页
# =========================

if page == "项目首页":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">AI + 金融风控 + 客户保护</div>
            <h1>FraudShield 金融交易反诈智能辅助系统</h1>
            <p>
            本系统面向银行、支付机构和消费金融平台的交易反诈场景，
            通过机器学习识别高风险交易，通过模型解释说明风险来源，
            并自动生成面向风控人员和客户的差异化提醒话术。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_tx = len(prediction_df)
    risk_tx = int(prediction_df["risk_level"].isin(["中风险", "高风险"]).sum())
    high_tx = int((prediction_df["risk_level"] == "高风险").sum())
    fraud_pred = int((prediction_df["predict_isFraud"] == 1).sum()) if "predict_isFraud" in prediction_df.columns else risk_tx

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">交易样本总量</div>
                <div class="metric-value">{total_tx:,}</div>
                <div class="metric-help">当前演示数据集</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">中高风险交易</div>
                <div class="metric-value">{risk_tx:,}</div>
                <div class="metric-help">需关注或复核</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">高风险交易</div>
                <div class="metric-value">{high_tx:,}</div>
                <div class="metric-help">建议拦截或人工复核</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">疑似欺诈预测</div>
                <div class="metric-value">{fraud_pred:,}</div>
                <div class="metric-help">模型判定为欺诈</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    st.markdown('<div class="section-title">系统工作流程</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-wrap">
            <div class="flow-step">
                <div class="flow-icon">📥</div>
                <div class="flow-title">交易输入</div>
                <div class="flow-desc">读取交易金额、类型、账户余额等数据</div>
            </div>
            <div class="flow-step">
                <div class="flow-icon">🧠</div>
                <div class="flow-title">AI 识别</div>
                <div class="flow-desc">机器学习模型输出欺诈风险分数</div>
            </div>
            <div class="flow-step">
                <div class="flow-icon">🚦</div>
                <div class="flow-title">风险分级</div>
                <div class="flow-desc">划分正常、低风险、中风险、高风险</div>
            </div>
            <div class="flow-step">
                <div class="flow-icon">🔍</div>
                <div class="flow-title">模型解释</div>
                <div class="flow-desc">说明交易为什么存在风险</div>
            </div>
            <div class="flow-step">
                <div class="flow-icon">💬</div>
                <div class="flow-title">智能提醒</div>
                <div class="flow-desc">生成风控解释和客户劝阻话术</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="section-title">项目定位</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="soft-card">
            FraudShield 不是简单的“诈骗识别模型”，而是一个面向金融机构的
            <b>反诈辅助决策系统</b>。它强调三个核心价值：
            <br><br>
            <b>1. 提升效率：</b> 自动筛选高风险交易，减少人工逐笔排查压力。<br>
            <b>2. 降低成本：</b> 将大量正常交易自动放行，让人工审核集中在真正可疑案例。<br>
            <b>3. 优化体验：</b> 对客户生成温和、可理解的风险提醒，减少生硬拦截带来的不满。
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<div class="section-title">当前阶段成果</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="soft-card">
            ✅ 完成 PaySim 交易数据读取<br>
            ✅ 完成欺诈识别模型训练<br>
            ✅ 输出交易风险分数和风险等级<br>
            ✅ 引入 SHAP 模型解释<br>
            ✅ 生成风控人员解释文本<br>
            ✅ 生成客户风险提醒话术<br>
            ✅ 完成可视化演示系统
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# 7. 风险总览
# =========================

elif page == "风险总览":
    st.markdown('<div class="section-title">风险总览</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">从整体交易样本中观察风险等级分布、疑似欺诈数量和风险金额情况。</div>',
        unsafe_allow_html=True
    )

    total_tx = len(prediction_df)
    high_tx = int((prediction_df["risk_level"] == "高风险").sum())
    mid_tx = int((prediction_df["risk_level"] == "中风险").sum())
    low_tx = int((prediction_df["risk_level"] == "低风险").sum())
    normal_tx = int((prediction_df["risk_level"] == "正常").sum())

    risk_amount = prediction_df[prediction_df["risk_level"].isin(["中风险", "高风险"])]["amount"].sum()
    avg_score = prediction_df["risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">总交易数</div>
                <div class="metric-value">{total_tx:,}</div>
                <div class="metric-help">系统已完成风险评分</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">高风险交易</div>
                <div class="metric-value">{high_tx:,}</div>
                <div class="metric-help">建议拦截或人工复核</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">中高风险金额</div>
                <div class="metric-value">{money(risk_amount)}</div>
                <div class="metric-help">需重点关注的交易金额</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">平均风险分数</div>
                <div class="metric-value">{avg_score:.4f}</div>
                <div class="metric-help">全部交易平均值</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    col_a, col_b = st.columns([0.9, 1.1])

    with col_a:
        risk_counts = prediction_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        level_order = ["正常", "低风险", "中风险", "高风险"]
        risk_counts["risk_level"] = pd.Categorical(
            risk_counts["risk_level"],
            categories=level_order,
            ordered=True
        )
        risk_counts = risk_counts.sort_values("risk_level")

        fig_pie = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            hole=0.55,
            title="风险等级构成"
        )
        fig_pie.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title_text="风险等级"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            text="count",
            title="各风险等级交易数量"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=420,
            xaxis_title="风险等级",
            yaxis_title="交易数量",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">交易清单预览</div>', unsafe_allow_html=True)

    show_df = prediction_df.copy()
    show_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
        "risk_score",
        "predict_isFraud",
        "risk_level",
        "risk_action"
    ]
    show_cols = [c for c in show_cols if c in show_df.columns]

    level_filter = st.multiselect(
        "筛选风险等级",
        options=["正常", "低风险", "中风险", "高风险"],
        default=["中风险", "高风险"]
    )

    filtered = show_df[show_df["risk_level"].isin(level_filter)].copy()
    filtered = filtered.sort_values("risk_score", ascending=False)

    st.dataframe(
        filtered[show_cols],
        use_container_width=True,
        height=380
    )


# =========================
# 8. 风险案例
# =========================

elif page == "风险案例":
    st.markdown('<div class="section-title">高风险交易案例</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">选择一笔风险交易，查看它的交易信息、风险依据和风控处置建议。</div>',
        unsafe_allow_html=True
    )

    if len(risk_cases_df) == 0:
        st.warning("当前没有风险案例数据。")
        st.stop()

    case_df = risk_cases_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    col_filter_1, col_filter_2 = st.columns([0.5, 0.5])

    with col_filter_1:
        available_levels = case_df["risk_level"].dropna().unique().tolist()
        selected_levels = st.multiselect(
            "风险等级",
            options=available_levels,
            default=available_levels
        )

    with col_filter_2:
        selected_type = st.selectbox(
            "交易类型",
            options=["全部"] + sorted(case_df["type"].dropna().unique().tolist())
        )

    if selected_levels:
        case_df = case_df[case_df["risk_level"].isin(selected_levels)]

    if selected_type != "全部":
        case_df = case_df[case_df["type"] == selected_type]

    if len(case_df) == 0:
        st.info("没有符合筛选条件的风险案例。")
        st.stop()

    def case_option(row):
        return f"{row.name + 1}｜{row['type']}｜金额 {money(row['amount'])}｜风险分 {float(row['risk_score']):.4f}"

    selected_label = st.selectbox(
        "选择案例",
        options=[case_option(row) for _, row in case_df.iterrows()]
    )

    selected_index = int(selected_label.split("｜")[0]) - 1
    row = case_df.iloc[selected_index]

    left, right = st.columns([0.9, 1.1])

    with left:
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="case-title">交易基本信息</div>
                <div class="case-line">交易类型：<b>{row['type']}</b></div>
                <div class="case-line">交易金额：<b>{money(row['amount'])}</b></div>
                <div class="case-line">付款账户：<b>{row['nameOrig']}</b></div>
                <div class="case-line">收款账户：<b>{row['nameDest']}</b></div>
                <div class="case-line">付款方交易前余额：<b>{money(row['oldbalanceOrg'])}</b></div>
                <div class="case-line">付款方交易后余额：<b>{money(row['newbalanceOrig'])}</b></div>
                <div class="case-line">真实标签 isFraud：<b>{row.get('isFraud', '未知')}</b></div>
                <div class="case-line">风险等级：{risk_badge(row['risk_level'])}</div>
                <div class="case-line">风险分数：<b>{float(row['risk_score']):.4f}</b></div>
                <div class="case-line">处置建议：<b>{row['risk_action']}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        reason_text = get_text(row, "business_risk_reason", "risk_reason")
        staff_text = get_text(row, "business_staff_explanation", "staff_explanation")

        st.markdown(
            f"""
            <div class="explain-box">
                <b>业务化风险原因</b><br><br>
                {reason_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        st.markdown(
            f"""
            <div class="explain-box">
                <b>给风控人员的解释</b><br><br>
                {staff_text}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# 9. 模型解释
# =========================

elif page == "模型解释":
    st.markdown('<div class="section-title">模型解释</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">展示模型整体最关注哪些交易特征，帮助风控人员理解模型判断依据。</div>',
        unsafe_allow_html=True
    )

    if global_importance_df.empty:
        st.warning("没有找到 global_feature_importance.csv，请先运行第二阶段模型解释代码。")
        st.stop()

    top_n = st.slider("展示前 N 个重要特征", min_value=5, max_value=15, value=10, step=1)

    top_df = global_importance_df.head(top_n).copy()
    top_df = top_df.sort_values("mean_abs_shap", ascending=True)

    fig = px.bar(
        top_df,
        x="mean_abs_shap",
        y="feature_cn",
        orientation="h",
        text="mean_abs_shap",
        title=f"全局特征重要性 Top {top_n}"
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=520,
        xaxis_title="平均绝对 SHAP 值",
        yaxis_title="特征名称",
        margin=dict(l=20, r=40, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("")

    left, right = st.columns([0.9, 1.1])

    with left:
        st.markdown(
            """
            <div class="soft-card">
            <div class="case-title">如何理解这张图？</div>
            <div class="case-line">
            SHAP 特征重要性表示模型在整体判断中更依赖哪些变量。
            数值越大，说明该特征对模型区分正常交易和欺诈交易的影响越明显。
            </div>
            <br>
            <div class="case-line">
            从当前结果看，模型主要关注付款方余额变化、交易后余额、交易金额和交易类型。
            这些因素都与金融反诈业务逻辑高度相关。
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        top_3 = global_importance_df.head(3)
        summary_lines = []
        for _, item in top_3.iterrows():
            summary_lines.append(
                f"• {item['feature_cn']}：mean_abs_shap = {item['mean_abs_shap']:.4f}"
            )

        st.markdown(
            f"""
            <div class="soft-card">
            <div class="case-title">核心发现</div>
            <div class="case-line">
            当前模型最重要的前三个风险识别特征为：
            <br><br>
            {'<br>'.join(summary_lines)}
            <br><br>
            这说明系统能够从账户余额变化、资金流出强度和交易结构异常中识别潜在风险。
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# 10. 客户提醒
# =========================

elif page == "客户提醒":
    st.markdown('<div class="section-title">客户提醒话术</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">系统会把模型风险结果转化为客户能理解的温和提醒，减少生硬拦截带来的体验问题。</div>',
        unsafe_allow_html=True
    )

    if len(risk_cases_df) == 0:
        st.warning("当前没有风险案例数据。")
        st.stop()

    case_df = risk_cases_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    def warning_option(row):
        return f"{row.name + 1}｜{row['type']}｜{money(row['amount'])}｜{row['risk_level']}"

    selected_label = st.selectbox(
        "选择要展示的客户提醒案例",
        options=[warning_option(row) for _, row in case_df.iterrows()]
    )

    selected_index = int(selected_label.split("｜")[0]) - 1
    row = case_df.iloc[selected_index]

    customer_warning = get_text(row, "business_customer_warning", "customer_warning")
    staff_text = get_text(row, "business_staff_explanation", "staff_explanation")

    left, right = st.columns([0.85, 1.15])

    with left:
        st.markdown(
            f"""
            <div class="warning-phone">
                <div class="phone-screen">
                    <div class="phone-header">银行安全提醒</div>
                    <div class="message-bubble">
                    {customer_warning}
                    </div>
                    <br>
                    <div style="font-size:0.82rem;color:#64748B;line-height:1.6;">
                    该提醒以客户保护为目标，不直接暴露模型细节，避免造成恐慌，同时引导客户主动核实交易。
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="case-title">对应交易信息</div>
                <div class="case-line">交易类型：<b>{row['type']}</b></div>
                <div class="case-line">交易金额：<b>{money(row['amount'])}</b></div>
                <div class="case-line">付款账户：<b>{row['nameOrig']}</b></div>
                <div class="case-line">收款账户：<b>{row['nameDest']}</b></div>
                <div class="case-line">风险等级：{risk_badge(row['risk_level'])}</div>
                <div class="case-line">风险分数：<b>{float(row['risk_score']):.4f}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        st.markdown(
            f"""
            <div class="explain-box">
                <b>后台风控解释</b><br><br>
                {staff_text}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# 11. 成果文件
# =========================

elif page == "成果文件":
    st.markdown('<div class="section-title">项目成果文件</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">这里汇总当前项目已经生成的核心文件，方便你整理项目书、PPT 和答辩材料。</div>',
        unsafe_allow_html=True
    )

    files = [
        ("预测结果文件", OUTPUT_DIR / "prediction_result.csv", "每笔交易的风险分数、风险等级和处置建议"),
        ("模型评估报告", OUTPUT_DIR / "evaluation_report.txt", "AUC、Precision、Recall、F1-score 和混淆矩阵"),
        ("全局特征重要性", OUTPUT_DIR / "global_feature_importance.csv", "SHAP 全局特征重要性结果"),
        ("风险案例解释", OUTPUT_DIR / "risk_cases_explained.csv", "原始 SHAP 风险解释结果"),
        ("业务化解释结果", OUTPUT_DIR / "risk_cases_business_explained.csv", "适合项目书和 PPT 的风险解释"),
        ("业务案例报告", OUTPUT_DIR / "business_case_report.txt", "高风险交易案例文字报告"),
    ]

    rows = []
    for name, path, desc in files:
        rows.append(
            {
                "文件名称": name,
                "是否存在": "已生成" if path.exists() else "未找到",
                "文件路径": str(path),
                "用途说明": desc
            }
        )

    file_df = pd.DataFrame(rows)
    st.dataframe(file_df, use_container_width=True, height=320)

    st.markdown("")

    st.markdown(
        """
        <div class="soft-card">
        <div class="case-title">演示系统启动命令</div>
        <div class="case-line">
        在 PowerShell 中进入项目目录后运行：
        </div>
        <br>
        <code>cd E:\\FraudShield</code><br>
        <code>.\\.venv\\Scripts\\activate</code><br>
        <code>streamlit run app.py</code>
        </div>
        """,
        unsafe_allow_html=True
    )