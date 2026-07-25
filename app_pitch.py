import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="FraudShield 路演展示版",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

PREDICTION_PATH = OUTPUT_DIR / "prediction_result.csv"
GLOBAL_IMPORTANCE_PATH = OUTPUT_DIR / "global_feature_importance.csv"
BUSINESS_CASE_PATH = OUTPUT_DIR / "risk_cases_business_explained.csv"
EXPLAINED_CASE_PATH = OUTPUT_DIR / "risk_cases_explained.csv"


# =========================
# 数据读取
# =========================

@st.cache_data
def load_data():
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


prediction_df, risk_cases_df, global_df = load_data()


# =========================
# 工具函数
# =========================

def money(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def safe_text(row, business_col, fallback_col):
    if business_col in row.index and pd.notna(row[business_col]):
        return str(row[business_col])
    if fallback_col in row.index and pd.notna(row[fallback_col]):
        return str(row[fallback_col])
    return "暂无说明。"


def risk_chip(level):
    if level == "高风险":
        return '<span class="chip chip-red">高风险</span>'
    if level == "中风险":
        return '<span class="chip chip-orange">中风险</span>'
    if level == "低风险":
        return '<span class="chip chip-blue">低风险</span>'
    return '<span class="chip chip-green">正常</span>'


# =========================
# CSS 美化
# =========================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    [data-testid="stAppViewContainer"] {
        background:
        radial-gradient(circle at 8% 6%, rgba(59, 130, 246, 0.12), transparent 26%),
        radial-gradient(circle at 90% 10%, rgba(239, 68, 68, 0.10), transparent 25%),
        linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
    }

    .brand-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 48%, #2563EB 100%);
        color: white;
        border-radius: 30px;
        padding: 34px 38px;
        margin-bottom: 24px;
        box-shadow: 0 22px 55px rgba(15, 23, 42, 0.22);
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 18px;
    }

    .logo-mark {
        width: 58px;
        height: 58px;
        border-radius: 18px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.28);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }

    .brand-name {
        font-size: 1rem;
        color: rgba(255,255,255,0.78);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 2.55rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        line-height: 1.18;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: rgba(255,255,255,0.88);
        line-height: 1.9;
        max-width: 920px;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 18px;
        margin-right: 8px;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.22);
        color: white;
        font-size: 0.88rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 22px;
        padding: 20px 22px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.07);
        height: 100%;
    }

    .metric-label {
        color: #64748B;
        font-size: 0.88rem;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-bottom: 4px;
    }

    .metric-desc {
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    .panel {
        background: rgba(255,255,255,0.93);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 24px;
        padding: 24px 26px;
        box-shadow: 0 14px 36px rgba(15,23,42,0.08);
        height: 100%;
    }

    .panel-title {
        font-size: 1.2rem;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 12px;
    }

    .panel-text {
        color: #334155;
        line-height: 1.9;
        font-size: 0.96rem;
    }

    .chip {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
    }

    .chip-red {
        color: #991B1B;
        background: #FEE2E2;
        border: 1px solid #FECACA;
    }

    .chip-orange {
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
    }

    .chip-blue {
        color: #075985;
        background: #E0F2FE;
        border: 1px solid #BAE6FD;
    }

    .chip-green {
        color: #166534;
        background: #DCFCE7;
        border: 1px solid #BBF7D0;
    }

    .case-highlight {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border-left: 6px solid #2563EB;
        border-radius: 20px;
        padding: 20px 22px;
        color: #1E293B;
        line-height: 1.9;
        box-shadow: 0 12px 28px rgba(15,23,42,0.07);
    }

    .phone {
        max-width: 440px;
        margin: 0 auto;
        background: #0F172A;
        border-radius: 34px;
        padding: 16px;
        box-shadow: 0 24px 58px rgba(15,23,42,0.28);
    }

    .screen {
        background: #F8FAFC;
        border-radius: 24px;
        padding: 20px;
        min-height: 360px;
    }

    .sms-title {
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 14px;
    }

    .sms-bubble {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 16px 17px;
        line-height: 1.85;
        color: #334155;
        box-shadow: 0 10px 24px rgba(15,23,42,0.07);
    }

    .script-box {
        background: #0F172A;
        color: #E5E7EB;
        border-radius: 24px;
        padding: 24px 26px;
        line-height: 1.95;
        box-shadow: 0 18px 42px rgba(15,23,42,0.18);
    }

    .script-box b {
        color: white;
    }

    .roadmap-step {
        background: white;
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 8px 22px rgba(15,23,42,0.05);
    }

    .roadmap-title {
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 5px;
    }

    .roadmap-desc {
        color: #64748B;
        line-height: 1.7;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 顶部 Hero
# =========================

total_tx = len(prediction_df)
high_count = int((prediction_df["risk_level"] == "高风险").sum())
mid_count = int((prediction_df["risk_level"] == "中风险").sum())
risk_count = int(prediction_df["risk_level"].isin(["中风险", "高风险"]).sum())
fraud_pred_count = int((prediction_df["predict_isFraud"] == 1).sum()) if "predict_isFraud" in prediction_df.columns else risk_count

st.markdown(
    """
    <div class="brand-hero">
        <div class="brand-row">
            <div class="logo-mark">🛡️</div>
            <div>
                <div class="brand-name">FraudShield</div>
                <div style="font-weight:800;font-size:1.1rem;">金融交易反诈智能辅助系统</div>
            </div>
        </div>
        <div class="hero-title">让金融反诈从“黑箱拦截”变成“可解释的客户保护”</div>
        <div class="hero-subtitle">
            面向银行、支付机构和消费金融平台，本系统通过机器学习识别高风险交易，
            通过 SHAP 解释风险来源，并生成面向风控人员与客户的差异化提醒话术，
            实现风险识别、人工复核和客户保护的业务闭环。
        </div>
        <span class="hero-tag">AI 风险识别</span>
        <span class="hero-tag">模型可解释</span>
        <span class="hero-tag">客户劝阻话术</span>
        <span class="hero-tag">金融科技创新应用</span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 顶部指标
# =========================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">演示交易样本</div>
            <div class="metric-value">{total_tx:,}</div>
            <div class="metric-desc">系统已完成风险评分</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">中高风险交易</div>
            <div class="metric-value">{risk_count:,}</div>
            <div class="metric-desc">建议提醒、验证或复核</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">高风险交易</div>
            <div class="metric-value">{high_count:,}</div>
            <div class="metric-desc">建议拦截或人工复核</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">疑似欺诈预测</div>
            <div class="metric-value">{fraud_pred_count:,}</div>
            <div class="metric-desc">模型判定为欺诈</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")


# =========================
# 路演 Tabs
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 风险态势", "② 典型案例", "③ 模型解释", "④ 客户保护", "⑤ 落地价值"]
)


# =========================
# Tab 1 风险态势
# =========================

with tab1:
    left, right = st.columns([0.95, 1.05])

    risk_counts = prediction_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    level_order = ["正常", "低风险", "中风险", "高风险"]
    risk_counts["risk_level"] = pd.Categorical(
        risk_counts["risk_level"],
        categories=level_order,
        ordered=True
    )
    risk_counts = risk_counts.sort_values("risk_level")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">智能风险筛查机制</div>
                <div class="panel-text">
                系统对全部交易进行自动风险评分，并按照风险程度划分为
                正常、低风险、中风险和高风险四个等级。
                <br><br>
                正常交易可快速放行，低风险交易进行轻量提示，
                中风险交易触发二次验证，高风险交易进入人工复核或临时拦截流程。
                <br><br>
                通过风险分级，金融机构能够将有限的审核资源集中到真正需要关注的交易，
                提升风险排查效率，减少无效人工审核。
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        fig = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            text="count",
            title="风险等级分布"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=420,
            xaxis_title="风险等级",
            yaxis_title="交易数量",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        prediction_df.sort_values("risk_score", ascending=False)[
            ["step", "type", "amount", "nameOrig", "nameDest", "isFraud", "risk_score", "risk_level", "risk_action"]
        ].head(20),
        use_container_width=True,
        height=320
    )


# =========================
# Tab 2 典型案例
# =========================

with tab2:
    if len(risk_cases_df) == 0:
        st.warning("没有找到风险案例。")
    else:
        case_df = risk_cases_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

        selected_idx = st.selectbox(
            "选择一个高风险交易案例",
            options=list(range(len(case_df))),
            format_func=lambda i: f"案例 {i+1}｜{case_df.loc[i, 'type']}｜金额 {money(case_df.loc[i, 'amount'])}｜风险分 {float(case_df.loc[i, 'risk_score']):.4f}"
        )

        row = case_df.loc[selected_idx]

        left, right = st.columns([0.85, 1.15])

        with left:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">交易信息</div>
                    <div class="panel-text">
                    交易类型：<b>{row['type']}</b><br>
                    交易金额：<b>{money(row['amount'])}</b><br>
                    付款账户：<b>{row['nameOrig']}</b><br>
                    收款账户：<b>{row['nameDest']}</b><br>
                    付款方交易前余额：<b>{money(row['oldbalanceOrg'])}</b><br>
                    付款方交易后余额：<b>{money(row['newbalanceOrig'])}</b><br>
                    真实标签 isFraud：<b>{row.get('isFraud', '未知')}</b><br>
                    风险分数：<b>{float(row['risk_score']):.4f}</b><br>
                    风险等级：{risk_chip(row['risk_level'])}<br>
                    处置建议：<b>{row['risk_action']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            reason = safe_text(row, "business_risk_reason", "risk_reason")
            staff = safe_text(row, "business_staff_explanation", "staff_explanation")

            st.markdown(
                f"""
                <div class="case-highlight">
                    <b>业务化风险原因</b><br><br>
                    {reason}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            st.markdown(
                f"""
                <div class="case-highlight">
                    <b>风控人员审核解释</b><br><br>
                    {staff}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# Tab 3 模型解释
# =========================

with tab3:
    if global_df.empty:
        st.warning("未找到模型解释文件 global_feature_importance.csv。")
    else:
        left, right = st.columns([1.1, 0.9])

        top_df = global_df.head(10).sort_values("mean_abs_shap", ascending=True)

        with left:
            fig = px.bar(
                top_df,
                x="mean_abs_shap",
                y="feature_cn",
                orientation="h",
                text="mean_abs_shap",
                title="模型最关注的风险特征 Top 10"
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(
                height=500,
                xaxis_title="平均绝对 SHAP 值",
                yaxis_title="特征名称",
                margin=dict(l=20, r=50, t=60, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            top3 = global_df.head(3)
            top3_html = ""
            for _, item in top3.iterrows():
                top3_html += f"• <b>{item['feature_cn']}</b>：{item['mean_abs_shap']:.4f}<br>"

            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">模型解释结论</div>
                    <div class="panel-text">
                    本项目不是只输出一个黑箱分数，而是进一步解释模型为什么认为交易有风险。
                    <br><br>
                    当前模型最关注的前三个特征为：
                    <br><br>
                    {top3_html}
                    <br>
                    这说明系统主要从账户余额变化、交易后余额状态、资金流出强度等角度识别欺诈风险，
                    与真实金融反诈业务逻辑具有一致性。
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# Tab 4 客户保护
# =========================

with tab4:
    if len(risk_cases_df) == 0:
        st.warning("没有找到风险案例。")
    else:
        case_df = risk_cases_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        row = case_df.iloc[0]

        customer_warning = safe_text(row, "business_customer_warning", "customer_warning")
        staff = safe_text(row, "business_staff_explanation", "staff_explanation")

        left, right = st.columns([0.8, 1.2])

        with left:
            st.markdown(
                f"""
                <div class="phone">
                    <div class="screen">
                        <div class="sms-title">银行安全提醒</div>
                        <div class="sms-bubble">
                        {customer_warning}
                        </div>
                        <br>
                        <div style="font-size:0.82rem;color:#64748B;line-height:1.7;">
                        说明：客户侧不展示复杂模型术语，而是以温和、清晰、可行动的方式提醒用户核实交易。
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">为什么这能优化客户体验？</div>
                    <div class="panel-text">
                    传统风控容易出现两种问题：一种是只拦截不解释，客户会觉得体验差；
                    另一种是完全放行，可能导致资金损失。
                    <br><br>
                    FraudShield 的做法是根据风险等级进行分层处置：
                    <br><br>
                    低风险：轻量提示，不打断交易。<br>
                    中风险：二次确认，引导客户核实。<br>
                    高风险：暂缓交易，进入人工复核。<br>
                    <br>
                    因此，系统既能保护客户资金安全，也能减少正常客户被误伤。
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            st.markdown(
                f"""
                <div class="case-highlight">
                    <b>后台同步给风控人员的解释</b><br><br>
                    {staff}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================
# Tab 5 落地价值
# =========================

with tab5:
    left, right = st.columns([0.9, 1.1])

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">落地路径</div>

                <div class="roadmap-step">
                    <div class="roadmap-title">第一步：本地原型验证</div>
                    <div class="roadmap-desc">
                    使用公开模拟交易数据完成模型训练、风险评分、模型解释和可视化展示。
                    </div>
                </div>

                <div class="roadmap-step">
                    <div class="roadmap-title">第二步：云端展示部署</div>
                    <div class="roadmap-desc">
                    将系统部署到云服务器或 Streamlit Cloud，形成可访问的网址，用于比赛展示和团队测试。
                    </div>
                </div>

                <div class="roadmap-step">
                    <div class="roadmap-title">第三步：金融机构试点</div>
                    <div class="roadmap-desc">
                    接入银行或支付机构的交易流水接口，对实时交易输出风险分数、风险原因和处置建议。
                    </div>
                </div>

                <div class="roadmap-step">
                    <div class="roadmap-title">第四步：内网生产部署</div>
                    <div class="roadmap-desc">
                    在金融机构内网部署模型服务、风控后台、权限管理和日志审计，形成正式反诈辅助系统。
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">商业应用价值</div>
                <div class="panel-text">
                <b>适用机构</b><br>
                银行、支付机构、消费金融平台、互联网金融平台以及金融科技服务企业。
                <br><br>

                <b>核心价值</b><br>
                通过自动风险评分减少人工逐笔排查压力，
                通过分级处置降低正常客户被误拦截的概率，
                通过可解释结果提高风控审核效率和模型可信度。
                <br><br>

                <b>部署方式</b><br>
                系统可部署在金融机构内网环境，并通过标准接口接入交易系统。
                对每笔交易实时返回风险分数、风险等级、风险原因和处置建议。
                <br><br>

                <b>推广方式</b><br>
                可采用本地化部署、系统接口接入或风控辅助服务等方式，
                根据不同金融机构的业务规模和数据安全要求进行适配。
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )