
import html
import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. 页面与路径
# =========================================================

st.set_page_config(
    page_title="FraudShield 实时风险检测",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime_assets"

MODEL_PATH = RUNTIME_DIR / "fraud_model.joblib"
FEATURES_PATH = RUNTIME_DIR / "feature_columns.joblib"
METADATA_PATH = RUNTIME_DIR / "model_metadata.json"

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

SUPPORTED_TYPES = [
    "TRANSFER",
    "CASH_OUT",
    "PAYMENT",
    "CASH_IN",
    "DEBIT",
]

MAX_BATCH_ROWS = 100_000


# =========================================================
# 2. 页面样式
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 7%, rgba(37, 99, 235, 0.12), transparent 27%),
            radial-gradient(circle at 93% 12%, rgba(16, 185, 129, 0.09), transparent 25%),
            linear-gradient(180deg, #F8FAFC 0%, #EEF3F8 100%);
    }

    .hero {
        border-radius: 28px;
        padding: 30px 34px;
        margin-bottom: 22px;
        color: white;
        background: linear-gradient(135deg, #0F172A 0%, #173B75 52%, #2563EB 100%);
        box-shadow: 0 22px 52px rgba(15, 23, 42, 0.20);
    }

    .hero-kicker {
        display: inline-block;
        padding: 6px 12px;
        margin-bottom: 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.24);
        background: rgba(255,255,255,0.13);
        font-size: 0.86rem;
    }

    .hero-title {
        margin: 0 0 10px 0;
        font-size: 2.25rem;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .hero-desc {
        max-width: 950px;
        color: rgba(255,255,255,0.88);
        font-size: 1.02rem;
        line-height: 1.85;
    }

    .soft-card {
        height: 100%;
        padding: 20px 22px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 22px;
        background: rgba(255,255,255,0.93);
        box-shadow: 0 12px 30px rgba(15,23,42,0.07);
    }

    .result-card {
        padding: 20px 22px;
        border-radius: 22px;
        background: white;
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 13px 32px rgba(15,23,42,0.08);
    }

    .metric-label {
        color: #64748B;
        font-size: 0.86rem;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: -0.035em;
    }

    .risk-high {
        color: #991B1B;
        background: #FEE2E2;
        border: 1px solid #FECACA;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 800;
    }

    .risk-mid {
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 800;
    }

    .risk-low {
        color: #075985;
        background: #E0F2FE;
        border: 1px solid #BAE6FD;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 800;
    }

    .risk-normal {
        color: #166534;
        background: #DCFCE7;
        border: 1px solid #BBF7D0;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 800;
    }

    .explain-box {
        margin-top: 12px;
        padding: 18px 20px;
        border-left: 5px solid #2563EB;
        border-radius: 16px;
        background: white;
        color: #334155;
        line-height: 1.85;
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }

    .notice {
        padding: 14px 16px;
        border-radius: 16px;
        color: #475569;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        line-height: 1.75;
        font-size: 0.91rem;
    }

    .small-note {
        color: #64748B;
        font-size: 0.84rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. 加载可信模型
# =========================================================

@st.cache_resource
def load_runtime_assets():
    missing = [
        path for path in [MODEL_PATH, FEATURES_PATH, METADATA_PATH]
        if not path.exists()
    ]

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "缺少云端运行资产：\n"
            f"{missing_text}\n"
            "请先运行 src/10_prepare_runtime_assets.py。"
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    threshold = float(
        metadata.get(
            "selected_threshold",
            metadata.get("prediction_threshold", 0.5),
        )
    )

    return model, feature_columns, metadata, threshold


try:
    model, feature_columns, metadata, decision_threshold = (
        load_runtime_assets()
    )
except Exception as exc:
    st.error("模型运行资产加载失败。")
    st.code(str(exc))
    st.stop()


# =========================================================
# 4. 特征工程：必须与 Stage 5.2 deployable 完全一致
# =========================================================

def build_deployable_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    step = pd.to_numeric(df["step"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce")
    old_origin = pd.to_numeric(df["oldbalanceOrg"], errors="coerce")
    old_dest = pd.to_numeric(df["oldbalanceDest"], errors="coerce")

    X["step"] = step.astype(np.float32)
    X["hour_of_day"] = ((step - 1) % 24).astype(np.float32)
    X["day_index"] = ((step - 1) // 24).astype(np.float32)

    X["amount"] = amount.astype(np.float32)
    X["amount_log"] = np.log1p(amount).astype(np.float32)

    X["oldbalanceOrg"] = old_origin.astype(np.float32)
    X["oldbalanceDest"] = old_dest.astype(np.float32)

    X["origin_balance_zero_before"] = (
        old_origin == 0
    ).astype(np.float32)

    X["dest_balance_zero_before"] = (
        old_dest == 0
    ).astype(np.float32)

    X["amount_origin_ratio"] = (
        amount / (old_origin + 1.0)
    ).clip(upper=1000).astype(np.float32)

    X["amount_dest_ratio"] = (
        amount / (old_dest + 1.0)
    ).clip(upper=1000).astype(np.float32)

    type_dummies = pd.get_dummies(
        df["type"].astype(str).str.upper().str.strip(),
        prefix="type",
        dtype=np.float32,
    )

    X = pd.concat([X, type_dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.reindex(columns=feature_columns, fill_value=0)
    X = X.astype(np.float32)

    return X


# =========================================================
# 5. 业务分级与文本
# =========================================================

def risk_level(score: float) -> str:
    # 高风险阈值来自自然验证集；其余等级仅用于比赛 Demo 分流展示。
    if score >= decision_threshold:
        return "高风险"
    if score >= decision_threshold * 0.60:
        return "中风险"
    if score >= decision_threshold * 0.25:
        return "低风险"
    return "正常"


def risk_action(level: str) -> str:
    actions = {
        "高风险": "建议暂缓交易，并进入人工复核或强化身份验证",
        "中风险": "建议进行二次确认，核实收款方身份和交易目的",
        "低风险": "建议轻量提示，确认交易由本人发起",
        "正常": "风险较低，可按常规流程处理",
    }
    return actions[level]


def build_business_reason(row: pd.Series, score: float) -> str:
    reasons = []

    tx_type = str(row["type"]).upper()
    amount = float(row["amount"])
    old_origin = float(row["oldbalanceOrg"])
    old_dest = float(row["oldbalanceDest"])

    ratio = amount / (old_origin + 1.0)
    hour = int((int(row["step"]) - 1) % 24)

    if tx_type == "TRANSFER":
        reasons.append("该交易属于资金直接转移的转账场景")
    elif tx_type == "CASH_OUT":
        reasons.append("该交易属于资金快速流出的现金转出场景")

    if 0.90 <= ratio <= 1.10 and old_origin > 0:
        reasons.append("交易金额与付款方交易前余额高度接近，呈现账户资金集中转出的特征")
    elif ratio > 1.0 and old_origin > 0:
        reasons.append("交易金额高于付款方记录的交易前余额，需要进一步核实资金来源与记账状态")

    if amount >= 100_000:
        reasons.append(f"交易金额为人民币 {amount:,.2f} 元，潜在资金损失较大")
    elif amount >= 10_000:
        reasons.append(f"交易金额为人民币 {amount:,.2f} 元，达到重点关注金额区间")

    if old_dest == 0 and tx_type in {"TRANSFER", "CASH_OUT"}:
        reasons.append("收款方交易前余额为零，建议结合账户历史和关联关系进一步核查")

    if hour <= 5:
        reasons.append("交易发生在凌晨时段，可结合用户历史作息进一步验证")

    if not reasons:
        reasons.append("当前风险分数由交易金额、交易类型、交易时间及账户交易前状态综合形成")

    return "；".join(reasons[:4])


def customer_warning(row: pd.Series, level: str) -> str:
    amount = float(row["amount"])
    tx_type_cn = {
        "TRANSFER": "转账",
        "CASH_OUT": "现金转出",
        "PAYMENT": "支付",
        "CASH_IN": "现金转入",
        "DEBIT": "借记",
    }.get(str(row["type"]).upper(), "金融")

    if level == "高风险":
        return (
            f"系统检测到您当前人民币 {amount:,.2f} 元的{tx_type_cn}交易存在较高风险。"
            "请确认收款方身份和交易目的，不要轻信陌生人提出的投资理财、"
            "刷单返利、冒充客服或安全账户转账要求。如非本人真实意愿，"
            "请暂停交易并联系官方客服。"
        )

    if level == "中风险":
        return (
            f"当前人民币 {amount:,.2f} 元的{tx_type_cn}交易存在一定异常。"
            "为保障资金安全，请再次核实收款方身份和交易用途。"
        )

    if level == "低风险":
        return (
            "当前交易存在轻微异常特征，请确认交易由本人操作，"
            "并核对收款方信息后继续。"
        )

    return "当前交易风险较低，请仍注意保护账户与验证码信息。"


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    X = build_deployable_features(df)
    scores = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["risk_score"] = scores
    result["predict_isFraud"] = (
        result["risk_score"] >= decision_threshold
    ).astype(int)
    result["risk_level"] = result["risk_score"].apply(risk_level)
    result["risk_action"] = result["risk_level"].apply(risk_action)

    result["business_risk_reason"] = [
        build_business_reason(row, float(score))
        for (_, row), score in zip(result.iterrows(), scores)
    ]

    result["customer_warning"] = [
        customer_warning(row, level)
        for (_, row), level in zip(
            result.iterrows(),
            result["risk_level"],
        )
    ]

    return result


# =========================================================
# 6. 输入检查
# =========================================================

def validate_input_dataframe(df: pd.DataFrame) -> list[str]:
    errors = []

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append("缺少必要字段：" + "、".join(missing))
        return errors

    if len(df) == 0:
        errors.append("文件中没有交易记录。")
        return errors

    if len(df) > MAX_BATCH_ROWS:
        errors.append(
            f"单次最多处理 {MAX_BATCH_ROWS:,} 行，"
            f"当前文件有 {len(df):,} 行。"
        )

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    for col in numeric_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid_count = int(converted.isna().sum())
        if invalid_count > 0:
            errors.append(
                f"字段 {col} 有 {invalid_count} 个非数字或空值。"
            )

    type_series = df["type"].astype(str).str.upper().str.strip()
    unknown_types = sorted(
        set(type_series) - set(SUPPORTED_TYPES)
    )
    if unknown_types:
        errors.append(
            "发现不支持的交易类型："
            + "、".join(unknown_types[:10])
        )

    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        if (amount < 0).any():
            errors.append("交易金额 amount 不能为负数。")

    for col in ["oldbalanceOrg", "oldbalanceDest"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if (values < 0).any():
                errors.append(f"{col} 不能为负数。")

    if "step" in df.columns:
        step = pd.to_numeric(df["step"], errors="coerce")
        if (step < 1).any():
            errors.append("时间步 step 必须大于或等于 1。")

    return errors


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    normalized["type"] = (
        normalized["type"].astype(str).str.upper().str.strip()
    )

    for col in [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]:
        normalized[col] = pd.to_numeric(
            normalized[col],
            errors="raise",
        )

    normalized["step"] = normalized["step"].astype(int)
    return normalized


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    last_error = None

    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "无法识别 CSV 编码，请另存为 UTF-8 CSV 后重新上传。"
    ) from last_error


def result_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


def result_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="风险检测结果",
            index=False,
        )
    return buffer.getvalue()


def badge(level: str) -> str:
    css_class = {
        "高风险": "risk-high",
        "中风险": "risk-mid",
        "低风险": "risk-low",
        "正常": "risk-normal",
    }[level]
    return f'<span class="{css_class}">{html.escape(level)}</span>'


# =========================================================
# 7. 页面主体
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">可信模型 · 事前特征 · 在线交互</div>
        <div class="hero-title">FraudShield 实时交易风险检测</div>
        <div class="hero-desc">
            使用自然时间外验证后选定的模型，对交易发起时可获得的信息进行即时评分。
            支持单笔交易检测和 CSV 批量筛查，输出风险分数、分级处置建议及客户提醒。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_info = st.tabs(
    ["单笔实时检测", "CSV 批量检测", "模型与使用说明"]
)


# =========================================================
# 8. 单笔检测
# =========================================================

with tab_single:
    st.subheader("输入一笔待检测交易")
    st.caption(
        "当前模型只使用交易发起时可获得的信息，"
        "不要求填写交易后的账户余额。"
    )

    with st.form("single_transaction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            step = st.number_input(
                "交易时间步 step",
                min_value=1,
                value=521,
                step=1,
                help="PaySim 中一个 step 可理解为一个小时。",
            )

            tx_type = st.selectbox(
                "交易类型 type",
                SUPPORTED_TYPES,
                index=0,
            )

        with c2:
            amount = st.number_input(
                "交易金额（人民币元）",
                min_value=0.0,
                value=86070.17,
                step=100.0,
                format="%.2f",
            )

            old_origin = st.number_input(
                "付款方交易前余额",
                min_value=0.0,
                value=86070.17,
                step=100.0,
                format="%.2f",
            )

        with c3:
            old_dest = st.number_input(
                "收款方交易前余额",
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f",
            )

            name_orig = st.text_input(
                "付款账户（可选）",
                value="DEMO_ORIGIN",
            )

            name_dest = st.text_input(
                "收款账户（可选）",
                value="DEMO_DEST",
            )

        submitted = st.form_submit_button(
            "开始风险检测",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        one_row = pd.DataFrame(
            [
                {
                    "step": int(step),
                    "type": tx_type,
                    "amount": float(amount),
                    "oldbalanceOrg": float(old_origin),
                    "oldbalanceDest": float(old_dest),
                    "nameOrig": name_orig,
                    "nameDest": name_dest,
                }
            ]
        )

        result = predict_dataframe(one_row).iloc[0]

        st.markdown("### 检测结果")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">模型风险分数</div>
                    <div class="metric-value">{result['risk_score']:.4f}</div>
                    <div class="small-note">
                        高风险判定阈值：{decision_threshold:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">风险等级</div>
                    <div style="margin-top:12px;">
                        {badge(result['risk_level'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">模型判定</div>
                    <div class="metric-value">
                        {'疑似欺诈' if result['predict_isFraud'] == 1 else '暂未判为欺诈'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="explain-box">
                <b>业务风险提示</b><br><br>
                {html.escape(str(result['business_risk_reason']))}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="explain-box">
                <b>处置建议</b><br><br>
                {html.escape(str(result['risk_action']))}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="explain-box">
                <b>客户提醒话术</b><br><br>
                {html.escape(str(result['customer_warning']))}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "当前“业务风险提示”为基于输入字段生成的可读说明，"
            "不等同于该笔交易的在线 SHAP 因果解释。"
        )


# =========================================================
# 9. 批量检测
# =========================================================

with tab_batch:
    st.subheader("上传 CSV 批量检测")
    st.caption(
        f"单次最多处理 {MAX_BATCH_ROWS:,} 笔交易。"
        "系统不会要求上传真实银行卡号、身份证号或手机号。"
    )

    template_df = pd.DataFrame(
        [
            {
                "step": 521,
                "type": "TRANSFER",
                "amount": 86070.17,
                "oldbalanceOrg": 86070.17,
                "oldbalanceDest": 0.0,
                "nameOrig": "DEMO_ORIGIN_001",
                "nameDest": "DEMO_DEST_001",
            },
            {
                "step": 522,
                "type": "PAYMENT",
                "amount": 1280.00,
                "oldbalanceOrg": 12000.00,
                "oldbalanceDest": 35000.00,
                "nameOrig": "DEMO_ORIGIN_002",
                "nameDest": "DEMO_DEST_002",
            },
        ]
    )

    st.download_button(
        "下载 CSV 模板",
        data=result_to_csv_bytes(template_df),
        file_name="fraudshield_batch_template.csv",
        mime="text/csv",
        use_container_width=False,
    )

    uploaded_file = st.file_uploader(
        "选择待检测 CSV 文件",
        type=["csv"],
        accept_multiple_files=False,
        help=(
            "必要字段：step、type、amount、"
            "oldbalanceOrg、oldbalanceDest。"
        ),
    )

    if uploaded_file is not None:
        try:
            uploaded_df = read_uploaded_csv(uploaded_file)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        st.write(
            f"已读取 **{len(uploaded_df):,}** 行、"
            f"**{len(uploaded_df.columns)}** 个字段。"
        )

        st.dataframe(
            uploaded_df.head(20),
            use_container_width=True,
            height=300,
        )

        errors = validate_input_dataframe(uploaded_df)

        if errors:
            st.error("文件暂时不能检测，请先修正以下问题：")
            for error in errors:
                st.write("• " + error)
        else:
            st.success("字段检查通过，可以开始批量检测。")

            if st.button(
                "开始批量风险检测",
                type="primary",
                use_container_width=True,
            ):
                try:
                    normalized_df = normalize_dataframe(uploaded_df)

                    with st.spinner("正在计算风险分数……"):
                        batch_result = predict_dataframe(normalized_df)

                    st.session_state["batch_result"] = batch_result

                except Exception as exc:
                    st.error("批量检测失败。")
                    st.code(str(exc))

    if "batch_result" in st.session_state:
        batch_result = st.session_state["batch_result"]

        st.markdown("### 批量检测结果")

        high_count = int(
            (batch_result["risk_level"] == "高风险").sum()
        )
        mid_count = int(
            (batch_result["risk_level"] == "中风险").sum()
        )
        fraud_count = int(
            batch_result["predict_isFraud"].sum()
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总交易数", f"{len(batch_result):,}")
        c2.metric("高风险", f"{high_count:,}")
        c3.metric("中风险", f"{mid_count:,}")
        c4.metric("疑似欺诈", f"{fraud_count:,}")

        risk_order = ["高风险", "中风险", "低风险", "正常"]
        selected_levels = st.multiselect(
            "筛选展示风险等级",
            risk_order,
            default=risk_order,
        )

        display_df = batch_result[
            batch_result["risk_level"].isin(selected_levels)
        ].sort_values(
            "risk_score",
            ascending=False,
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=430,
        )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "下载 CSV 检测结果",
                data=result_to_csv_bytes(batch_result),
                file_name="fraudshield_detection_result.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            try:
                excel_bytes = result_to_excel_bytes(batch_result)
                st.download_button(
                    "下载 Excel 检测结果",
                    data=excel_bytes,
                    file_name="fraudshield_detection_result.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )
            except ImportError:
                st.warning(
                    "未安装 openpyxl，目前只能下载 CSV。"
                )


# =========================================================
# 10. 模型说明
# =========================================================

with tab_info:
    st.subheader("可信模型与使用边界")

    natural_metrics = metadata.get(
        "natural_test_metrics",
        {},
    )

    i1, i2, i3, i4 = st.columns(4)
    i1.metric(
        "自然测试交易数",
        f"{int(natural_metrics.get('rows', 0)):,}",
    )
    i2.metric(
        "自然测试欺诈比例",
        f"{natural_metrics.get('fraud_prevalence', 0) * 100:.4f}%",
    )
    i3.metric(
        "测试 Precision",
        f"{natural_metrics.get('precision', 0) * 100:.2f}%",
    )
    i4.metric(
        "测试 Recall",
        f"{natural_metrics.get('recall', 0) * 100:.2f}%",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""
            <div class="soft-card">
                <b>当前模型</b><br><br>
                模型类型：{html.escape(str(metadata.get('model_name', '未知')))}<br>
                特征集合：deployable（交易发起时可获得）<br>
                决策阈值：{decision_threshold:.6f}<br>
                特征数量：{len(feature_columns)}<br><br>
                模型选择依据：自然验证集 PR-AUC/AP，而不是最终测试集。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="soft-card">
                <b>重要使用边界</b><br><br>
                1. 当前模型基于 PaySim 公开模拟交易数据。<br>
                2. 结果用于比赛原型和技术可行性展示，不替代金融机构正式决策。<br>
                3. 生产部署需要使用真实机构数据重新训练、校准阈值并接受合规审查。<br>
                4. 不应在公开演示环境上传真实敏感金融或身份信息。
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="notice">
            <b>字段说明：</b><br>
            step：交易时间步；type：交易类型；amount：交易金额；
            oldbalanceOrg：付款方交易前余额；
            oldbalanceDest：收款方交易前余额。<br><br>
            nameOrig 和 nameDest 仅用于结果表中标识交易，可以不填写，
            且不参与模型预测。
        </div>
        """,
        unsafe_allow_html=True,
    )
