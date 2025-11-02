# =======================================
# 💹 财务分析仪表盘（AI预测 + 导出版）
# =======================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import io

# -------------------------
# 页面配置
# -------------------------
st.set_page_config(page_title="财务分析仪表盘", page_icon="💼", layout="wide")
st.title("💹 智能财务分析仪表盘")
st.markdown("通过交互式图表 + AI预测，洞察企业财务趋势。")

# -------------------------
# 模拟数据加载
# -------------------------
@st.cache_data(ttl=3600)
def load_data():
    years = np.arange(2018, 2025)
    df = pd.DataFrame({
        "年份": years,
        "营业收入": np.random.randint(80, 150, len(years)),
        "净利润": np.random.randint(10, 50, len(years)),
        "负债率": np.random.uniform(20, 60, len(years)),
        "每股收益": np.random.uniform(1.5, 4.5, len(years)),
    })
    df["收入同比(%)"] = df["营业收入"].pct_change() * 100
    df["利润同比(%)"] = df["净利润"].pct_change() * 100
    return df

df = load_data()

# -------------------------
# AI预测模块
# -------------------------
def ai_forecast(df, col_name, predict_years=1):
    """用线性回归预测未来若干年数据"""
    model = LinearRegression()
    X = df[["年份"]]
    y = df[col_name]
    model.fit(X, y)
    next_year = df["年份"].max() + np.arange(1, predict_years + 1)
    y_pred = model.predict(next_year.reshape(-1, 1))
    forecast_df = pd.DataFrame({"年份": next_year, col_name: y_pred})
    return forecast_df

# -------------------------
# 侧边栏交互区
# -------------------------
with st.sidebar:
    st.header("⚙️ 控制面板")
    metrics = st.multiselect(
        "选择要分析的财务指标：",
        ["营业收入", "净利润", "负债率", "每股收益"],
        ["营业收入", "净利润"]
    )
    show_growth = st.checkbox("显示同比增长率", True)
    do_forecast = st.checkbox("启用 AI 模型预测未来1年", True)
    st.markdown("---")
    st.markdown("📅 数据范围：2018 - 2024")
    st.markdown("<small style='color:gray'>数据来源：示例生成</small>", unsafe_allow_html=True)

# -------------------------
# 主体布局
# -------------------------
col1, col2 = st.columns([2, 1])

# ========== 左侧：趋势与预测 ==========
with col1:
    st.subheader("📈 财务指标趋势")

    plot_df = df.copy()

    # AI预测
    if do_forecast:
        for col in ["营业收入", "净利润"]:
            if col in df.columns:
                forecast_df = ai_forecast(df, col)
                forecast_df["预测"] = True
                plot_df["预测"] = False
                plot_df = pd.concat([plot_df, forecast_df], ignore_index=True)
        st.info("🔮 AI 模型预测已启用：预测下一年收入与利润趋势")

    # 折线图
    fig = px.line(
        plot_df,
        x="年份",
        y=metrics,
        color_discrete_sequence=px.colors.qualitative.Set2,
        markers=True,
        title="主要财务指标趋势"
    )
    st.plotly_chart(fig, width="stretch")

    if show_growth:
        st.markdown("#### 同比变化率")
        fig_growth = px.bar(
            df,
            x="年份",
            y=["收入同比(%)", "利润同比(%)"],
            barmode="group",
            text_auto=".1f",
            title="收入与利润同比变化率 (%)"
        )
        st.plotly_chart(fig_growth, width="stretch")

# ========== 右侧：指标卡片 ==========
with col2:
    st.subheader("📊 关键财务指标")
    st.metric("营业收入（最新）", f"{df['营业收入'].iloc[-1]} 亿元", f"{df['收入同比(%)'].iloc[-1]:.1f}%")
    st.metric("净利润（最新）", f"{df['净利润'].iloc[-1]} 亿元", f"{df['利润同比(%)'].iloc[-1]:.1f}%")
    st.metric("负债率", f"{df['负债率'].iloc[-1]:.1f} %")
    st.metric("每股收益", f"{df['每股收益'].iloc[-1]:.2f} 元")

    st.markdown("---")
    st.markdown("#### 🧾 指标说明")
    st.markdown("""
    - **营业收入**：主营业务总收入  
    - **净利润**：扣除成本、费用后的净收益  
    - **负债率**：总负债 / 总资产 × 100%  
    - **每股收益**：净利润 / 流通股本  
    """)

# ========== 导出 Excel ==========
st.markdown("---")
st.subheader("📤 导出报告")

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="原始数据")
    if do_forecast:
        forecast_df.to_excel(writer, index=False, sheet_name="AI预测")

st.download_button(
    label="📥 下载财务分析报告（Excel）",
    data=output.getvalue(),
    file_name="财务分析报告.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ========== 折叠原始数据 ==========
with st.expander("📘 查看原始数据"):
    st.dataframe(df, use_container_width=True)

st.markdown("<small style='color:gray'>© 2025 智能财务分析仪表盘 - 由 Streamlit 构建</small>", unsafe_allow_html=True)
