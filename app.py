import streamlit as st

import pandas as pd

import joblib

import plotly.express as px

import plotly.graph_objects as go



from assistant import ask_assistant





# =========================================================

# PAGE CONFIG

# =========================================================



st.set_page_config(

    page_title="AI Business Analyst",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded"

)





# =========================================================

# CUSTOM CSS

# =========================================================



st.markdown(
    """
    <style>
    /* ===== Minimal UI ===== */

    .stApp {
        font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #fafafa;
        color: #202124;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #eeeeee;
    }

    /* Main title */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #202124;
        margin: 0;
    }

    .subtitle {
        color: #777777;
        font-size: 14px;
        margin-top: 7px;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 650;
        color: #202124;
        margin: 8px 0 16px;
    }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #eeeeee;
        border-radius: 12px;
        padding: 17px 19px;
        min-height: 100px;
        box-shadow: 0 2px 8px rgba(0,0,0,.025);
    }

    .kpi-card:hover {
        border-color: #dddddd;
    }

    .kpi-title {
        color: #888888;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    .kpi-value {
        color: #202124;
        font-size: 27px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #777777 !important;
        background: transparent !important;
        font-weight: 550 !important;
        padding: 9px 14px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #202124 !important;
        background: #f4f4f4 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="tab-highlight"] {
        background: #202124 !important;
        height: 2px !important;
    }

    /* Inputs */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #999999 !important;
        box-shadow: none !important;
    }

    /* Buttons */
    .stButton > button {
        background: #ffffff;
        color: #333333;
        border: 1px solid #dddddd;
        border-radius: 8px;
        font-weight: 550;
        box-shadow: none;
    }

    .stButton > button:hover {
        background: #f5f5f5;
        border-color: #bbbbbb;
    }

    button[kind="primary"] {
        background: #202124 !important;
        color: #ffffff !important;
        border: 1px solid #202124 !important;
        box-shadow: none !important;
    }

    button[kind="primary"]:hover {
        background: #333333 !important;
    }

    /* Chat */
    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #eeeeee;
        border-radius: 12px;
        margin-bottom: 8px;
    }

    div[data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 10px !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 9px;
        border: 1px solid #eeeeee;
    }

    /* Plotly */
    .js-plotly-plot {
        background: #ffffff;
        border: 1px solid #ededed;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,.02);
    }

    hr {
        border-color: #eeeeee !important;
        margin: 1rem 0 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #555555;
    }

    section[data-testid="stSidebar"] label {
        color: #555555 !important;
        font-size: 13px !important;
    }

    .footer {
        text-align: center;
        color: #999999;
        padding: 30px 0 8px;
        font-size: 12px;
    }

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True
)





# =========================================================

# SESSION STATE

# =========================================================



if "gemini_api_key" not in st.session_state:

    st.session_state.gemini_api_key = ""



if "messages" not in st.session_state:

    st.session_state.messages = []



if "chat_history" not in st.session_state:

    st.session_state.chat_history = []





# =========================================================

# LOAD DATA

# =========================================================



@st.cache_data

def load_data():



    df = pd.read_csv(

        "data/superstore_clean.csv"

    )



    segments = pd.read_csv(

        "data/customer_segments.csv"

    )



    return df, segments





# =========================================================

# LOAD MODEL

# =========================================================



@st.cache_resource

def load_model():



    model = joblib.load(

        "xgboost_profit_model.pkl"

    )



    return model





# =========================================================

# TRY LOAD

# =========================================================



try:



    df, customer_segments = load_data()



    profit_model = load_model()



except Exception as e:



    st.error(

        f"ไม่สามารถโหลดข้อมูลหรือ Model ได้: {e}"

    )



    st.stop()





# =========================================================

# DATA PREPARATION

# =========================================================



# Make sure dates are datetime



if "Order Date" in df.columns:



    df["Order Date"] = pd.to_datetime(

        df["Order Date"],

        errors="coerce"

    )

# =========================================================
# INTERACTIVE DASHBOARD HELPERS
# =========================================================

def apply_dashboard_filters(data, selected_category, selected_region, selected_segment):
    filtered = data.copy()

    if selected_category != "All":
        filtered = filtered[filtered["Category"] == selected_category]

    if selected_region != "All":
        filtered = filtered[filtered["Region"] == selected_region]

    if selected_segment != "All":
        filtered = filtered[filtered["Segment"] == selected_segment]

    return filtered






# =========================================================

# SIDEBAR

# =========================================================



with st.sidebar:



    st.markdown(

        """

        <h2>📊 AI Business Analyst</h2>

        """,

        unsafe_allow_html=True

    )



    st.markdown("---")





    # =====================================================

    # GEMINI API

    # =====================================================



    st.markdown("### 🔑 Gemini API")



    api_key = st.text_input(

        "Gemini API Key",

        type="password",

        value=st.session_state.gemini_api_key,

        placeholder="AIza...",

        help="API Key จะถูกใช้เฉพาะใน session นี้"

    )





    if api_key:



        st.session_state.gemini_api_key = api_key



        st.success(

            "✅ API Key connected"

        )



    else:



        st.warning(

            "⚠️ กรุณากรอก API Key"

        )





    st.markdown("---")





    # =====================================================

    # CAPABILITIES

    # =====================================================



    st.markdown("### 🤖 AI Capabilities")



    st.markdown(

        """

        ✓ Customer Segmentation



        ✓ Profit Prediction



        ✓ Business Analysis



        ✓ Natural Language Q&A

        """

    )





    st.markdown("---")





    # =====================================================

    # MODEL STATUS

    # =====================================================



    st.markdown("### ⚙️ Model Status")



    st.success(

        "K-Means: Loaded"

    )



    st.success(

        "XGBoost: Loaded"

    )



    if st.session_state.gemini_api_key:



        st.success(

            "Gemini API: Connected"

        )



    else:



        st.warning(

            "Gemini API: Not Connected"

        )





# =========================================================

# HEADER

# =========================================================



st.markdown(

    '<div class="main-title">📊 AI Business Analyst</div>',

    unsafe_allow_html=True

)



st.markdown(

    """

    <div class="subtitle">

    AI-powered business analytics using Machine Learning and Gemini

    </div>

    """,

    unsafe_allow_html=True

)





# =========================================================

# KPI CALCULATIONS

# =========================================================



total_sales = df["Sales"].sum()



total_profit = df["Profit"].sum()



total_orders = df["Order ID"].nunique()



total_customers = df["Customer ID"].nunique()





# =========================================================

# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Total Sales</div><div class="kpi-value">${total_sales:,.2f}</div></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Total Profit</div><div class="kpi-value">${total_profit:,.2f}</div></div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Orders</div><div class="kpi-value">{total_orders:,}</div></div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Customers</div><div class="kpi-value">{total_customers:,}</div></div>',
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)





# =========================================================

# TABS

# =========================================================



tab1, tab2, tab3, tab4 = st.tabs(

    [

        "📊 Overview",

        "👥 Customer Analytics",

        "💰 Profit Prediction",

        "💬 AI Assistant"

    ]

)





# =========================================================
# TAB 1 : OVERVIEW
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">📊 Business Overview</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # GLOBAL DASHBOARD FILTERS
    # -----------------------------------------------------

    st.markdown("#### 🎛️ Dashboard Filters")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        selected_category = st.selectbox(
            "Category",
            ["All"] + sorted(df["Category"].dropna().unique().tolist()),
            key="overview_category"
        )

    with filter_col2:
        selected_region = st.selectbox(
            "Region",
            ["All"] + sorted(df["Region"].dropna().unique().tolist()),
            key="overview_region"
        )

    with filter_col3:
        selected_segment = st.selectbox(
            "Customer Segment",
            ["All"] + sorted(df["Segment"].dropna().unique().tolist()),
            key="overview_segment"
        )

    with filter_col4:
        min_date = df["Order Date"].min()
        max_date = df["Order Date"].max()

        selected_dates = st.date_input(
            "Order Date",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="overview_dates"
        )

    dashboard_df = apply_dashboard_filters(
        df,
        selected_category,
        selected_region,
        selected_segment
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        dashboard_df = dashboard_df[
            (dashboard_df["Order Date"].dt.date >= start_date) &
            (dashboard_df["Order Date"].dt.date <= end_date)
        ]

    # -----------------------------------------------------
    # DYNAMIC KPIs
    # -----------------------------------------------------

    st.markdown("#### 📌 Current Selection")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Total Sales", f"${dashboard_df['Sales'].sum():,.0f}")

    with k2:
        st.metric("Total Profit", f"${dashboard_df['Profit'].sum():,.0f}")

    with k3:
        st.metric("Orders", f"{dashboard_df['Order ID'].nunique():,}")

    with k4:
        st.metric("Customers", f"{dashboard_df['Customer ID'].nunique():,}")

    if dashboard_df.empty:
        st.warning("ไม่พบข้อมูลตาม Filter ที่เลือก")
    else:

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # SALES BY CATEGORY
        # -------------------------------------------------

        with col1:

            category_sales = (
                dashboard_df.groupby("Category", as_index=False)
                .agg(
                    Sales=("Sales", "sum"),
                    Profit=("Profit", "sum"),
                    Orders=("Order ID", "nunique")
                )
                .sort_values("Sales", ascending=False)
            )

            fig_category = px.bar(
                category_sales,
                x="Category",
                y="Sales",
                title="Sales by Category",
                text_auto=".2s",
                custom_data=["Profit", "Orders"]
            )

            fig_category.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Sales: $%{y:,.2f}<br>"
                    "Profit: $%{customdata[0]:,.2f}<br>"
                    "Orders: %{customdata[1]:,}"
                    "<extra></extra>"
                )
            )

            fig_category.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=55, b=20)
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True,
                key="interactive_category_chart"
            )

            st.caption("💡 Hover เพื่อดู Sales, Profit และจำนวน Order")

        # -------------------------------------------------
        # PROFIT BY REGION
        # -------------------------------------------------

        with col2:

            region_profit = (
                dashboard_df.groupby("Region", as_index=False)
                .agg(
                    Profit=("Profit", "sum"),
                    Sales=("Sales", "sum"),
                    Orders=("Order ID", "nunique")
                )
                .sort_values("Profit", ascending=False)
            )

            fig_region = px.bar(
                region_profit,
                x="Region",
                y="Profit",
                title="Profit by Region",
                text_auto=".2s",
                custom_data=["Sales", "Orders"]
            )

            fig_region.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Profit: $%{y:,.2f}<br>"
                    "Sales: $%{customdata[0]:,.2f}<br>"
                    "Orders: %{customdata[1]:,}"
                    "<extra></extra>"
                )
            )

            fig_region.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=55, b=20)
            )

            st.plotly_chart(
                fig_region,
                use_container_width=True,
                key="interactive_region_chart"
            )

            st.caption("💡 Hover เพื่อดู Profit, Sales และจำนวน Order")

        # -------------------------------------------------
        # MONTHLY SALES TREND
        # -------------------------------------------------

        st.markdown("### 📈 Monthly Sales Trend")

        monthly_sales = (
            dashboard_df.dropna(subset=["Order Date"])
            .assign(
                Month=lambda x: x["Order Date"].dt.to_period("M").astype(str)
            )
            .groupby("Month", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum")
            )
        )

        fig_monthly = px.line(
            monthly_sales,
            x="Month",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend",
            custom_data=["Profit"]
        )

        fig_monthly.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sales: $%{y:,.2f}<br>"
                "Profit: $%{customdata[0]:,.2f}"
                "<extra></extra>"
            )
        )

        fig_monthly.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Sales",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=55, b=20)
        )

        st.plotly_chart(
            fig_monthly,
            use_container_width=True,
            key="interactive_monthly_chart"
        )

        st.caption("💡 Hover ตามเดือนเพื่อดู Sales และ Profit")

        # -------------------------------------------------
        # SUB-CATEGORY DETAIL
        # -------------------------------------------------

        st.markdown("### 🔍 Product Detail")

        subcategory_sales = (
            dashboard_df.groupby("Sub-Category", as_index=False)
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum")
            )
            .sort_values("Sales", ascending=False)
        )

        fig_subcategory = px.bar(
            subcategory_sales,
            x="Sub-Category",
            y="Sales",
            title="Sales by Sub-Category",
            custom_data=["Profit", "Quantity"]
        )

        fig_subcategory.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sales: $%{y:,.2f}<br>"
                "Profit: $%{customdata[0]:,.2f}<br>"
                "Quantity: %{customdata[1]:,}"
                "<extra></extra>"
            )
        )

        fig_subcategory.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=55, b=20)
        )

        st.plotly_chart(
            fig_subcategory,
            use_container_width=True,
            key="interactive_subcategory_chart"
        )


# =========================================================
# TAB 2 : CUSTOMER ANALYTICS
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">👥 Customer Analytics</div>',
        unsafe_allow_html=True
    )

    cluster_profiles = {
        0: {
            "name": "High Value Customer",
            "description": "มูลค่าการซื้อสูง ซื้อค่อนข้างบ่อย และสร้างกำไรสูง",
        },
        1: {
            "name": "Regular Customer",
            "description": "ซื้อค่อนข้างสม่ำเสมอ และมีมูลค่าการซื้อระดับปานกลาง",
        },
        2: {
            "name": "Inactive / At-Risk Customer",
            "description": "ไม่ได้ซื้อมานาน ซื้อไม่บ่อย และมีมูลค่าการซื้อค่อนข้างต่ำ",
        }
    }

    cluster_count = (
        customer_segments["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_count.columns = ["Cluster", "Customers"]

    cluster_count["Group"] = cluster_count["Cluster"].map(
        lambda x: cluster_profiles.get(
            int(x), {"name": "Unknown"}
        )["name"]
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_cluster = px.bar(
            cluster_count,
            x="Cluster",
            y="Customers",
            text="Customers",
            color="Group",
            hover_data={
                "Cluster": True,
                "Customers": True,
                "Group": True
            },
            title="Customers by Cluster"
        )

        fig_cluster.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )

        st.plotly_chart(
            fig_cluster,
            use_container_width=True,
            key="interactive_cluster_bar"
        )

    with col2:

        fig_cluster_pie = px.pie(
            cluster_count,
            names="Group",
            values="Customers",
            hole=0.55,
            title="Customer Cluster Distribution",
            hover_data=["Cluster"]
        )

        fig_cluster_pie.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=55, b=20)
        )

        st.plotly_chart(
            fig_cluster_pie,
            use_container_width=True,
            key="interactive_cluster_pie"
        )

    st.markdown("### 🎯 Explore a Customer Group")

    selected_cluster = st.selectbox(
        "เลือก Cluster เพื่อดูรายละเอียด",
        sorted(customer_segments["Cluster"].dropna().unique()),
        format_func=lambda x: (
            f"Cluster {int(x)} — "
            f"{cluster_profiles.get(int(x), {'name': 'Unknown'})['name']}"
        ),
        key="selected_cluster"
    )

    profile = cluster_profiles.get(
        int(selected_cluster),
        {"name": "Unknown", "description": "ไม่พบข้อมูล"}
    )

    cluster_customers = customer_segments[
        customer_segments["Cluster"] == selected_cluster
    ]

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric("Customers", f"{len(cluster_customers):,}")

    with p2:
        percentage = (
            len(cluster_customers) / len(customer_segments) * 100
            if len(customer_segments) > 0
            else 0
        )
        st.metric("Share", f"{percentage:.1f}%")

    with p3:
        st.metric("Cluster", int(selected_cluster))

    st.info(
        f"**{profile['name']}** — {profile['description']}"
    )

    st.markdown("### 🔎 Find Customer Cluster")

    customer_id = st.text_input(
        "Customer ID",
        placeholder="Example: AA-10315",
        key="customer_search"
    )

    if customer_id:

        customer = customer_segments[
            customer_segments["Customer ID"].astype(str).str.upper()
            == customer_id.strip().upper()
        ]

        if customer.empty:

            st.error("ไม่พบ Customer ID นี้")

        else:

            cluster = int(customer.iloc[0]["Cluster"])

            profile = cluster_profiles.get(
                cluster,
                {"name": "Unknown", "description": "ไม่พบข้อมูล"}
            )

            st.success(
                f"Customer **{customer_id.strip().upper()}** "
                f"อยู่ใน **Cluster {cluster} — {profile['name']}**"
            )

            st.caption(profile["description"])

            st.dataframe(
                customer,
                use_container_width=True,
                hide_index=True
            )


# =========================================================

# TAB 3 : PROFIT PREDICTION

# =========================================================



with tab3:



    st.markdown(

        '<div class="section-title">💰 Profit Prediction</div>',

        unsafe_allow_html=True

    )



    st.write(

        """

        ทดลองทำนายกำไรของ Order ใหม่ด้วย

        **XGBoost Regression Model**

        """

    )





    col1, col2 = st.columns(2)





    with col1:



        sales = st.number_input(

            "Sales",

            min_value=0.0,

            value=500.0,

            step=10.0

        )





        quantity = st.number_input(

            "Quantity",

            min_value=1,

            value=2,

            step=1

        )





        discount_percent = st.number_input(

            "Discount (%)",

            min_value=0.0,

            max_value=100.0,

            value=10.0,

            step=1.0

        )





        category = st.selectbox(

            "Category",

            [

                "Furniture",

                "Office Supplies",

                "Technology"

            ]

        )





        sub_category = st.selectbox(

            "Sub-Category",

            sorted(

                df["Sub-Category"]

                .dropna()

                .unique()

                .tolist()

            )

        )





    with col2:



        region = st.selectbox(

            "Region",

            sorted(

                df["Region"]

                .dropna()

                .unique()

                .tolist()

            )

        )





        ship_mode = st.selectbox(

            "Ship Mode",

            sorted(

                df["Ship Mode"]

                .dropna()

                .unique()

                .tolist()

            )

        )





        segment = st.selectbox(

            "Segment",

            sorted(

                df["Segment"]

                .dropna()

                .unique()

                .tolist()

            )

        )





    # -----------------------------------------------------

    # PREDICT BUTTON

    # -----------------------------------------------------



    if st.button(

        "🔮 Predict Profit",

        use_container_width=True

    ):



        try:



            new_order = pd.DataFrame({



                "Sales": [sales],



                "Quantity": [quantity],



                "Discount": [

                    discount_percent / 100

                ],



                "Category": [category],



                "Sub-Category": [sub_category],



                "Region": [region],



                "Ship Mode": [ship_mode],



                "Segment": [segment]



            })





            prediction = profit_model.predict(

                new_order

            )





            predicted_profit = float(

                prediction[0]

            )





            st.success(

                f"💰 Predicted Profit: "

                f"**${predicted_profit:,.2f}**"

            )





            st.caption(

                "หมายเหตุ: ค่านี้เป็นผลการทำนายจาก "

                "Machine Learning Model ไม่ใช่กำไรจริงที่รับประกัน"

            )





        except Exception as e:



            st.error(

                f"Prediction Error: {e}"

            )





# =========================================================

# TAB 4 : AI ASSISTANT

# =========================================================



with tab4:



    st.markdown(

        '<div class="section-title">💬 AI Business Analyst Assistant</div>',

        unsafe_allow_html=True

    )





    st.write(

        """

        ถามคำถามเกี่ยวกับ Customer Segmentation,

        Profit Prediction และ Business Analysis ได้เลย

        """

    )





    # -----------------------------------------------------

    # API KEY CHECK

    # -----------------------------------------------------



    if not st.session_state.gemini_api_key:



        st.warning(

            "🔑 กรุณากรอก Gemini API Key "

            "ที่ Sidebar ก่อนใช้งาน AI Assistant"

        )





    # -----------------------------------------------------

    # EXAMPLE QUESTIONS

    # -----------------------------------------------------



    st.markdown("### 💡 Example Questions")





    example_questions = [



        "ลูกค้า AA-10315 อยู่กลุ่มไหน",



        "ลูกค้า AA-10315 อยู่กลุ่มไหน และถ้าซื้อ Copiers ราคา 500 จำนวน 2 ชิ้น ลด 10% ใน West แบบ Second Class สำหรับ Consumer จะคาดว่าจะได้กำไรเท่าไหร่",



        "ถ้าขาย Copiers ราคา 500 จำนวน 2 ชิ้น ลด 10% ใน West สำหรับ Consumer จะได้กำไรประมาณเท่าไหร่"



    ]





    example_cols = st.columns(3)





    for i, question in enumerate(

        example_questions

    ):



        with example_cols[i]:



            if st.button(

                question,

                key=f"example_{i}",

                use_container_width=True

            ):



                st.session_state["selected_question"] = question





    # -----------------------------------------------------

    # SELECTED QUESTION

    # -----------------------------------------------------



    selected_question = st.session_state.get(

        "selected_question",

        ""

    )





    # -----------------------------------------------------

    # CHAT INPUT

    # -----------------------------------------------------



    user_question = st.chat_input(

        "พิมพ์คำถามของคุณ..."

    )





    if selected_question:



        user_question = selected_question



        st.session_state["selected_question"] = ""





    # -----------------------------------------------------

    # DISPLAY OLD MESSAGES

    # -----------------------------------------------------



    for message in st.session_state.messages:



        with st.chat_message(

            message["role"]

        ):



            st.markdown(

                message["content"]

            )





    # -----------------------------------------------------

    # ASK AI

    # -----------------------------------------------------



    if user_question:



        # Add user message



        st.session_state.messages.append(



            {

                "role": "user",

                "content": user_question

            }



        )





        # Display user message



        with st.chat_message("user"):



            st.markdown(

                user_question

            )





        # Check API Key



        if not st.session_state.gemini_api_key:



            st.error(

                "กรุณากรอก Gemini API Key "

                "ใน Sidebar ก่อนใช้งาน AI"

            )



        else:



            # -------------------------------------------------

            # CALL GEMINI

            # -------------------------------------------------



            with st.chat_message(

                "assistant"

            ):



                with st.spinner(

                    "🤖 AI กำลังวิเคราะห์..."

                ):



                    try:



                        answer = ask_assistant(



                            user_question,



                            st.session_state.gemini_api_key,



                            st.session_state.chat_history



                        )





                        st.markdown(

                            answer

                        )





                        # Save assistant message



                        st.session_state.messages.append(



                            {

                                "role": "assistant",

                                "content": answer

                            }



                        )





                    except Exception as e:



                        st.error(

                            f"เกิดข้อผิดพลาด: {e}"

                        )





# =========================================================

# FOOTER

# =========================================================



st.markdown(

    """

    <div class="footer">

        AI Business Analyst • K-Means Customer Segmentation

        • XGBoost Profit Prediction • Gemini AI

    </div>

    """,

    unsafe_allow_html=True

)
