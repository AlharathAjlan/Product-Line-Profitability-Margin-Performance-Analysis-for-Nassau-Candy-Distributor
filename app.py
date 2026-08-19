import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy Profitability Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Data loading + cleaning (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # remove zero/negative sales or units (per Step 1 validation)
    df = df[(df["Sales"] > 0) & (df["Units"] > 0)].copy()

    # standardize text labels
    df["Division"] = df["Division"].astype(str).str.strip().str.title()
    df["Product Name"] = df["Product Name"].astype(str).str.strip()

    # parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    return df


@st.cache_data
def build_product_master(df: pd.DataFrame) -> pd.DataFrame:
    pm = df.groupby("Product Name").agg(
        Total_Sales=("Sales", "sum"),
        Total_Units=("Units", "sum"),
        Total_Cost=("Cost", "sum"),
        Total_Gross_Profit=("Gross Profit", "sum"),
    ).reset_index()

    pm["Gross_Margin_%"] = (pm["Total_Gross_Profit"] / pm["Total_Sales"] * 100).round(2)
    pm["Profit_per_Unit"] = (pm["Total_Gross_Profit"] / pm["Total_Units"]).round(2)
    pm["Profit_Contribution_%"] = (pm["Total_Gross_Profit"] / pm["Total_Gross_Profit"].sum() * 100).round(2)
    pm["Revenue_Contribution_%"] = (pm["Total_Sales"] / pm["Total_Sales"].sum() * 100).round(2)
    pm["Cost_Ratio_%"] = (pm["Total_Cost"] / pm["Total_Sales"] * 100).round(2)

    # attach Division (one product = one division in this dataset)
    div_map = df.drop_duplicates("Product Name").set_index("Product Name")["Division"]
    pm["Division"] = pm["Product Name"].map(div_map)

    # classification (median-based, matches the research paper)
    sales_median = pm["Total_Sales"].median()
    margin_median = pm["Gross_Margin_%"].median()

    def classify(row):
        high_sales = row["Total_Sales"] >= sales_median
        high_margin = row["Gross_Margin_%"] >= margin_median
        if high_sales and high_margin:
            return "High-Profit / High-Margin"
        elif high_sales and not high_margin:
            return "High-Sales / Low-Margin"
        elif not high_sales and not high_margin:
            return "Low-Sales / Low-Margin"
        else:
            return "Low-Sales / High-Margin"

    pm["Category"] = pm.apply(classify, axis=1)
    return pm


DATA_PATH = "Nassau Candy Distributor.csv"  # <-- point this at your actual CSV filename
df = load_data(DATA_PATH)
product_master = build_product_master(df)

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("Nassau Candy Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "Product Profitability Overview",
        "Division Performance",
        "Cost vs Margin Diagnostics",
        "Profit Concentration (Pareto)",
        "Cost Anomaly Detection",
    ],
)

# ---------------------------------------------------------------------
# Global filters
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

min_date = df["Order Date"].min()
max_date = df["Order Date"].max()
date_range = st.sidebar.date_input(
    "Order Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

division_options = st.sidebar.multiselect(
    "Division", options=sorted(df["Division"].unique()), default=list(df["Division"].unique())
)

margin_threshold = st.sidebar.slider(
    "Minimum Gross Margin % (product-level filter)", 0, 100, 0
)

product_search = st.sidebar.text_input("Product search (contains)", "")

# apply filters to order-level data
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered_df = df[(df["Order Date"] >= start_date) & (df["Order Date"] <= end_date)]
else:
    filtered_df = df.copy()

filtered_df = filtered_df[filtered_df["Division"].isin(division_options)]

# apply filters to product-level table
filtered_pm = product_master[product_master["Division"].isin(division_options)]
filtered_pm = filtered_pm[filtered_pm["Gross_Margin_%"] >= margin_threshold]
if product_search:
    filtered_pm = filtered_pm[filtered_pm["Product Name"].str.contains(product_search, case=False, na=False)]

st.sidebar.markdown(f"**{len(filtered_pm)}** of {len(product_master)} products match filters")

if len(filtered_pm) == 0:
    st.warning("No products match the selected filters. Adjust filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------
# MODULE 1: Product Profitability Overview
# ---------------------------------------------------------------------
if page == "Product Profitability Overview":
    st.title("Product Profitability Overview")

    overall_margin = filtered_df["Gross Profit"].sum() / filtered_df["Sales"].sum() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Gross Margin", f"{overall_margin:.2f}%")
    col2.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
    col3.metric("Total Gross Profit", f"${filtered_df['Gross Profit'].sum():,.0f}")

    st.subheader("Product Margin Leaderboard")
    leaderboard = filtered_pm.sort_values("Gross_Margin_%", ascending=False)
    fig1 = px.bar(
        leaderboard, x="Product Name", y="Gross_Margin_%", color="Gross_Margin_%",
        color_continuous_scale="RdYlGn", text="Gross_Margin_%",
    )
    fig1.update_traces(texttemplate="%{text}%", textposition="outside")
    fig1.add_hline(y=overall_margin, line_dash="dash", line_color="gray", annotation_text="Company average")
    fig1.update_xaxes(tickangle=-40)
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Profit Contribution by Product")
    fig2 = px.pie(
        leaderboard, names="Product Name", values="Total_Gross_Profit", hole=0.4,
    )
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Product Classification")
    cat_order = ["High-Profit / High-Margin", "High-Sales / Low-Margin", "Low-Sales / High-Margin", "Low-Sales / Low-Margin"]
    cat_counts = filtered_pm["Category"].value_counts().reindex(cat_order).fillna(0).reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig3 = px.bar(cat_counts, x="Category", y="Count", color="Category", text="Count")
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Full Product Table")
    st.dataframe(
        leaderboard[["Product Name", "Division", "Total_Sales", "Total_Gross_Profit",
                      "Gross_Margin_%", "Profit_per_Unit", "Profit_Contribution_%",
                      "Revenue_Contribution_%", "Category"]],
        width="stretch",
    )

# ---------------------------------------------------------------------
# MODULE 2: Division Performance Dashboard
# ---------------------------------------------------------------------
elif page == "Division Performance":
    st.title("Division Performance Dashboard")

    overall_margin = filtered_df["Gross Profit"].sum() / filtered_df["Sales"].sum() * 100

    division_agg = filtered_df.groupby("Division").agg(
        Total_Sales=("Sales", "sum"),
        Total_Cost=("Cost", "sum"),
        Total_Gross_Profit=("Gross Profit", "sum"),
    ).reset_index()
    division_agg["Average_Margin_%"] = (
        division_agg["Total_Gross_Profit"] / division_agg["Total_Sales"] * 100
    ).round(2)
    division_agg["Margin_vs_Overall"] = (division_agg["Average_Margin_%"] - overall_margin).round(2)
    division_agg["Profit_Contribution_%"] = (
        division_agg["Total_Gross_Profit"] / division_agg["Total_Gross_Profit"].sum() * 100
    ).round(2)
    division_agg["Revenue_Contribution_%"] = (
        division_agg["Total_Sales"] / division_agg["Total_Sales"].sum() * 100
    ).round(2)

    def efficiency_flag(row):
        if row["Average_Margin_%"] >= overall_margin:
            return "Strong Financial Efficiency"
        elif row["Average_Margin_%"] >= overall_margin * 0.5:
            return "Below Average — Monitor"
        else:
            return "Structural Margin Issue"

    division_agg["Efficiency_Status"] = division_agg.apply(efficiency_flag, axis=1)
    division_agg = division_agg.sort_values("Total_Sales", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Divisions Shown", len(division_agg))
    col2.metric("Highest Margin Division", division_agg.loc[division_agg["Average_Margin_%"].idxmax(), "Division"])
    col3.metric("Company Average Margin", f"{overall_margin:.2f}%")

    st.subheader("Revenue vs. Profit Comparison")
    rev_profit_long = division_agg.melt(
        id_vars="Division", value_vars=["Revenue_Contribution_%", "Profit_Contribution_%"],
        var_name="Metric", value_name="Contribution %"
    )
    fig1 = px.bar(
        rev_profit_long, x="Division", y="Contribution %", color="Metric", barmode="group",
        text="Contribution %",
    )
    fig1.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig1, width="stretch")
    st.caption(
        "When a division's Profit Contribution bar sits noticeably above or below its Revenue "
        "Contribution bar, that division is more or less profit-efficient than its sales volume alone would suggest."
    )

    st.subheader("Margin Distribution by Division")
    fig2 = px.bar(
        division_agg, x="Division", y="Average_Margin_%", color="Efficiency_Status",
        text="Average_Margin_%",
        color_discrete_map={
            "Strong Financial Efficiency": "#2E86AB",
            "Below Average — Monitor": "#F4A261",
            "Structural Margin Issue": "#E63946",
        },
    )
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.add_hline(y=overall_margin, line_dash="dash", line_color="gray", annotation_text="Company average")
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Division-Level Table")
    st.dataframe(
        division_agg[["Division", "Total_Sales", "Total_Cost", "Total_Gross_Profit",
                       "Average_Margin_%", "Margin_vs_Overall", "Revenue_Contribution_%",
                       "Profit_Contribution_%", "Efficiency_Status"]],
        width="stretch",
    )

    st.subheader("Isolate a Product's Effect on Its Division")
    isolate_division = st.selectbox("Division", division_agg["Division"])
    isolate_product = st.selectbox(
        "Exclude this product",
        filtered_pm[filtered_pm["Division"] == isolate_division]["Product Name"],
    )
    remaining = filtered_df[
        (filtered_df["Division"] == isolate_division) & (filtered_df["Product Name"] != isolate_product)
    ]
    if len(remaining) > 0 and remaining["Sales"].sum() > 0:
        adj_margin = remaining["Gross Profit"].sum() / remaining["Sales"].sum() * 100
        current_margin = division_agg.loc[division_agg["Division"] == isolate_division, "Average_Margin_%"].values[0]
        st.metric(
            f"{isolate_division} margin excluding {isolate_product}",
            f"{adj_margin:.2f}%",
            delta=f"{adj_margin - current_margin:+.2f} pts vs. current division margin",
        )
    else:
        st.info("No remaining orders in this division after excluding that product.")