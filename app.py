import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

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


# ---------------------------------------------------------------------
# MODULE 3: Cost vs Margin Diagnostics
# ---------------------------------------------------------------------
elif page == "Cost vs Margin Diagnostics":
    st.title("Cost vs Margin Diagnostics")

    st.subheader("Cost vs Sales Scatter (color = Gross Margin %)")
    fig1 = px.scatter(
        filtered_pm, x="Total_Sales", y="Total_Cost", color="Gross_Margin_%",
        size="Total_Units", hover_name="Product Name",
        color_continuous_scale="RdYlGn", size_max=40,
    )
    max_val = max(filtered_pm["Total_Sales"].max(), filtered_pm["Total_Cost"].max())
    fig1.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color="gray", dash="dash"), name="Cost = Sales (0% margin)", showlegend=True,
    ))
    fig1.update_layout(xaxis_title="Total Sales", yaxis_title="Total Cost")
    st.plotly_chart(fig1, width="stretch")
    st.caption(
        "Points near the diagonal have thin margins regardless of size. "
        "Point size reflects total units sold."
    )

    st.markdown("---")
    st.subheader("Cost-Heavy, Margin-Poor Products")
    cost_ratio_threshold = st.slider("Cost Ratio % threshold (flag above this)", 0, 100, 60)
    cost_heavy = filtered_pm[filtered_pm["Cost_Ratio_%"] >= cost_ratio_threshold].sort_values(
        "Cost_Ratio_%", ascending=False
    )
    if len(cost_heavy) > 0:
        st.dataframe(
            cost_heavy[["Product Name", "Division", "Total_Sales", "Total_Cost",
                         "Cost_Ratio_%", "Gross_Margin_%"]],
            width="stretch",
        )
    else:
        st.info(f"No products exceed a {cost_ratio_threshold}% cost ratio.")

    st.markdown("---")
    st.subheader("Pricing Inefficiency Check")
    st.caption(
        "Compares Cost per Unit against Sales per Unit. A product priced in line with peers "
        "but with disproportionately high cost per unit signals a cost/sourcing problem rather "
        "than a pricing problem."
    )
    pricing_check = filtered_pm.copy()
    pricing_check["Sales_per_Unit"] = (pricing_check["Total_Sales"] / pricing_check["Total_Units"]).round(2)
    pricing_check["Cost_per_Unit"] = (pricing_check["Total_Cost"] / pricing_check["Total_Units"]).round(2)
    fig2 = px.bar(
        pricing_check.sort_values("Cost_Ratio_%", ascending=False),
        x="Product Name", y=["Cost_per_Unit", "Sales_per_Unit"], barmode="group",
    )
    fig2.update_xaxes(tickangle=-40)
    st.plotly_chart(fig2, width="stretch")

    st.markdown("---")
    st.subheader("Action Flags")

    def diagnose(row):
        if row["Gross_Margin_%"] < 20 and row["Total_Sales"] >= filtered_pm["Total_Sales"].median():
            return "URGENT: Repricing or Cost Renegotiation"
        elif row["Gross_Margin_%"] < 20:
            return "Discontinuation Review"
        elif row["Gross_Margin_%"] < 50:
            return "Monitor — Below-Average Margin"
        else:
            return "Healthy"

    flagged = filtered_pm.copy()
    flagged["Action_Flag"] = flagged.apply(diagnose, axis=1)
    flag_order = ["URGENT: Repricing or Cost Renegotiation", "Discontinuation Review",
                  "Monitor — Below-Average Margin", "Healthy"]
    flag_colors = {
        "URGENT: Repricing or Cost Renegotiation": "#E63946",
        "Discontinuation Review": "#F4A261",
        "Monitor — Below-Average Margin": "#F9C74F",
        "Healthy": "#2E86AB",
    }
    fig3 = px.bar(
        flagged["Action_Flag"].value_counts().reindex(flag_order).fillna(0).reset_index(),
        x="Action_Flag", y="count", color="Action_Flag", color_discrete_map=flag_colors, text="count",
    )
    st.plotly_chart(fig3, width="stretch")
    st.dataframe(
        flagged[["Product Name", "Division", "Total_Sales", "Gross_Margin_%", "Cost_Ratio_%", "Action_Flag"]]
        .sort_values("Gross_Margin_%"),
        width="stretch",
    )

# ---------------------------------------------------------------------
# MODULE 4: Profit Concentration (Pareto) Analysis
# ---------------------------------------------------------------------
elif page == "Profit Concentration (Pareto)":
    st.title("Profit Concentration (Pareto) Analysis")

    def build_pareto(df_, group_col, value_col):
        agg = df_.groupby(group_col)[value_col].sum().sort_values(ascending=False).reset_index()
        agg["Cumulative_%"] = (agg[value_col].cumsum() / agg[value_col].sum() * 100).round(2)
        agg["Rank_%"] = ((agg.index + 1) / len(agg) * 100).round(1)
        return agg

    pareto_metric = st.radio("Concentration metric", ["Revenue (Sales)", "Profit (Gross Profit)"], horizontal=True)
    value_col = "Sales" if pareto_metric.startswith("Revenue") else "Gross Profit"

    st.subheader(f"Product Concentration — {pareto_metric}")
    pareto_products = build_pareto(filtered_df, "Product Name", value_col)
    items_for_80 = int((pareto_products["Cumulative_%"] < 80).sum() + 1)
    pct_items_for_80 = round(items_for_80 / len(pareto_products) * 100, 1)

    col1, col2 = st.columns(2)
    col1.metric("Products needed for 80%", f"{items_for_80} of {len(pareto_products)}")
    col2.metric("% of catalog", f"{pct_items_for_80}%")

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=pareto_products["Product Name"], y=pareto_products[value_col], name=pareto_metric))
    fig1.add_trace(go.Scatter(
        x=pareto_products["Product Name"], y=pareto_products["Cumulative_%"],
        name="Cumulative %", yaxis="y2", line=dict(color="red"),
    ))
    fig1.add_hline(y=80, yref="y2", line_dash="dash", line_color="gray")
    fig1.update_layout(
        yaxis=dict(title=pareto_metric),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        xaxis=dict(tickangle=-40),
    )
    st.plotly_chart(fig1, width="stretch")
    st.dataframe(pareto_products, width="stretch")

    st.markdown("---")
    st.subheader("Geographic Concentration — % of States Contributing 80% of Revenue")
    pareto_states = build_pareto(filtered_df, "State/Province", "Sales")
    states_for_80 = int((pareto_states["Cumulative_%"] < 80).sum() + 1)
    col3, col4 = st.columns(2)
    col3.metric("States needed for 80% of revenue", f"{states_for_80} of {len(pareto_states)}")
    col4.metric("% of states", f"{round(states_for_80/len(pareto_states)*100, 1)}%")
    st.dataframe(pareto_states.head(20), width="stretch")

    st.markdown("---")
    st.subheader("Regional Distribution & Over-Dependency Check")
    region_stats = filtered_df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
    region_stats["Revenue_Share_%"] = (region_stats["Sales"] / region_stats["Sales"].sum() * 100).round(2)
    fig2 = px.bar(region_stats, x="Region", y="Revenue_Share_%", text="Revenue_Share_%", color="Region")
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        "No region falling far below the others indicates diversified geographic exposure; "
        "a dominant region would signal over-dependency risk."
    )

# ---------------------------------------------------------------------
# MODULE 5: Cost Anomaly Detection (ML)
# ---------------------------------------------------------------------
elif page == "Cost Anomaly Detection":
    st.title("Cost Anomaly Detection (Machine Learning)")
    st.caption(
        "A regression model learns typical cost patterns from Sales, Units, Division, Region, "
        "Ship Mode, and order timing — then flags individual orders whose actual cost deviates "
        "sharply from what the model expects."
    )

    if len(filtered_df) < 50:
        st.warning("Not enough filtered orders to train a reliable model. Broaden your filters.")
    else:
        model_df = filtered_df.copy()
        model_df["OrderMonth"] = model_df["Order Date"].dt.month
        model_df["OrderQuarter"] = model_df["Order Date"].dt.quarter

        X = pd.get_dummies(
            model_df[["Sales", "Units", "Division", "Region", "Ship Mode", "OrderMonth", "OrderQuarter"]],
            columns=["Division", "Region", "Ship Mode", "OrderMonth", "OrderQuarter"],
            drop_first=True,
        )
        y_cost = model_df["Cost"]

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y_cost, model_df.index, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        col1, col2 = st.columns(2)
        col1.metric("Model R²", f"{r2_score(y_test, preds):.4f}")
        col2.metric("Model MAE", f"${mean_absolute_error(y_test, preds):.2f}")

        results = model_df.loc[idx_test, ["Product Name", "Division", "Sales", "Cost"]].copy()
        results["Predicted_Cost"] = preds
        results["Residual"] = results["Cost"] - results["Predicted_Cost"]
        results["Abs_Residual"] = results["Residual"].abs()

        std_multiplier = st.slider("Anomaly sensitivity (standard deviations)", 1.0, 3.0, 2.0, 0.5)
        threshold = std_multiplier * results["Residual"].std()
        results["Anomaly"] = results["Abs_Residual"] > threshold

        st.metric(
            "Flagged Anomalies",
            f"{int(results['Anomaly'].sum())} of {len(results)} test orders",
            delta=f"±${threshold:.2f} threshold",
        )

        normal = results[~results["Anomaly"]]
        anomalies = results[results["Anomaly"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=normal["Cost"], y=normal["Predicted_Cost"], mode="markers", name="Normal",
            marker=dict(color="#2E86AB", opacity=0.5, size=6),
        ))
        fig.add_trace(go.Scatter(
            x=anomalies["Cost"], y=anomalies["Predicted_Cost"], mode="markers", name="Anomaly",
            marker=dict(color="#E63946", size=11, line=dict(color="black", width=1)),
            text=anomalies["Product Name"], hovertemplate="%{text}<br>Actual: %{x}<br>Predicted: %{y}",
        ))
        max_val = max(results["Cost"].max(), results["Predicted_Cost"].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color="gray", dash="dash"), name="Perfect prediction",
        ))
        fig.update_layout(xaxis_title="Actual Cost", yaxis_title="Predicted Cost")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Which products are flagged most often?")
        if len(anomalies) > 0:
            anomaly_counts = anomalies["Product Name"].value_counts().reset_index()
            anomaly_counts.columns = ["Product Name", "Times Flagged"]
            fig2 = px.bar(anomaly_counts, x="Product Name", y="Times Flagged", color="Times Flagged",
                          color_continuous_scale="Reds")
            fig2.update_xaxes(tickangle=-40)
            st.plotly_chart(fig2, width="stretch")

            st.subheader("Flagged Orders (sorted by severity)")
            st.dataframe(
                anomalies.sort_values("Abs_Residual", ascending=False)[
                    ["Product Name", "Division", "Sales", "Cost", "Predicted_Cost", "Residual"]
                ],
                width="stretch",
            )
        else:
            st.info("No anomalies flagged at this sensitivity level.")
