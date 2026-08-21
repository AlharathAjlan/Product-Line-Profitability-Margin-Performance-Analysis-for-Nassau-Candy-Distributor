# Nassau Candy Distributor — Product Line Profitability & Margin Analytics

Interactive Streamlit dashboard analyzing product-line profitability, division performance, cost structure, and revenue concentration for Nassau Candy Distributor. Includes a machine learning module for automated cost anomaly detection. Built as part of a Machine Learning internship project (Unified Mentor).

## Overview

Sales volume alone is misleading for a distributor — some products sell in high volume but generate low profit, while others are highly efficient but too small to matter. This project identifies **which products truly drive profit**, **which divisions underperform financially**, and **where pricing, sourcing, or product rationalization is needed**.

**Type:** Data Analytics (diagnostic segmentation & concentration analysis), extended with a supervised Machine Learning module for cost anomaly detection.

## Key Findings

- Overall gross margin: **65.91%**.
- Just **5 of 15 products** (33.3%) — the entire Wonka Bar line — generate over **92% of revenue** and **95% of profit**.
- **Kazookles** is the one clear, quantified problem product: lowest margin (7.69%), the sole driver of the Other division's underperformance, and the dominant source of flagged cost anomalies in the ML model (~50% of all flagged orders).
- Excluding Kazookles, the Other division's margin rises from 44.84% to **50.14%** — the issue is product-specific, not division-wide.
- Geographic revenue is comparatively diversified: 27.1% of states are needed for 80% of revenue, vs. 33.3% of products for the same share — the real over-dependency risk is in the product portfolio, not customer geography.
- Order timing (season, quarter, day of week) has no measurable effect on margin — confirmed independently by both aggregation and a regression model.

Full methodology and findings are documented in the accompanying research paper.

## Dashboard Modules

| Page | Contents |
|---|---|
| Product Profitability Overview | Margin leaderboard, profit contribution chart, 4-category product classification, full product table |
| Division Performance | Revenue vs. profit comparison, margin distribution with efficiency status, interactive "isolate a product's effect on its division" tool |
| Cost vs Margin Diagnostics | Cost-vs-sales scatter, adjustable cost-ratio flagging, pricing-vs-cost comparison, Action Flags (URGENT/Monitor/Healthy) |
| Profit Concentration (Pareto) | Revenue/profit Pareto charts with cumulative %, state-level and regional concentration |
| Cost Anomaly Detection | Live-trained regression model flagging orders with abnormal cost structure, adjustable sensitivity |

All pages respond to global filters: Order Date range, Division, minimum margin threshold, and product search.

## KPIs Tracked

- **Gross Margin (%)** — Gross Profit ÷ Sales
- **Profit per Unit** — Gross Profit ÷ Units
- **Revenue Contribution** — product sales ÷ total sales
- **Profit Contribution** — product profit ÷ total profit
- **Margin Volatility** — variability of margin over time (tested via ML; found to be negligible)

## Tech Stack

- [Streamlit](https://streamlit.io/) — dashboard framework
- [Pandas](https://pandas.pydata.org/) — data processing
- [Plotly](https://plotly.com/python/) — interactive charts
- [scikit-learn](https://scikit-learn.org/) — regression modeling for cost anomaly detection

## Setup & Local Run

```bash
# clone the repo
git clone https://github.com/AlharathAjlan/Product-Line-Profitability-Margin-Performance-Analysis-for-Nassau-Candy-Distributor.git
cd Product-Line-Profitability-Margin-Performance-Analysis-for-Nassau-Candy-Distributor

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
python -m streamlit run app.py
```

The app expects a CSV file in the project folder with the following columns:

```
Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID,
Country/Region, City, State/Province, Postal Code, Division, Region,
Product ID, Product Name, Sales, Units, Gross Profit, Cost
```

Update the `DATA_PATH` variable near the top of `app.py` to match your filename.

## Data Notes
**Known data quality note:** `Ship Date` values in the source dataset fall systematically 6 months to 4+ years after `Order Date` across the entire dataset. This was investigated and confirmed to be a structural characteristic of the data (not a small subset of errors), so shipping-delay metrics are intentionally excluded from this analysis.

## Machine Learning Module

The Cost Anomaly Detection page trains a linear regression model live on the currently filtered data to predict `Cost` from `Sales`, `Units`, `Division`, `Region`, `Ship Mode`, and order timing. Orders whose actual cost deviates from the model's prediction beyond an adjustable threshold (in standard deviations) are flagged for review. This operationalizes the manual cost diagnostics into a capability that can flag future orders automatically rather than requiring periodic manual review.

A staged model-development process (baseline sanity check → drop cost → add seasonality → predict cost directly) is documented in the research paper, validating that product/division — not time or geography — is the primary driver of both profit and cost anomalies.

## Project Structure

```
nassau_dashboard/
├── app.py              # Streamlit dashboard (5 modules)
├── requirements.txt
├── README.md
├── .gitignore
└── Nassau Candy Distributor.csv
```

## Deliverables

- Report paper (EDA, profitability analysis, ML validation, insights, recommendations)
- This Streamlit dashboard (live analytics)
- Executive summary for stakeholders

## Status

All Dashboard Modules and User Capabilities (date range selector, division filter, margin threshold slider, product search) from the project requirements are implemented, plus an additional ML-powered cost anomaly detection module.
