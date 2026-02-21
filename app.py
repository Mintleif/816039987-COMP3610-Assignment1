"""
NYC Taxi Trip Dashboard
COMP 3610 - Assignment 1

Run with:
streamlit run app.py
"""

import duckdb  # SQL engine used to query parquet files efficiently
import streamlit as st  # Web app framework for building dashboards
import pandas as pd  # Data manipulation library
import plotly.express as px  # Interactive plotting library


# Configure Streamlit page settings (must be called before anything else)
st.set_page_config(
    page_title="NYC Taxi Dashboard",  # Browser tab title
    page_icon="🚕",  # Browser tab icon
    layout="wide",  # Use full screen width
    initial_sidebar_state="expanded"  # Sidebar open by default
)


# Custom CSS styling for headers
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E3A5F;
}
.sub-header {
    font-size: 1.1rem;
    color: #555;
}
</style>
""",
    unsafe_allow_html=True,  # Allow raw HTML styling
)


# URLs for NYC taxi trip data and taxi zone lookup table
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


# Mapping payment type codes to readable labels
PAYMENT_MAP = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided Trip",
}

# Reverse mapping (label → code) for filtering
LABEL_TO_CODE = {v: k for k, v in PAYMENT_MAP.items()}


@st.cache_data  # Cache zone data to avoid reloading
def load_zones() -> pd.DataFrame:
    zones_df = pd.read_csv(ZONE_URL)  # Load taxi zone lookup table
    return zones_df


@st.cache_data  # Cache result to avoid repeated queries
def get_min_max_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    # Query parquet file directly using DuckDB without loading full dataset
    res = duckdb.query(f"""
        SELECT
            MIN(tpep_pickup_datetime) AS min_dt,
            MAX(tpep_pickup_datetime) AS max_dt
        FROM read_parquet('{TRIP_URL}')
    """).to_df()

    # Convert SQL result to pandas timestamps
    min_dt = pd.to_datetime(res.loc[0, "min_dt"], errors="coerce")
    max_dt = pd.to_datetime(res.loc[0, "max_dt"], errors="coerce")
    return min_dt, max_dt


@st.cache_data  # Cache filtered dataset for performance
def get_filtered_df(
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    hour_min: int,
    hour_max: int,
    payment_labels: list[str],
    selected_zone_names: list[str],
) -> pd.DataFrame:

    zones_df = load_zones()  # Load zone lookup table

    # Convert selected payment labels to numeric codes
    payment_codes = [LABEL_TO_CODE[p] for p in payment_labels if p in LABEL_TO_CODE]

    # If no payment types selected, return empty DataFrame
    if not payment_codes:
        return pd.DataFrame()

    zone_filter_sql = ""  # Default: no zone filtering

    # If zones are selected, convert zone names to LocationIDs
    if selected_zone_names:
        loc_ids = zones_df[zones_df["Zone"].isin(selected_zone_names)]["LocationID"] \
            .dropna().astype(int).unique().tolist()

        # Build SQL condition for selected zones
        if loc_ids:
            zone_filter_sql = f"AND PULocationID IN ({','.join(map(str, loc_ids))})"

    # Main SQL query to filter dataset efficiently in DuckDB
    sql = f"""
        SELECT
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            PULocationID,
            DOLocationID,
            trip_distance,
            fare_amount,
            tip_amount,
            total_amount,
            payment_type
        FROM read_parquet('{TRIP_URL}')
        WHERE
            tpep_pickup_datetime >= TIMESTAMP '{start_datetime}'
            AND tpep_pickup_datetime < TIMESTAMP '{end_datetime}'
            AND EXTRACT('hour' FROM tpep_pickup_datetime) BETWEEN {hour_min} AND {hour_max}
            AND payment_type IN ({','.join(map(str, payment_codes))})
            AND trip_distance > 0
            AND fare_amount > 0
            AND fare_amount <= 500
            AND tpep_dropoff_datetime > tpep_pickup_datetime
            {zone_filter_sql}
    """

    df = duckdb.query(sql).to_df()  # Execute SQL query

    # Convert datetime columns properly
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    # Create trip duration in minutes
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Create time-based features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    # Merge pickup zone names into main dataset
    df = df.merge(
        zones_df[["LocationID", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    ).rename(columns={"Zone": "pickup_zone"})

    # Convert payment codes to readable labels
    df["payment_label"] = df["payment_type"].map(PAYMENT_MAP).fillna("Other/Missing")

    # Set ordered weekday categories for proper heatmap ordering
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["pickup_day_of_week"] = pd.Categorical(
        df["pickup_day_of_week"],
        categories=day_order,
        ordered=True
    )

    return df


# Main dashboard header
st.markdown('<div class="main-header">NYC Taxi Trip Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Yellow Taxi Trips - January 2024</div>', unsafe_allow_html=True)

# Description text
st.markdown(
    """
    This dashboard provides insights into trip patterns, fare dynamics, and payment methods for January 2024.
    Explore key metrics and trends in NYC taxi trips. Use the filters in the sidebar to customize the data view.
    """,
    unsafe_allow_html=True,
)

st.divider()  # Horizontal line separator


# Sidebar filters
st.sidebar.header("Filters")

zones = load_zones()  # Load zones
min_dt, max_dt = get_min_max_dates()  # Get full dataset date range

min_date = min_dt.date()
max_date = max_dt.date()

# Date range selector
date_range = st.sidebar.date_input(
    "Pickup Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Handle single vs range selection
if isinstance(date_range, (list, tuple)):
    start_date = date_range[0]
    end_date = date_range[-1]
else:
    start_date = end_date = date_range

# Convert to datetime
start_datetime = pd.to_datetime(start_date)
end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)

# Hour range slider
hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))

# Payment filter
payment_labels_all = sorted(list(LABEL_TO_CODE.keys()))
selected_payments = st.sidebar.multiselect(
    "Payment Type",
    payment_labels_all,
    default=payment_labels_all,
)

# Zone filter
zones_list = sorted(zones["Zone"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Pickup Zones",
    zones_list,
    default=[],
    help="Leave empty to include all zones.",
)

# Get filtered dataset
filtered_df = get_filtered_df(
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    hour_min=hour_range[0],
    hour_max=hour_range[1],
    payment_labels=selected_payments,
    selected_zone_names=selected_zones,
)

# Stop app if no data matches filters
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# Key metrics section
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns([1, 0.8, 1.5, 1, 1])

# Display summary metrics
col1.metric("Total Trips", f"{len(filtered_df):,}")
col2.metric("Average Fare", f"${filtered_df['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered_df['total_amount'].sum():,.2f}")
col4.metric("Avg Distance", f"{filtered_df['trip_distance'].mean():.2f} mi")
col5.metric("Avg Duration", f"{filtered_df['trip_duration_minutes'].mean():.2f} min")

st.divider()


# Create dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Top Pickup Zones", "Avg Fare by Hour", "Trip Distance Dist.", "Payment Types", "Day/Hour Heatmap"]
)


# Bar chart showing top pickup zones
with tab1:
    st.subheader("Top 10 Pickup Zones by Trip Count")

@@ -242,7 +265,6 @@ def load_data() -> pd.DataFrame:
    )


# Line chart showing average fare by hour
with tab2:
    st.subheader("Average Fare by Hour of Day")

@@ -252,20 +274,25 @@ def load_data() -> pd.DataFrame:
        .reset_index()
    )

    fig2 = px.line(avg_fare_hour, x="pickup_hour", y="fare_amount", markers=True,
                   title="Average Fare by Pickup Hour")
    fig2 = px.line(
        avg_fare_hour,
        x="pickup_hour",
        y="fare_amount",
        markers=True,
        title="Average Fare by Pickup Hour",
        labels={"pickup_hour": "Pickup Hour", "fare_amount": "Average Fare ($)"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "**Insight:** Average fare shows a pronounced spike in the early morning, peaking around 5 AM at nearly \$28, "
        "which is significantly higher than the typical daytime range of roughly \$17 - \$20. "
        "**Insight:** Average fare shows a pronounced spike in the early morning, peaking around 5 AM at nearly $28, "
        "which is significantly higher than the typical daytime range of roughly $17–$20. "
        "After 7 AM, fares stabilize and remain relatively consistent throughout business hours. "
        "This pattern suggests early-morning trips are likely longer-distance rides, such as airport travel, "
        "rather than simply congestion-driven commuter traffic."
    )


# Histogram of trip distances
with tab3:
    st.subheader("Distribution of Trip Distances")

@@ -274,23 +301,21 @@ def load_data() -> pd.DataFrame:
    fig3 = px.histogram(
        filtered_df[filtered_df["trip_distance"] <= max_distance],
        x="trip_distance",
        nbins=50,
        title="Trip Distance Distribution (Trimmed at 99th Percentile)"
        nbins=40,
        title="Trip Distance Distribution (Trimmed at 99th Percentile)",
        labels={"trip_distance": "Trip Distance (miles)"}
    )

    fig3.update_layout(xaxis_title="Trip Distance (miles)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "**Insight:** The distribution is heavily right-skewed, with the majority of trips concentrated under approximately 3 miles. "
        "Trip counts decline sharply after 4 - 5 miles, indicating that most taxi rides are short urban journeys. "
        "A smaller secondary cluster appears in the 8 - 12 mile range and again near 18 - 20 miles, "
        "Trip counts decline sharply after 4–5 miles, indicating that most taxi rides are short urban journeys. "
        "A smaller secondary cluster appears in the 8–12 mile range and again near 18–20 miles, "
        "which is consistent with airport or cross-borough travel. "
        "Trimming at the 99th percentile prevents extreme outliers from compressing the main distribution while preserving the overall histogram structure."
    )


# Payment type breakdown
with tab4:
    st.subheader("Payment Type Breakdown")

@@ -301,20 +326,18 @@ def load_data() -> pd.DataFrame:
    )
    payment_counts.columns = ["Payment Type", "Trips"]

    fig4 = px.bar(payment_counts, x="Payment Type", y="Trips",
                  title="Payment Method Usage")
    fig4 = px.bar(payment_counts, x="Payment Type", y="Trips", title="Payment Method Usage")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        "**Insight:** Credit card payments overwhelmingly dominate taxi transactions, accounting for the vast majority of trips "
        "(well over 2 million rides), while cash represents a much smaller but still significant portion. "
        "**Insight:** Credit card payments overwhelmingly dominate taxi transactions, accounting for the vast majority of trips, "
        "while cash represents a much smaller but still significant portion. "
        "Other categories such as dispute, no charge, and missing payments contribute only a very small fraction of total trips. "
        "This heavy reliance on credit card transactions helps explain why tip percentage analysis is most reliable when restricted "
        "to card payments, as digital transactions consistently record gratuity amounts."
    )


# Heatmap showing trip volume by day and hour
with tab5:
    st.subheader("Trips by Day of Week and Hour")

@@ -348,4 +371,4 @@ def load_data() -> pd.DataFrame:


st.divider()
st.success("Dashboard loaded successfully. Use the sidebar filters to explore the data.")
