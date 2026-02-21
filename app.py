"""
NYC Taxi Trip Dashboard
COMP 3610 - Assignment 1

Run with:
streamlit run app.py
"""

import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="NYC Taxi Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    unsafe_allow_html=True,
)


TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

PAYMENT_MAP = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided Trip",
}

LABEL_TO_CODE = {v: k for k, v in PAYMENT_MAP.items()}


@st.cache_data
def load_zones() -> pd.DataFrame:
    zones_df = pd.read_csv(ZONE_URL)
    return zones_df


@st.cache_data
def get_min_max_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    # Get min/max directly from parquet without loading all rows into pandas
    res = duckdb.query(f"""
        SELECT
            MIN(tpep_pickup_datetime) AS min_dt,
            MAX(tpep_pickup_datetime) AS max_dt
        FROM read_parquet('{TRIP_URL}')
    """).to_df()

    min_dt = pd.to_datetime(res.loc[0, "min_dt"], errors="coerce")
    max_dt = pd.to_datetime(res.loc[0, "max_dt"], errors="coerce")
    return min_dt, max_dt


@st.cache_data
def get_filtered_df(
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    hour_min: int,
    hour_max: int,
    payment_labels: list[str],
    selected_zone_names: list[str],
) -> pd.DataFrame:
    zones_df = load_zones()

    # Convert readable payment labels -> payment_type codes for SQL
    payment_codes = [LABEL_TO_CODE[p] for p in payment_labels if p in LABEL_TO_CODE]
    if not payment_codes:
        # If user unselects everything, return empty
        return pd.DataFrame()

    zone_filter_sql = ""
    if selected_zone_names:
        loc_ids = zones_df[zones_df["Zone"].isin(selected_zone_names)]["LocationID"].dropna().astype(int).unique().tolist()
        if loc_ids:
            zone_filter_sql = f"AND PULocationID IN ({','.join(map(str, loc_ids))})"

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

    df = duckdb.query(sql).to_df()

    # Datetimes + derived fields (cheap once data is filtered)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    # Attach pickup zone names for charts
    df = df.merge(
        zones_df[["LocationID", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    ).rename(columns={"Zone": "pickup_zone"})

    # Payment labels for chart
    df["payment_label"] = df["payment_type"].map(PAYMENT_MAP).fillna("Other/Missing")

    # Ordered weekdays for heatmap
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["pickup_day_of_week"] = pd.Categorical(df["pickup_day_of_week"], categories=day_order, ordered=True)

    return df


# Header
st.markdown('<div class="main-header">NYC Taxi Trip Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Yellow Taxi Trips - January 2024</div>', unsafe_allow_html=True)

st.markdown(
    """
    This dashboard provides insights into trip patterns, fare dynamics, and payment methods for January 2024.  
    Explore key metrics and trends in NYC taxi trips. Use the filters in the sidebar to customize the data view.
    """,
    unsafe_allow_html=True,
)

st.divider()


# Sidebar filters
st.sidebar.header("Filters")

zones = load_zones()
min_dt, max_dt = get_min_max_dates()

min_date = min_dt.date()
max_date = max_dt.date()

date_range = st.sidebar.date_input(
    "Pickup Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)):
    start_date = date_range[0]
    end_date = date_range[-1]
else:
    start_date = end_date = date_range

start_datetime = pd.to_datetime(start_date)
end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)

hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))

payment_labels_all = sorted(list(LABEL_TO_CODE.keys()))
selected_payments = st.sidebar.multiselect(
    "Payment Type",
    payment_labels_all,
    default=payment_labels_all,
)

zones_list = sorted(zones["Zone"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Pickup Zones",
    zones_list,
    default=[],
    help="Leave empty to include all zones.",
)

filtered_df = get_filtered_df(
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    hour_min=hour_range[0],
    hour_max=hour_range[1],
    payment_labels=selected_payments,
    selected_zone_names=selected_zones,
)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# Key metrics
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns([1, 0.8, 1.5, 1, 1])
col1.metric("Total Trips", f"{len(filtered_df):,}")
col2.metric("Average Fare", f"${filtered_df['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered_df['total_amount'].sum():,.2f}")
col4.metric("Avg Distance", f"{filtered_df['trip_distance'].mean():.2f} mi")
col5.metric("Avg Duration", f"{filtered_df['trip_duration_minutes'].mean():.2f} min")

st.divider()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Top Pickup Zones", "Avg Fare by Hour", "Trip Distance Dist.", "Payment Types", "Day/Hour Heatmap"]
)


with tab1:
    st.subheader("Top 10 Pickup Zones by Trip Count")

    top_zones = (
        filtered_df["pickup_zone"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_zones.columns = ["Pickup Zone", "Trips"]

    fig1 = px.bar(top_zones, x="Pickup Zone", y="Trips", title="Top 10 Pickup Zones")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        "**Insight:** Midtown Center, Upper East Side South, and JFK Airport consistently rank among the highest pickup zones, "
        "each exceeding roughly 130,000 trips in the full dataset. The presence of both major commercial districts and airport zones "
        "indicates that taxi demand is driven by a combination of commuter, residential, and travel-related activity. "
        "The relatively small difference between the top four zones suggests sustained high demand across central Manhattan, "
        "while the drop after these zones shows demand becomes more dispersed outside the most transit-connected areas."
    )


with tab2:
    st.subheader("Average Fare by Hour of Day")

    avg_fare_hour = (
        filtered_df.groupby("pickup_hour")["fare_amount"]
        .mean()
        .reset_index()
    )

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
        "**Insight:** Average fare shows a pronounced spike in the early morning, peaking around 5 AM at nearly $28, "
        "which is significantly higher than the typical daytime range of roughly $17–$20. "
        "After 7 AM, fares stabilize and remain relatively consistent throughout business hours. "
        "This pattern suggests early-morning trips are likely longer-distance rides, such as airport travel, "
        "rather than simply congestion-driven commuter traffic."
    )


with tab3:
    st.subheader("Distribution of Trip Distances")

    max_distance = filtered_df["trip_distance"].quantile(0.99)

    fig3 = px.histogram(
        filtered_df[filtered_df["trip_distance"] <= max_distance],
        x="trip_distance",
        nbins=40,
        title="Trip Distance Distribution (Trimmed at 99th Percentile)",
        labels={"trip_distance": "Trip Distance (miles)"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "**Insight:** The distribution is heavily right-skewed, with the majority of trips concentrated under approximately 3 miles. "
        "Trip counts decline sharply after 4–5 miles, indicating that most taxi rides are short urban journeys. "
        "A smaller secondary cluster appears in the 8–12 mile range and again near 18–20 miles, "
        "which is consistent with airport or cross-borough travel. "
        "Trimming at the 99th percentile prevents extreme outliers from compressing the main distribution while preserving the overall histogram structure."
    )


with tab4:
    st.subheader("Payment Type Breakdown")

    payment_counts = (
        filtered_df["payment_label"]
        .value_counts()
        .reset_index()
    )
    payment_counts.columns = ["Payment Type", "Trips"]

    fig4 = px.bar(payment_counts, x="Payment Type", y="Trips", title="Payment Method Usage")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        "**Insight:** Credit card payments overwhelmingly dominate taxi transactions, accounting for the vast majority of trips, "
        "while cash represents a much smaller but still significant portion. "
        "Other categories such as dispute, no charge, and missing payments contribute only a very small fraction of total trips. "
        "This heavy reliance on credit card transactions helps explain why tip percentage analysis is most reliable when restricted "
        "to card payments, as digital transactions consistently record gratuity amounts."
    )


with tab5:
    st.subheader("Trips by Day of Week and Hour")

    heatmap_data = (
        filtered_df.groupby(["pickup_day_of_week", "pickup_hour"])
        .size()
        .reset_index(name="Trips")
    )

    fig5 = px.density_heatmap(
        heatmap_data,
        x="pickup_hour",
        y="pickup_day_of_week",
        z="Trips",
        title="Trip Volume Heatmap",
        labels={
            "pickup_hour": "Pickup Hour",
            "pickup_day_of_week": "Day of Week",
            "Trips": "Number of Trips"
        }
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        "**Insight:** Trip volume is consistently highest during late morning through early evening hours "
        "(roughly 10 AM to 4 PM) across most weekdays, with particularly strong intensity midweek. "
        "Very early morning hours (around midnight to 5 AM) show the lowest activity levels throughout the week. "
        "Weekend patterns differ slightly, with Saturday and Sunday maintaining steadier activity later into the evening. "
        "This suggests that daytime commercial activity drives the majority of taxi demand rather than narrow commuter spikes."
    )


st.divider()
st.success("Dashboard loaded successfully. Use the sidebar filters to explore the data.")
