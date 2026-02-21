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


# Configure page settings before any Streamlit content is rendered
st.set_page_config(
    page_title="NYC Taxi Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS styling for dashboard headers
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


# Load cleaned dataset and perform final preparation steps
@st.cache_data
def load_data() -> pd.DataFrame:
    trip_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    zone_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

    zones = pd.read_csv(zone_url)

    df = duckdb.query(f"""
        SELECT
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            PULocationID,
            DOLocationID,
            passenger_count,
            trip_distance,
            fare_amount,
            tip_amount,
            total_amount,
            payment_type
        FROM read_parquet('{trip_url}')
    """).to_df()

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    df = df.dropna(subset=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "fare_amount"
    ])

    df = df[
        (df["trip_distance"] > 0) &
        (df["fare_amount"] > 0) &
        (df["fare_amount"] <= 500) &
        (df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"])
    ]

    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["trip_speed_mph"] = df["trip_distance"] / (
        df["trip_duration_minutes"].replace(0, pd.NA) / 60
    )

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()

    df = df.merge(
        zones[["LocationID", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    ).rename(columns={"Zone": "pickup_zone"})

    payment_map = {
        1: "Credit Card",
        2: "Cash",
        3: "No Charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided Trip",
    }
    df["payment_label"] = df["payment_type"].map(payment_map).fillna("Other/Missing")

    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["pickup_day_of_week"] = pd.Categorical(
        df["pickup_day_of_week"],
        categories=day_order,
        ordered=True
    )

    return df

df = load_data()


# Dashboard title and subtitle
st.markdown('<div class="main-header">NYC Taxi Trip Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Yellow Taxi Trips - January 2024</div>', unsafe_allow_html=True)
#description of the dashboard
st.markdown(
    """
    This dashboard provides insights into trip patterns, fare dynamics, and payment methods for January 2024.  
    Explore key metrics and trends in NYC taxi trips.  Use the filters in the sidebar to customize the data view.
    """,
    unsafe_allow_html=True,
)
st.divider()


# Sidebar filters allow interactive exploration of the dataset
st.sidebar.header("Filters")

min_date = pd.to_datetime(df["tpep_pickup_datetime"].min()).date()
max_date = pd.to_datetime(df["tpep_pickup_datetime"].max()).date()

# Date range selector
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

# Hour range selector
hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))

# Payment type selector
payment_labels = sorted(df["payment_label"].dropna().unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Payment Type",
    payment_labels,
    default=payment_labels,
)

# Optional zone selector
zones_list = sorted(df["pickup_zone"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Pickup Zones",
    zones_list,
    default=[],
    help="Leave empty to include all zones.",
)


# Apply filtering logic based on sidebar selections
filtered_df = df[
    (df["tpep_pickup_datetime"] >= start_datetime)
    & (df["tpep_pickup_datetime"] < end_datetime)
    & (df["pickup_hour"] >= hour_range[0])
    & (df["pickup_hour"] <= hour_range[1])
    & (df["payment_label"].isin(selected_payments))
]

if selected_zones:
    filtered_df = filtered_df[filtered_df["pickup_zone"].isin(selected_zones)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# Display high-level summary statistics
st.subheader("Key Metrics")

#was squeezing total revenue value 
#col1, col2, col3, col4, col5 = st.columns(5)
col1, col2, col3, col4, col5 = st.columns([1, 0.8, 1.5, 1, 1])
col1.metric("Total Trips", f"{len(filtered_df):,}")
col2.metric("Average Fare", f"${filtered_df['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered_df['total_amount'].sum():,.2f}")
col4.metric("Avg Distance", f"{filtered_df['trip_distance'].mean():.2f} mi")
col5.metric("Avg Duration", f"{filtered_df['trip_duration_minutes'].mean():.2f} min")

st.divider()


# Organize visualizations into tabs for structured dashboard layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Top Pickup Zones", "Avg Fare by Hour", "Trip Distance Dist.", "Payment Types", "Day/Hour Heatmap"]
)


# Bar chart showing top pickup zones
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


# Line chart showing average fare by hour
with tab2:
    st.subheader("Average Fare by Hour of Day")

    avg_fare_hour = (
        filtered_df.groupby("pickup_hour")["fare_amount"]
        .mean()
        .reset_index()
    )

    fig2 = px.line(avg_fare_hour, x="pickup_hour", y="fare_amount", markers=True,
                   title="Average Fare by Pickup Hour")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "**Insight:** Average fare shows a pronounced spike in the early morning, peaking around 5 AM at nearly \$28, "
        "which is significantly higher than the typical daytime range of roughly \$17 - \$20. "
        "After 7 AM, fares stabilize and remain relatively consistent throughout business hours. "
        "This pattern suggests early-morning trips are likely longer-distance rides, such as airport travel, "
        "rather than simply congestion-driven commuter traffic."
    )


# Histogram of trip distances
with tab3:
    st.subheader("Distribution of Trip Distances")

    max_distance = filtered_df["trip_distance"].quantile(0.99)

    fig3 = px.histogram(
        filtered_df[filtered_df["trip_distance"] <= max_distance],
        x="trip_distance",
        nbins=50,
        title="Trip Distance Distribution (Trimmed at 99th Percentile)"
    )

    fig3.update_layout(xaxis_title="Trip Distance (miles)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "**Insight:** The distribution is heavily right-skewed, with the majority of trips concentrated under approximately 3 miles. "
        "Trip counts decline sharply after 4 - 5 miles, indicating that most taxi rides are short urban journeys. "
        "A smaller secondary cluster appears in the 8 - 12 mile range and again near 18 - 20 miles, "
        "which is consistent with airport or cross-borough travel. "
        "Trimming at the 99th percentile prevents extreme outliers from compressing the main distribution while preserving the overall histogram structure."
    )


# Payment type breakdown
with tab4:
    st.subheader("Payment Type Breakdown")

    payment_counts = (
        filtered_df["payment_label"]
        .value_counts()
        .reset_index()
    )
    payment_counts.columns = ["Payment Type", "Trips"]

    fig4 = px.bar(payment_counts, x="Payment Type", y="Trips",
                  title="Payment Method Usage")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        "**Insight:** Credit card payments overwhelmingly dominate taxi transactions, accounting for the vast majority of trips "
        "(well over 2 million rides), while cash represents a much smaller but still significant portion. "
        "Other categories such as dispute, no charge, and missing payments contribute only a very small fraction of total trips. "
        "This heavy reliance on credit card transactions helps explain why tip percentage analysis is most reliable when restricted "
        "to card payments, as digital transactions consistently record gratuity amounts."
    )


# Heatmap showing trip volume by day and hour
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