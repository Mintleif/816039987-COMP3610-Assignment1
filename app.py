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
