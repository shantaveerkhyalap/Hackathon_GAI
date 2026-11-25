import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def clean_and_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-speed, robust preprocessor for NYC TLC and general taxi trip records.
    Optimized for vectorization with flexible column detection.
    """
    initial_count = len(df)
    if initial_count == 0:
        return pd.DataFrame(), {"raw_record_count": 0, "clean_record_count": 0, "removed_records": 0, "retention_rate_pct": 0.0}

    # Case-insensitive column matching map
    norm_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    # Identify pickup datetime column
    pickup_col = None
    for candidate in [
        'tpep_pickup_datetime', 'pickup_datetime', 'lpep_pickup_datetime', 
        'pickup_time', 'start_time', 'trip_pickup_datetime', 'pickup'
    ]:
        if candidate in norm_cols:
            pickup_col = norm_cols[candidate]
            break

    # Identify dropoff datetime column
    dropoff_col = None
    for candidate in [
        'tpep_dropoff_datetime', 'dropoff_datetime', 'lpep_dropoff_datetime',
        'dropoff_time', 'end_time', 'trip_dropoff_datetime', 'dropoff'
    ]:
        if candidate in norm_cols:
            dropoff_col = norm_cols[candidate]
            break

    # Identify distance column
    distance_col = None
    for candidate in [
        'trip_distance', 'distance', 'trip_miles', 'miles', 'trip_dist'
    ]:
        if candidate in norm_cols:
            distance_col = norm_cols[candidate]
            break

    if not pickup_col or not dropoff_col or not distance_col:
        found = list(df.columns)
        raise ValueError(
            f"Missing required columns (pickup datetime, dropoff datetime, or distance). "
            f"Detected columns: {found[:10]}"
        )

    # Subsample if dataset is massive (>100k rows) for instant sub-second dashboard reactivity
    if initial_count > 100000:
        df_work = df.sample(n=100000, random_state=42).copy()
    else:
        df_work = df.copy()

    # Fast datetime & numeric parsing with format='mixed'
    df_work['pickup_datetime'] = pd.to_datetime(df_work[pickup_col], errors='coerce', format='mixed')
    df_work['dropoff_datetime'] = pd.to_datetime(df_work[dropoff_col], errors='coerce', format='mixed')
    df_work['trip_distance'] = pd.to_numeric(df_work[distance_col], errors='coerce')

    # Drop missing coordinates
    df_work = df_work.dropna(subset=['pickup_datetime', 'dropoff_datetime', 'trip_distance'])

    # Compute duration in minutes (Vectorized)
    df_work['duration_minutes'] = (df_work['dropoff_datetime'] - df_work['pickup_datetime']).dt.total_seconds() / 60.0

    # Vectorized filtering: positive duration (under 24 hours) & positive distance (under 100 miles)
    valid_mask = (
        (df_work['duration_minutes'] >= 0.5) &
        (df_work['duration_minutes'] <= 1440.0) &
        (df_work['trip_distance'] > 0.0) &
        (df_work['trip_distance'] <= 100.0)
    )
    df_clean = df_work[valid_mask].drop_duplicates().copy()

    # Vectorized temporal extraction
    dt_series = df_clean['pickup_datetime']
    df_clean['pickup_hour'] = dt_series.dt.hour
    df_clean['day_of_week_num'] = dt_series.dt.dayofweek
    df_clean['pickup_weekday'] = dt_series.dt.day_name()
    df_clean['day_type'] = np.where(df_clean['day_of_week_num'] >= 5, 'Weekend', 'Weekday')

    # Fast vectorized Daypart assignment via np.select
    hours = df_clean['pickup_hour'].values
    daypart_conds = [
        (hours >= 6) & (hours < 11),
        (hours >= 11) & (hours < 16),
        (hours >= 16) & (hours < 21)
    ]
    daypart_choices = ['Morning', 'Afternoon', 'Evening']
    df_clean['time_of_day'] = np.select(daypart_conds, daypart_choices, default='Night')

    # Fast vectorized Distance Tiers assignment
    dists = df_clean['trip_distance'].values
    dist_conds = [
        dists < 3.0,
        (dists >= 3.0) & (dists <= 10.0)
    ]
    dist_choices = ['Short (<3 mi)', 'Medium (3-10 mi)']
    df_clean['distance_tier'] = np.select(dist_conds, dist_choices, default='Long (>10 mi)')

    # Fast vectorized Duration Tiers assignment
    durs = df_clean['duration_minutes'].values
    dur_conds = [
        durs < 15.0,
        (durs >= 15.0) & (durs <= 30.0),
        (durs > 30.0) & (durs <= 60.0)
    ]
    dur_choices = ['Under 15m', '15-30m', '30-60m']
    df_clean['duration_tier'] = np.select(dur_conds, dur_choices, default='Over 60m')

    # Cleaning statistics
    clean_count = len(df_clean)
    cleaning_stats = {
        "raw_record_count": initial_count,
        "clean_record_count": clean_count,
        "removed_records": max(0, initial_count - clean_count),
        "retention_rate_pct": round((clean_count / initial_count) * 100, 2) if initial_count > 0 else 0.0
    }

    return df_clean, cleaning_stats
