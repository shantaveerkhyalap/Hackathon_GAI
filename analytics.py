import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_aggregates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level aggregated summary statistics across dimensions.
    
    Privacy Guardrail Guarantee:
    Zero granular trip trajectories, individual GPS coordinates, or specific row
    timestamps are retained. Output consists strictly of mathematical aggregations
    (counts, percentiles, means, proportions).
    """
    if df.empty:
        return {}

    total_trips = int(len(df))
    
    # Core Overview Metrics
    avg_distance = float(round(df['trip_distance'].mean(), 2))
    median_distance = float(round(df['trip_distance'].median(), 2))
    p90_distance = float(round(df['trip_distance'].quantile(0.90), 2))
    
    avg_duration = float(round(df['duration_minutes'].mean(), 2))
    median_duration = float(round(df['duration_minutes'].median(), 2))
    p90_duration = float(round(df['duration_minutes'].quantile(0.90), 2))
    
    # Speed estimation (mph) = distance / (duration_minutes / 60)
    valid_speeds = df['trip_distance'] / (df['duration_minutes'] / 60.0)
    avg_speed_mph = float(round(valid_speeds[valid_speeds.between(1, 80)].mean(), 1)) if not valid_speeds.empty else 0.0

    # Hourly Distribution & Durations
    hourly_counts = df['pickup_hour'].value_counts().sort_index()
    hourly_pct = (hourly_counts / total_trips * 100).round(2).to_dict()
    peak_hour = int(hourly_counts.idxmax())
    lowest_hour = int(hourly_counts.idxmin())

    # Average duration during peak hour
    peak_hour_df = df[df['pickup_hour'] == peak_hour]
    peak_hour_avg_dur = float(round(peak_hour_df['duration_minutes'].mean(), 1)) if not peak_hour_df.empty else 0.0

    # High Demand Window (e.g. 5 PM to 8 PM / hours 17, 18, 19, 20)
    high_demand_df = df[df['pickup_hour'].isin([17, 18, 19, 20])]
    high_demand_avg_dur = float(round(high_demand_df['duration_minutes'].mean(), 1)) if not high_demand_df.empty else 0.0
    high_demand_share_pct = float(round((len(high_demand_df) / total_trips) * 100, 1)) if total_trips > 0 else 0.0

    # Day of Week Distribution
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = df['pickup_weekday'].value_counts()
    dow_stats = {day: int(dow_counts.get(day, 0)) for day in days_order}
    busiest_day = str(dow_counts.idxmax())

    # Day Type (Weekday vs Weekend)
    day_type_counts = df['day_type'].value_counts()
    weekday_share_pct = float(round((day_type_counts.get('Weekday', 0) / total_trips) * 100, 1))
    weekend_share_pct = float(round((day_type_counts.get('Weekend', 0) / total_trips) * 100, 1))

    # Time of Day (Dayparts) - durations & volumes
    daypart_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    daypart_counts = df['time_of_day'].value_counts()
    daypart_stats = {
        dp: {
            "trip_count": int(daypart_counts.get(dp, 0)),
            "share_pct": float(round((daypart_counts.get(dp, 0) / total_trips) * 100, 1)),
            "avg_distance": float(round(df[df['time_of_day'] == dp]['trip_distance'].mean(), 2)) if dp in daypart_counts else 0.0,
            "avg_duration": float(round(df[df['time_of_day'] == dp]['duration_minutes'].mean(), 1)) if dp in daypart_counts else 0.0
        }
        for dp in daypart_order
    }

    # Distance Tiers
    dist_tier_counts = df['distance_tier'].value_counts()
    distance_tier_breakdown = {
        tier: {
            "count": int(dist_tier_counts.get(tier, 0)),
            "share_pct": float(round((dist_tier_counts.get(tier, 0) / total_trips) * 100, 1)),
            "avg_duration": float(round(df[df['distance_tier'] == tier]['duration_minutes'].mean(), 1)) if tier in dist_tier_counts else 0.0
        }
        for tier in ['Short (<3 mi)', 'Medium (3-10 mi)', 'Long (>10 mi)']
    }

    # Duration Tiers
    dur_tier_counts = df['duration_tier'].value_counts()
    duration_tier_breakdown = {
        tier: {
            "count": int(dur_tier_counts.get(tier, 0)),
            "share_pct": float(round((dur_tier_counts.get(tier, 0) / total_trips) * 100, 1))
        }
        for tier in ['Under 15m', '15-30m', '30-60m', 'Over 60m']
    }

    # Important Trips Summary
    important_trips = {
        "peak_hour": peak_hour,
        "peak_hour_avg_duration": peak_hour_avg_dur,
        "high_demand_window": "5 PM – 8 PM",
        "high_demand_avg_duration": high_demand_avg_dur,
        "high_demand_share_pct": high_demand_share_pct,
        "morning_avg_duration": daypart_stats.get("Morning", {}).get("avg_duration", 0.0),
        "afternoon_avg_duration": daypart_stats.get("Afternoon", {}).get("avg_duration", 0.0),
        "evening_avg_duration": daypart_stats.get("Evening", {}).get("avg_duration", 0.0),
        "night_avg_duration": daypart_stats.get("Night", {}).get("avg_duration", 0.0)
    }

    return {
        "summary": {
            "total_trips": total_trips,
            "avg_distance_miles": avg_distance,
            "median_distance_miles": median_distance,
            "p90_distance_miles": p90_distance,
            "avg_duration_minutes": avg_duration,
            "median_duration_minutes": median_duration,
            "p90_duration_minutes": p90_duration,
            "avg_speed_mph": avg_speed_mph,
            "busiest_day": busiest_day,
            "peak_hour": peak_hour,
            "lowest_hour": lowest_hour,
            "weekday_share_pct": weekday_share_pct,
            "weekend_share_pct": weekend_share_pct
        },
        "important_trips": important_trips,
        "daypart_distribution": daypart_stats,
        "hourly_distribution_pct": hourly_pct,
        "weekday_distribution": dow_stats,
        "distance_tier_breakdown": distance_tier_breakdown,
        "duration_tier_breakdown": duration_tier_breakdown
    }
