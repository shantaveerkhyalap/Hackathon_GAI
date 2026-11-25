import os
import json
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


STAKEHOLDER_PROMPT = """You are TaxiLens AI, an urban mobility intelligence engine.
Based ONLY on the provided aggregated historical taxi statistics, generate actionable insights and recommendations for three key stakeholders.

CRITICAL PRIVACY & ANALYTICAL RULES:
- Use strictly aggregated statistical metrics (counts, percentages, percentiles, dayparts, hourly shares, trip durations).
- Never expose, invent, or attempt to reconstruct individual trip trajectories, coordinates, or timestamps.
- Avoid making real-time traffic predictions; clearly frame all recommendations as patterns derived from historical data.
- Return your output as clean, valid JSON matching the exact schema specified below without enclosing markdown backticks.

STAKEHOLDER GOALS:
1. Passenger Travel Recommendations:
   - Provide recommended travel times to avoid peak congestion.
   - Point out expected travel durations during different time periods.
   - Example advice: "Based on historical travel patterns, avoiding 5 PM to 8 PM may reduce travel delays. Traveling before peak hours can result in shorter trip durations."
2. Urban Traffic Management Suggestions:
   - Identify peak congestion periods from historical volume.
   - Suggest municipal traffic management strategies (signal timing, transit synchronization, lane allocation).
   - Example advice: "High taxi demand between 5 PM and 8 PM indicates potential congestion periods. City authorities can improve traffic management and public transport availability during these hours."
3. Taxi Fleet Optimization:
   - Identify high-demand time periods.
   - Suggest when operators should increase fleet availability to minimize passenger wait times.
   - Example advice: "Taxi operators should increase fleet availability during evening peak hours to reduce waiting time and improve service efficiency."

REQUIRED JSON OUTPUT SCHEMA:
{
  "passengers": {
    "title": "Passenger Travel Assistant",
    "summary": "2-3 sentence overview of historical transit durations, congestion windows, and travel expectations for passengers.",
    "observations": [
      "Observation 1 citing peak travel hours vs off-peak windows from historical data",
      "Observation 2 citing average journey durations during peak vs morning/afternoon trips"
    ],
    "actions": [
      "Action 1 advising optimal departure windows to avoid peak congestion delays",
      "Action 2 recommending travel time buffers based on historical duration percentiles"
    ]
  },
  "urban_planners": {
    "title": "Urban Mobility Intelligence",
    "summary": "2-3 sentence strategic briefing on municipal arterial load, peak congestion periods, and demand shifts.",
    "observations": [
      "Observation 1 highlighting peak demand hours (e.g. 5 PM to 8 PM) and weekday vs weekend volume concentration",
      "Observation 2 identifying highest-density dayparts requiring municipal corridor oversight"
    ],
    "actions": [
      "Action 1 recommending public transit synchronization or dynamic lane management during peak surges",
      "Action 2 suggesting demand-staggering policies or curb-space allocation for high-density hours"
    ]
  },
  "fleet_operators": {
    "title": "Fleet Management Advisor",
    "summary": "2-3 sentence operational guidance for taxi operators on dispatch timing, vehicle availability, and fleet staging.",
    "observations": [
      "Observation 1 pinpointing exact hours and dayparts where taxi demand crests",
      "Observation 2 analyzing trip distance distribution and vehicle turnover frequency"
    ],
    "actions": [
      "Action 1 specifying time windows to increase active fleet availability to reduce passenger wait times",
      "Action 2 recommending driver turnaround tactics in high-density short-hop zones vs regional corridors"
    ]
  }
}
"""

QA_SYSTEM_PROMPT = """You are TaxiLens AI Assistant, a specialized conversational intelligence engine for urban transit analytics.
You have access ONLY to pre-computed aggregate statistical metrics across NYC taxi mobility patterns.
CRITICAL PRIVACY RULE: Never attempt to invent or expose individual user journey paths, personal coordinates, or row-level timestamps.
Always clearly state that recommendations and metrics reflect historical taxi patterns. Answer concisely in 2-4 sentences.
"""

def generate_stakeholder_insights(aggregates: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates tailored recommendations for:
    1. Individual Travelers (Time Management)
    2. Urban Planners (Traffic Management)
    3. Taxi Companies (Fleet Optimization)
    
    Uses Gemini Flash if configured, otherwise uses the deterministic rule-based engine.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    
    if key and GENAI_AVAILABLE:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=STAKEHOLDER_PROMPT
            )
            prompt = (
                f"Aggregated Historical Statistics:\n```json\n{json.dumps(aggregates, indent=2)}\n```\n\n"
                "Generate the structured 3-stakeholder JSON response strictly adhering to the schema."
            )
            response = model.generate_content(prompt)
            if response and response.text:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                parsed = json.loads(cleaned_text.strip())
                if "passengers" in parsed and "urban_planners" in parsed and "fleet_operators" in parsed:
                    return parsed
        except Exception as e:
            print(f"[TaxiLens Narrator] Gemini call failed ({e}); falling back to deterministic synthesizer.")

    return _generate_deterministic_stakeholder_insights(aggregates)


def _generate_deterministic_stakeholder_insights(aggregates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic rule-based generator creating structured stakeholder recommendations
    grounded in historical taxi aggregate statistics.
    """
    summary = aggregates.get("summary", {})
    dayparts = aggregates.get("daypart_distribution", {})
    dist_tiers = aggregates.get("distance_tier_breakdown", {})
    important = aggregates.get("important_trips", {})
    
    total_trips = summary.get("total_trips", 0)
    avg_dur = summary.get("avg_duration_minutes", 15.0)
    median_dur = summary.get("median_duration_minutes", 11.7)
    p90_dur = summary.get("p90_duration_minutes", 27.7)
    avg_speed = summary.get("avg_speed_mph", 11.4)
    peak_hr = summary.get("peak_hour", 18)
    lowest_hr = summary.get("lowest_hour", 4)
    busiest_day = summary.get("busiest_day", "Thursday")
    weekday_share = summary.get("weekday_share_pct", 74.3)
    
    peak_str = f"{peak_hr % 12 or 12}:00 {'PM' if peak_hr >= 12 else 'AM'}"
    
    peak_dur = important.get("peak_hour_avg_duration", 16.5)
    eve_dur = important.get("evening_avg_duration", 15.8)
    morn_dur = important.get("morning_avg_duration", 13.9)
    hd_share = important.get("high_demand_share_pct", 28.5)
    
    short_share = dist_tiers.get("Short (<3 mi)", {}).get("share_pct", 73.4)
    evening_share = dayparts.get("Evening", {}).get("share_pct", 34.0)

    return {
        "passengers": {
            "title": "Passenger Travel Assistant",
            "summary": (
                f"Based on historical travel patterns, average trip duration during the {peak_str} peak reaches {peak_dur:.1f} minutes, "
                f"compared to {morn_dur:.1f} minutes during morning trips and {eve_dur:.1f} minutes in the evening. "
                f"Avoiding 5 PM to 8 PM can significantly reduce travel delays, as urban transit velocity slows to ~{avg_speed:.1f} mph."
            ),
            "observations": [
                f"Historical demand peaks at {peak_str}, with evening trips averaging {eve_dur:.1f} minutes compared to {morn_dur:.1f} minutes in the morning.",
                f"Over {short_share}% of rides are short hops (<3 miles); however, peak congestion can increase 90th percentile durations up to {p90_dur:.1f} minutes."
            ],
            "actions": [
                f"Based on historical patterns, traveling before 8 AM or after 10 AM may reduce travel delays and yield shorter trip durations.",
                f"When traveling during the 5 PM to 8 PM high-demand window, allow an extra 15–20 minute buffer to account for recurring urban slowdowns."
            ]
        },
        "urban_planners": {
            "title": "Urban Mobility Intelligence",
            "summary": (
                f"High taxi demand between 5 PM and 8 PM ({hd_share}% of daily volume) indicates concentrated congestion periods. "
                f"Arterial pressure is heavily weekday-driven ({weekday_share}% of total volume), with maximum street saturation occurring on {busiest_day} evenings."
            ),
            "observations": [
                f"High taxi demand between 5 PM and 8 PM indicates potential congestion periods requiring coordinated arterial management.",
                f"Short trips under 3 miles comprise {short_share}% of volume, demonstrating intense demand for localized downtown surface transit."
            ],
            "actions": [
                f"City authorities can improve traffic management and public transport availability during the 5 PM to 8 PM evening window.",
                f"Implement dedicated transit/taxi priority corridors and synchronize traffic signals along key avenues during peak evening commute hours."
            ]
        },
        "fleet_operators": {
            "title": "Fleet Management Advisor",
            "summary": (
                f"Evening hours show the highest demand across the metropolitan area, with trips peaking at {peak_str}. "
                f"Taxi operators should increase fleet availability during evening peak hours to reduce waiting time and improve service efficiency."
            ),
            "observations": [
                f"Evening hours represent {evening_share}% of daily ride requests, with {short_share}% concentrated in rapid under-3-mile trips.",
                f"Late-night and early-morning off-peak hours experience longer average distances with lower overall trip volume."
            ],
            "actions": [
                f"Taxi operators should increase fleet availability between 5 PM and 8 PM to reduce passenger waiting time and maximize driver utilization.",
                f"Reposition available vehicles into high-density commercial corridors starting by 3:30 PM ahead of the evening peak rush."
            ]
        }
    }


def generate_insights(aggregates: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """Legacy backward-compatible synthesis helper."""
    insights = generate_stakeholder_insights(aggregates, api_key)
    p = insights.get("passengers", {})
    u = insights.get("urban_planners", {})
    f = insights.get("fleet_operators", {})
    
    return f"""### 1. Smart Travel Recommendations for Passengers
{p.get('summary', '')}

### 2. Urban Mobility & Traffic Management Insights
{u.get('summary', '')}

### 3. Taxi Fleet Optimization Recommendations
{f.get('summary', '')}
"""


def answer_query(query: str, aggregates: Dict[str, Any], api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    
    if key and GENAI_AVAILABLE:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=QA_SYSTEM_PROMPT
            )
            prompt = (
                f"Context Aggregate Statistics (Zero raw rows):\n```json\n{json.dumps(aggregates, indent=2)}\n```\n\n"
                f"User Question: {query}\n\n"
                "Provide a clear, metric-grounded answer in 2-4 sentences, noting that insights derive from historical taxi patterns."
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[TaxiLens Narrator] Gemini Q&A failed ({e}); using deterministic Q&A matcher.")

    return _deterministic_answer(query, aggregates)


def _deterministic_answer(query: str, aggregates: Dict[str, Any]) -> str:
    q = query.lower()
    summary = aggregates.get("summary", {})
    dayparts = aggregates.get("daypart_distribution", {})
    dist_tiers = aggregates.get("distance_tier_breakdown", {})
    important = aggregates.get("important_trips", {})
    
    peak_h = summary.get("peak_hour", 18)
    peak_h_str = f"{peak_h % 12 or 12}:00 {'PM' if peak_h >= 12 else 'AM'}"
    busiest_day = summary.get("busiest_day", "Thursday")
    avg_dist = summary.get("avg_distance_miles", 3.17)
    med_dist = summary.get("median_distance_miles", 1.7)
    avg_dur = summary.get("avg_duration_minutes", 15.0)
    peak_dur = important.get("peak_hour_avg_duration", 16.5)
    morn_dur = important.get("morning_avg_duration", 13.9)
    eve_dur = important.get("evening_avg_duration", 15.8)
    weekday_share = summary.get("weekday_share_pct", 74.3)
    short_share = dist_tiers.get("Short (<3 mi)", {}).get("share_pct", 73.4)

    if any(w in q for w in ["busiest", "busy", "peak", "when"]):
        return (
            f"Based on historical patterns, taxis experience peak demand at **{peak_h_str}**, with average trip duration reaching **{peak_dur:.1f} minutes**. "
            f"Across the week, **{busiest_day}** records the highest trip volume, with weekdays accounting for **{weekday_share}%** of all trips. "
            f"Avoiding 5 PM to 8 PM can result in shorter travel durations."
        )
    elif any(w in q for w in ["average", "pattern", "typical", "distance", "duration", "time", "important"]):
        return (
            f"Historical records indicate an average trip duration of **{avg_dur:.1f} minutes** and distance of **{avg_dist:.2f} miles**. "
            f"During important trip windows, morning trips average **{morn_dur:.1f} minutes**, evening trips average **{eve_dur:.1f} minutes**, "
            f"and peak hour ({peak_h_str}) trips average **{peak_dur:.1f} minutes**."
        )
    else:
        return (
            f"Based on historical travel patterns, traveling before peak hours (before 8 AM or between 11 AM and 3 PM) results in shorter trip durations. "
            f"High taxi demand between 5 PM and 8 PM indicates potential congestion periods where city authorities and taxi operators recommend adjusted travel and increased fleet availability."
        )
