# TaxiLens AI – Taxi Trip Pattern Narrator

An analytical urban mobility intelligence platform that transforms historical NYC taxi trip records into statistical patterns, interactive visualizations, and executive narrative summaries.

Designed and engineered for the Hackathon Mobility Analytics Challenge.

---

## System Architecture

```text
Hackathon_project/
├── data/
│   └── sample_taxi_trips.csv      # Sample TLC dataset (15,000 trips)
├── backend/
│   ├── __init__.py                # Package exports
│   ├── preprocessor.py            # Data hygiene, schema validation, temporal extraction
│   ├── analytics.py               # Statistical aggregation engine & privacy guardrail
│   └── narrator.py                # Gemini Flash narrative generator & analytical fallback
├── frontend/
│   └── app.py                     # Streamlit executive dashboard & Plotly visual suite
├── requirements.txt               # Project dependencies
├── Dockerfile                     # Containerization blueprint
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions (protects .env / venv)
├── .env.example                   # Environment configuration template
└── README.md                      # Documentation & execution instructions
```

---

## Key Features

1. **Automated Preprocessing & Data Hygiene Pipeline**
   - Cleans null values, schema mismatches, and duplicate records.
   - Filters out non-positive trip durations and unfeasible speeds.
   - Extracts rich analytical dimensions: `duration_minutes`, `pickup_hour`, `pickup_weekday`, `day_type`, and `time_of_day`.

2. **Privacy-Preserving Aggregation Engine**
   - **Zero Raw Row Exposure**: No individual trip records, trajectory coordinates, or personal rider data ever leave the local computing boundary.
   - Transforms thousands of micro-journeys into high-level statistical aggregates (percentiles, cadence curves, daypart distribution, tier shares).

3. **Interactive Analytical Dashboard**
   - KPI metric cards tracking valid trip counts, average & median distance, trip duration, peak rush hours, and transit velocity.
   - **Average Time for Important Trips**: Visual benchmark cards showing expected durations across peak hours, evening rushes, and daylight periods.
   - Interactive Plotly charts: Hourly trip demand profile, Daypart distribution, Distance tier breakdown, and Weekly rhythm.
   - Support for custom TLC CSV uploads or instant one-click analysis using the bundled 15,000-trip sample.

4. **Multi-Stakeholder AI Mobility Intelligence (Gemini Flash + Fallback)**
   - Generates tailored, structured recommendations for:
     1. **Passenger Travel Assistant** (Time management & congestion bypass).
     2. **Urban Mobility Intelligence** (Arterial pressure & transit synchronization).
     3. **Fleet Management Advisor** (Vehicle staging & dispatch turnover).
   - Fully resilient: seamlessly utilizes an internal analytical synthesis engine if no Gemini API key is configured.

---

## Dataset Reference

TaxiLens AI is compatible with official New York City Taxi and Limousine Commission (TLC) Trip Record Data:
- **Official TLC Dataset Portal**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

A pre-packaged sample dataset (`data/sample_taxi_trips.csv`) with 15,000 trips is bundled for quick validation and live demos.

---

## Getting Started

### Method A: Run Locally (Python 3.10+)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv

# Windows:
venv\Scripts\activate
# Linux / WSL / macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set your Gemini API key in .env
cp .env.example .env

# 4. Launch the application
streamlit run frontend/app.py
```

---

### Method B: Run with Docker

Build and run the containerized application on any operating system:

```bash
# 1. Build the Docker image
docker build -t taxilens-ai .

# 2. Run the container
docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" taxilens-ai
```

Open your browser at **`http://localhost:8501`**.
