# Role Classification Back-End

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/skillab-project/role_classification)

## Description

This project implements the backend API for a **skill-based job classification framework** that determines whether a job posting belongs to an **Established** or **Emerging** occupation. It is built with FastAPI (Python) and exposes endpoints for:

- Fetching job postings from the SkillLab Tracker API, filtered by keywords and/or occupation IDs.
- Resolving ESCO skill URIs to human-readable labels.
- Building a sparse skill-presence feature matrix from job postings.
- Training a binary classifier (XGBoost, Random Forest, or Logistic Regression) on up to 50,000 jobs.
- Generating per-job SHAP-based explanations and radar skill-category profiles.
- Streaming the full analysis result as JSON, with file-based caching and lock management to prevent duplicate runs.

The service is part of the broader [SkillLab](https://github.com/skillab-project) project, which analyses the European labour market using Open Job Advertisements (OJA) data.

---

## Getting Started Guide

Follow the steps below to set up the service locally on your machine.

### Prerequisites

- **Git:** Installed on your system. ([Download Git](https://git-scm.com/downloads))
- **Python:** Version 3.11 or newer is recommended. ([Download Python](https://www.python.org/downloads/)) Ensure `pip` is available.
- **Access to the SkillLab Tracker API:** You will need valid credentials (`TRACKER_USERNAME` and `TRACKER_PASSWORD`) for `https://skillab-tracker.csd.auth.gr/api`.

---

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/skillab-project/role_classification.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd role_classification
   ```

3. **Create and Activate a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   ```

   Activate it:
   - **Linux/macOS:** `source venv/bin/activate`
   - **Windows:** `venv\Scripts\activate`

4. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Environment Setup:**

   Create a `.env` file in the project root with the following variables:

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   ```

   > **Note:** The `.env` file is included in the repository as an empty placeholder. Fill it in with your own credentials before running the service.

---

## Running the Application

### Locally

Start the FastAPI development server with Uvicorn:

```bash
uvicorn role_classification:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000`. Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.

> **Note:** The app is mounted under the `/role-classification` root path when deployed behind a reverse proxy. For local development, the default root `/` is used.

---

### With Docker

To run the application as a Docker container, ensure Docker and Docker Compose are installed on your system.

**Build and run with Docker Compose:**

```bash
docker-compose up --build
```

The service will be exposed on port `8005` of your host machine (mapped to container port `8000`).

**Or build and run manually:**

```bash
docker build -t role-classification .
docker run -p 8005:8000 --env-file .env role-classification
```

---

## API Endpoints

All endpoints are prefixed with `/api/analysis`.

### `POST /api/analysis/jobs_emergingdck_train`

Trains a binary classifier to distinguish **Emerging** from **Established** job roles based on skill profiles. Results are streamed as a JSON response and cached to disk for subsequent requests.

**Query Parameters:**

| Parameter       | Type    | Default    | Description                                                                 |
|-----------------|---------|------------|-----------------------------------------------------------------------------|
| `keywords`      | string  | —          | Comma-separated keywords to filter jobs (e.g. `"machine learning,cloud"`)   |
| `occupation_ids`| string  | —          | Comma-separated ESCO occupation IDs to filter jobs                          |
| `model_type`    | string  | `xgboost`  | Classifier to use: `xgboost`, `random_forest`, or `logistic`                |
| `max_jobs`      | integer | `60000`    | Maximum number of jobs to fetch. Set to `0` for unlimited.                  |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/analysis/jobs_emergingdck_train?keywords=machine+learning&model_type=xgboost"
```

**Response Structure (streamed JSON):**

```json
{
  "message": "✅ XGBOOST trained.",
  "model_type": "xgboost",
  "jobs_used": 12400,
  "total_jobs_available": 54000,
  "skills_dim": 1328,
  "positive_label_ratio": 0.43,
  "filters_used": { "keywords": ["machine learning"], "occupation_ids": [] },
  "descriptive_statistics": { ... },
  "job_diagnostics": [
    {
      "job_title": "Machine Learning Engineer",
      "classification": "Emerging",
      "probability": 0.91,
      "emerging_score": 91,
      "explanation": [ { "skill": "python", "impact": 0.34 }, ... ],
      "radar_profile": { "ai": 85, "data": 60, "cloud": 20, ... },
      "category_badges": ["🤖", "📊"]
    },
    ...
  ],
  "global_top_emerging_skills": [ ... ],
  "global_top_established_skills": [ ... ]
}
```

**Caching & Locking:**

- Completed analyses are saved to `Completed_Analyses/` and streamed directly on repeat requests.
- A `.lock` file prevents duplicate concurrent runs. Locks older than 30 minutes are treated as stale and removed automatically.

---

## Running the Tests

The `tests/` directory contains the test suite. With the virtual environment activated, run:

```bash
pytest tests/
```

---

## Project Structure

```
role_classification/
├── role_classification.py   # FastAPI app, classifier logic, streaming endpoint
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image definition
├── docker-compose.yml       # Compose configuration (port 8005)
├── .env                     # Environment variables (fill in before running)
├── jenkins/                 # CI/CD pipeline configuration
└── tests/                   # Test suite
```

---

## Technologies

- **Python 3.11**
- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **XGBoost / scikit-learn** — Classification models (XGBoost, Random Forest, Logistic Regression)
- **SHAP** — Model explainability
- **pandas / NumPy / SciPy** — Data processing and sparse matrix operations
- **python-dotenv** — Environment variable management
- **Docker / Docker Compose** — Containerised deployment

---

## License

This project is licensed under the **Eclipse Public License 2.0 (EPL-2.0)**. See the [LICENSE](LICENSE) file for details.
