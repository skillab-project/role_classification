import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Import the app
from role_classification import app

client = TestClient(app)

# ==========================================
# 1. MOCK DATA
# ==========================================

MOCK_JOBS_RESPONSE = {
    "items": [
        {
            "title": "AI Research Scientist",
            "skills": ["http://data.europa.eu/esco/skill/s1", "http://data.europa.eu/esco/skill/s2"]
        },
        {
            "title": "PHP Web Developer",
            "skills": ["http://data.europa.eu/esco/skill/s3"]
        },
        {
            "title": "GenAI Prompt Engineer",
            "skills": ["http://data.europa.eu/esco/skill/s1", "http://data.europa.eu/esco/skill/s4"]
        },
        {
            "title": "Legacy Mainframe Technician",
            "skills": ["http://data.europa.eu/esco/skill/s5"]
        }
    ]
}

MOCK_SKILLS_RESPONSE = {
    "items": [
        {"id": "http://data.europa.eu/esco/skill/s1", "label": "artificial intelligence development"},
        {"id": "http://data.europa.eu/esco/skill/s2", "label": "machine learning"},
        {"id": "http://data.europa.eu/esco/skill/s3", "label": "php"},
        {"id": "http://data.europa.eu/esco/skill/s4", "label": "prompt engineering"},
        {"id": "http://data.europa.eu/esco/skill/s5", "label": "cobol"}
    ]
}

# ==========================================
# 2. TESTS
# ==========================================

@patch("requests.post")
def test_train_emerging_classifier_xgboost(mock_post):
    """Test full training pipeline using XGBoost (Default)."""
    
    # Mocking sequence of API calls:
    # 1. Login
    # 2. Jobs Page 1
    # 3. Skills Page 1
    
    mock_login = MagicMock()
    mock_login.text = '"fake_token"'
    mock_login.status_code = 200
    
    mock_jobs = MagicMock()
    mock_jobs.json.return_value = MOCK_JOBS_RESPONSE
    mock_jobs.status_code = 200
    
    mock_skills = MagicMock()
    mock_skills.json.return_value = MOCK_SKILLS_RESPONSE
    mock_skills.status_code = 200
    
    # We provide enough responses for the loops in the code
    # Code fetches max_pages of jobs and 40 pages of skills
    mock_post.side_effect = [mock_login] + [mock_jobs]*8 + [mock_skills]*40

    response = client.post("/api/analysis/jobs_emergingdck_train?keywords=ai,data&model_type=xgboost")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert data["model_type"] == "xgboost"
    assert "descriptive_statistics" in data
    assert data["jobs_used"] > 0
    # Check if SHAP/Global impact lists are present
    assert len(data["global_top_emerging_skills"]) >= 0

@patch("requests.post")
def test_train_emerging_classifier_logistic(mock_post):
    """Test the Logistic Regression path (uses coefficients instead of SHAP)."""
    
    mock_login = MagicMock()
    mock_login.text = '"token"'
    mock_jobs = MagicMock()
    mock_jobs.json.return_value = MOCK_JOBS_RESPONSE
    mock_skills = MagicMock()
    mock_skills.json.return_value = MOCK_SKILLS_RESPONSE
    
    mock_post.side_effect = [mock_login] + [mock_jobs]*8 + [mock_skills]*40

    response = client.post("/api/analysis/jobs_emergingdck_train?keywords=software&model_type=logistic")
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "logistic"

@patch("requests.post")
def test_no_jobs_found(mock_post):
    """Verify error handling when no jobs are returned by the API."""
    
    mock_login = MagicMock()
    mock_login.text = '"token"'
    mock_jobs = MagicMock()
    mock_jobs.json.return_value = {"items": []} # Empty list
    
    mock_post.side_effect = [mock_login, mock_jobs]

    response = client.post("/api/analysis/jobs_emergingdck_train?keywords=nonexistent")
    
    assert response.status_code == 200
    assert "error" in response.json()
    assert response.json()["error"] == "No jobs found"

def test_invalid_model_type():
    """Verify validation for model_type parameter."""
    # Note: We don't need to patch requests here because validation happens before API calls
    response = client.post("/api/analysis/jobs_emergingdck_train?keywords=ai&model_type=brain_waves")
    
    assert response.status_code == 200
    assert "error" in response.json()
    assert "Invalid model_type" in response.json()["error"]