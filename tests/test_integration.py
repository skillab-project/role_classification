import pytest
import os
from fastapi.testclient import TestClient
from role_classification import app
from dotenv import load_dotenv

client = TestClient(app)
load_dotenv()


"""
Test: Verify that the credentials provided allow us to 
actually reach and authenticate with the Tracker API.
"""
def test_tracker_api_connectivity():
    # We check if env vars are present
    username = os.getenv("TRACKER_USERNAME")
    password = os.getenv("TRACKER_PASSWORD")
    
    if not username or not password:
        pytest.skip("Skipping integration test: Tracker credentials not found.")

    # Call the training endpoint with a simple keyword
    # We use a small max_pages=1 to keep the test fast
    response = client.post(
        "/api/analysis/jobs_emergingdck_train",
        params={
            "keywords": "software",
            "max_pages": 1,
            "model_type": "xgboost"
        }
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    if "error" in data:
        pytest.fail(f"API returned error: {data['error']}")
    
    assert "message" in data
    assert data["jobs_used"] > 0, "Integration test found 0 jobs. Check keywords or API status."
    assert "descriptive_statistics" in data
    print(f"Integration Success: Found {data['jobs_used']} real jobs.")


"""
Test: Verifies specifically that the skill lookup logic works
with the real ESCO mapping on the server.
"""
def test_tracker_skills_endpoint():
    response = client.post(
        "/api/analysis/jobs_emergingdck_train",
        params={
            "keywords": "ai",
            "max_pages": 1,
            "model_type": "xgboost"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    # Check if the skills dimension is populated (meaning it fetched skill labels)
    assert data["skills_dim"] > 0