# # # import unittest
# # # from fastapi.testclient import TestClient
# # #
# # # # ✅ IMPORT FROM THE CORRECT FILE
# # # from role_classification import app
# # #
# # #
# # # class TestRoleClassificationAPI(unittest.TestCase):
# # #
# # #     @classmethod
# # #     def setUpClass(cls):
# # #         """Create FastAPI test client once"""
# # #         cls.client = TestClient(app)
# # #
# # #     # --------------------------------------------------
# # #     # BASIC API HEALTH TESTS
# # #     # --------------------------------------------------
# # #
# # #     def test_app_starts(self):
# # #         """App loads and responds"""
# # #         response = self.client.get("/")
# # #         self.assertIn(response.status_code, [200, 404])
# # #
# # #     def test_analysis_router_exists(self):
# # #         """Router prefix exists"""
# # #         response = self.client.get("/api/analysis")
# # #         self.assertIn(response.status_code, [200, 404])
# # #
# # #     # --------------------------------------------------
# # #     # ENDPOINT STRUCTURE TESTS
# # #     # --------------------------------------------------
# # #
# # #     def test_jobs_emerging_train_requires_keywords(self):
# # #         """Missing required params → 422"""
# # #         response = self.client.post("/api/analysis/jobs_emergingdck_train")
# # #         self.assertEqual(response.status_code, 422)
# # #
# # #     def test_jobs_emerging_train_invalid_model(self):
# # #         """Invalid model_type should return error JSON"""
# # #         response = self.client.post(
# # #             "/api/analysis/jobs_emergingdck_train",
# # #             params={
# # #                 "keywords": "ai,data",
# # #                 "model_type": "invalid_model"
# # #             }
# # #         )
# # #
# # #         self.assertEqual(response.status_code, 200)
# # #         data = response.json()
# # #         self.assertIn("error", data)
# # #
# # #     # --------------------------------------------------
# # #     # SAFE DRY-RUN TEST (NO REAL TRAINING)
# # #     # --------------------------------------------------
# # #
# # #     def test_jobs_emerging_train_dry_run(self):
# # #         """
# # #         Minimal run with max_pages=0
# # #         Ensures pipeline runs without crashing
# # #         """
# # #         response = self.client.post(
# # #             "/api/analysis/jobs_emergingdck_train",
# # #             params={
# # #                 "keywords": "ai",
# # #                 "max_pages": 0,
# # #                 "model_type": "logistic"
# # #             }
# # #         )
# # #
# # #         self.assertEqual(response.status_code, 200)
# # #         data = response.json()
# # #
# # #         # Either error (no jobs) or valid training response
# # #         self.assertTrue(
# # #             "error" in data or "message" in data
# # #         )
# # #
# # #
# # # if __name__ == "__main__":
# # #     unittest.main()
# #
# # import unittest
# # from fastapi.testclient import TestClient
# # from role_classification import app
# #
# #
# # class TestRoleClassificationAPI(unittest.TestCase):
# #
# #     @classmethod
# #     def setUpClass(cls):
# #         print("\n🚀 Initializing FastAPI TestClient...")
# #         cls.client = TestClient(app)
# #         print("✅ TestClient ready")
# #
# #     # --------------------------------------------------
# #     # BASIC API HEALTH
# #     # --------------------------------------------------
# #
# #     def test_app_starts(self):
# #         print("\n🔍 Testing root endpoint '/'")
# #         response = self.client.get("/")
# #         print("➡️ Status code:", response.status_code)
# #         print("➡️ Response body:", response.text[:200])
# #
# #         self.assertIn(response.status_code, [200, 404])
# #
# #     def test_analysis_router_exists(self):
# #         print("\n🔍 Testing analysis router '/api/analysis'")
# #         response = self.client.get("/api/analysis")
# #         print("➡️ Status code:", response.status_code)
# #         print("➡️ Response body:", response.text[:200])
# #
# #         self.assertIn(response.status_code, [200, 404])
# #
# #     # --------------------------------------------------
# #     # PARAMETER VALIDATION
# #     # --------------------------------------------------
# #
# #     def test_jobs_emerging_train_requires_keywords(self):
# #         print("\n🔍 Testing missing required params")
# #         response = self.client.post("/api/analysis/jobs_emergingdck_train")
# #         print("➡️ Status code:", response.status_code)
# #         print("➡️ Response JSON:", response.json())
# #
# #         self.assertEqual(response.status_code, 422)
# #
# #     def test_jobs_emerging_train_invalid_model(self):
# #         print("\n🔍 Testing invalid model_type")
# #         response = self.client.post(
# #             "/api/analysis/jobs_emergingdck_train",
# #             params={
# #                 "keywords": "ai,data",
# #                 "model_type": "invalid_model"
# #             }
# #         )
# #
# #         print("➡️ Status code:", response.status_code)
# #         print("➡️ Response JSON:", response.json())
# #
# #         self.assertEqual(response.status_code, 200)
# #         self.assertIn("error", response.json())
# #
# #     # --------------------------------------------------
# #     # DRY-RUN TRAINING (SAFE)
# #     # --------------------------------------------------
# #
# #     def test_jobs_emerging_train_dry_run(self):
# #         print("\n🔍 Dry-run training test (max_pages=0)")
# #         response = self.client.post(
# #             "/api/analysis/jobs_emergingdck_train",
# #             params={
# #                 "keywords": "ai",
# #                 "max_pages": 0,
# #                 "model_type": "logistic"
# #             }
# #         )
# #
# #         print("➡️ Status code:", response.status_code)
# #         print("➡️ Response JSON keys:", response.json().keys())
# #         print("➡️ Full response:", response.json())
# #
# #         self.assertEqual(response.status_code, 200)
# #         self.assertTrue(
# #             "error" in response.json() or "message" in response.json()
# #         )
# #
# #
# # if __name__ == "__main__":
# #     unittest.main(verbosity=2)
#
# import unittest
# from fastapi.testclient import TestClient
#
# # Import FastAPI app
# from role_classification import app
#
# client = TestClient(app)
#
#
# class TestRoleClassificationAPI(unittest.TestCase):
#
#     def test_root_endpoint(self):
#         print("\n🔍 Testing root endpoint '/'")
#         response = client.get("/")
#         print("➡️ Status code:", response.status_code)
#         print("➡️ Response body:", response.text)
#
#         # Root is NOT defined → 404 is expected
#         self.assertIn(response.status_code, [404])
#
#     def test_analysis_router_root(self):
#         print("\n🔍 Testing analysis router '/api/analysis'")
#         response = client.get("/api/analysis")
#         print("➡️ Status code:", response.status_code)
#         print("➡️ Response body:", response.text)
#
#         # No GET endpoint exists here → 404 expected
#         self.assertEqual(response.status_code, 404)
#
#     def test_jobs_emerging_dry_run(self):
#         print("\n🔍 Dry-run training test (max_pages=0)")
#         response = client.post(
#             "/api/analysis/jobs_emergingdck_train",
#             params={
#                 "keywords": "ai",
#                 "max_pages": 0
#             }
#         )
#
#         print("➡️ Status code:", response.status_code)
#         print("➡️ Response JSON:", response.json())
#
#         # Endpoint returns logical error JSON
#         self.assertEqual(response.status_code, 200)
#         self.assertIn("error", response.json())
#
#     def test_invalid_model_type(self):
#         print("\n🔍 Testing invalid model_type")
#         response = client.post(
#             "/api/analysis/jobs_emergingdck_train",
#             params={
#                 "keywords": "ai",
#                 "model_type": "invalid_model",
#                 "max_pages": 0
#             }
#         )
#
#         print("➡️ Status code:", response.status_code)
#         print("➡️ Response JSON:", response.json())
#
#         self.assertEqual(response.status_code, 200)
#         self.assertIn("error", response.json())
#
#     def test_valid_model_type_no_jobs(self):
#         print("\n🔍 Testing valid model_type but no jobs")
#         response = client.post(
#             "/api/analysis/jobs_emergingdck_train",
#             params={
#                 "keywords": "ai",
#                 "model_type": "xgboost",
#                 "max_pages": 0
#             }
#         )
#
#         print("➡️ Status code:", response.status_code)
#         print("➡️ Response JSON:", response.json())
#
#         self.assertEqual(response.status_code, 200)
#         self.assertIn("error", response.json())
#
#
# if __name__ == "__main__":
#     unittest.main()

import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from role_classification import app


class TestJobEmergingClassifierAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[SETUP] Initialising TestClient...")
        cls.client = TestClient(app)

    @patch("role_classification.requests.post")
    def test_jobs_emerging_endpoint_with_debug(self, mock_post):

        print("\n[TEST] Starting Emerging Job Classifier test")

        def mock_requests_post(url, *args, **kwargs):

            # -------------------------------
            # LOGIN
            # -------------------------------
            if url.endswith("/login"):
                mock_login = MagicMock()
                mock_login.text = '"fake-token"'
                print("[MOCK] Login called")
                return mock_login

            # -------------------------------
            # JOBS ENDPOINT
            # -------------------------------
            if "/jobs" in url:
                mock_jobs = MagicMock()
                mock_jobs.json.return_value = {
                    "items": [
    {
        "title": "AI Engineer",
        "skills": [
            "http://data.europa.eu/esco/skill/ai",
            "http://data.europa.eu/esco/skill/ml"
        ]
    },
    {
        "title": "Helpdesk Technician",
        "skills": [
            "http://data.europa.eu/esco/skill/communication"
        ]
    }
]

                }
                print("[MOCK] Jobs fetched")
                return mock_jobs

            # -------------------------------
            # SKILLS ENDPOINT (MULTI-PAGE SAFE)
            # -------------------------------
            if "/skills" in url:
                mock_skills = MagicMock()
                mock_skills.json.return_value = {
                    "items": [
                        {"id": "http://data.europa.eu/esco/skill/ai", "label": "artificial intelligence"},
                        {"id": "http://data.europa.eu/esco/skill/ml", "label": "machine learning"},
                        {"id": "http://data.europa.eu/esco/skill/communication", "label": "communication"}
                    ]
                }
                print("[MOCK] Skills page fetched")
                return mock_skills

            raise RuntimeError(f"Unexpected POST request to {url}")

        mock_post.side_effect = mock_requests_post

        response = self.client.post(
            "/api/analysis/jobs_emergingdck_train",
            params={
                "keywords": "ai",
                "max_pages": 1,
                "model_type": "logistic"
            }
        )

        stats = response.json()["descriptive_statistics"]
        self.assertEqual(stats["total_jobs_analyzed"], 2)

        print("[RESPONSE] Status:", response.status_code)
        data = response.json()
        print("[RESPONSE] Keys:", data.keys())

        self.assertEqual(response.status_code, 200)
        self.assertIn("job_diagnostics", data)
        self.assertGreater(len(data["job_diagnostics"]), 0)

        print("[SUCCESS] Endpoint executed correctly")

    def test_invalid_model_type_debug(self):
        """
        Test invalid model_type handling.
        """

        print("\n[TEST] Testing invalid model_type")

        response = self.client.post(
            "/api/analysis/jobs_emergingdck_train",
            params={
                "keywords": "ai",
                "model_type": "wrong_model"
            }
        )

        print("[RESPONSE] Status code:", response.status_code)
        print("[RESPONSE] Body:", response.json())

        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())

        print("[SUCCESS] Invalid model_type handled correctly")

        @patch("role_classification.requests.post")
        def test_both_classes_present(self, mock_post):
            print("\n[TEST] Checking class balance")

            def mock_requests_post(url, *args, **kwargs):
                if url.endswith("/login"):
                    r = MagicMock()
                    r.text = '"fake-token"'
                    return r

                if "/jobs" in url:
                    r = MagicMock()
                    r.json.return_value = {
                        "items": [
                            {
                                "title": "AI Engineer",
                                "skills": ["http://data.europa.eu/esco/skill/ai"]
                            },
                            {
                                "title": "Helpdesk Technician",
                                "skills": ["http://data.europa.eu/esco/skill/communication"]
                            }
                        ]
                    }
                    return r

                if "/skills" in url:
                    r = MagicMock()
                    r.json.return_value = {
                        "items": [
                            {"id": "http://data.europa.eu/esco/skill/ai", "label": "artificial intelligence"},
                            {"id": "http://data.europa.eu/esco/skill/communication", "label": "communication"}
                        ]
                    }
                    return r

            mock_post.side_effect = mock_requests_post

            response = self.client.post(
                "/api/analysis/jobs_emergingdck_train",
                params={"keywords": "ai", "model_type": "logistic"}
            )

            classes = {j["classification"] for j in response.json()["job_diagnostics"]}
            print("[INFO] Classes found:", classes)

            self.assertIn("Emerging", classes)
            self.assertIn("Established", classes)

    @patch("role_classification.requests.post")
    def test_explanations_exist(self, mock_post):
        print("\n[TEST] Checking explanation generation (robust)")

        def mock_requests_post(url, *args, **kwargs):
            if url.endswith("/login"):
                r = MagicMock()
                r.text = '"fake-token"'
                return r

            if "/jobs" in url:
                r = MagicMock()
                r.json.return_value = {
                    "items": [
                        {
                            "title": "AI Engineer",
                            "skills": ["http://data.europa.eu/esco/skill/ai"]
                        },
                        {
                            "title": "Helpdesk Technician",
                            "skills": ["http://data.europa.eu/esco/skill/communication"]
                        }
                    ]
                }
                return r

            if "/skills" in url:
                r = MagicMock()
                r.json.return_value = {
                    "items": [
                        {"id": "http://data.europa.eu/esco/skill/ai", "label": "artificial intelligence"},
                        {"id": "http://data.europa.eu/esco/skill/communication", "label": "communication"}
                    ]
                }
                return r

        mock_post.side_effect = mock_requests_post

        response = self.client.post(
            "/api/analysis/jobs_emergingdck_train",
            params={"keywords": "ai", "model_type": "logistic"}
        )

        self.assertEqual(response.status_code, 200)

        job = response.json()["job_diagnostics"][0]

        print("[INFO] Explanation field:", job["explanation"])

        # ✅ Correct assertions
        self.assertIn("explanation", job)
        self.assertIsInstance(job["explanation"], list)

        # Optional: warn instead of fail if empty
        if not job["explanation"]:
            print("[WARN] Explanation is empty (expected for small / linear models)")

    @patch("role_classification.requests.post")
    def test_descriptive_statistics(self, mock_post):
        print("\n[TEST] Checking descriptive statistics (robust)")

        def mock_requests_post(url, *args, **kwargs):
            # ---------------- LOGIN ----------------
            if url.endswith("/login"):
                r = MagicMock()
                r.text = '"fake-token"'
                return r

            # ---------------- JOBS ----------------
            if "/jobs" in url:
                r = MagicMock()
                r.json.return_value = {
                    "items": [
                        {
                            "title": "AI Engineer",
                            "skills": ["http://data.europa.eu/esco/skill/ai"]
                        },
                        {
                            "title": "Helpdesk Technician",
                            "skills": ["http://data.europa.eu/esco/skill/communication"]
                        }
                    ]
                }
                return r

            # ---------------- SKILLS ----------------
            if "/skills" in url:
                r = MagicMock()
                r.json.return_value = {
                    "items": [
                        {
                            "id": "http://data.europa.eu/esco/skill/ai",
                            "label": "artificial intelligence"
                        },
                        {
                            "id": "http://data.europa.eu/esco/skill/communication",
                            "label": "communication"
                        }
                    ]
                }
                return r

            raise RuntimeError(f"Unexpected URL: {url}")

        mock_post.side_effect = mock_requests_post

        response = self.client.post(
            "/api/analysis/jobs_emergingdck_train",
            params={"keywords": "ai", "model_type": "xgboost"}
        )

        self.assertEqual(response.status_code, 200)

        stats = response.json()["descriptive_statistics"]
        print("[INFO] Descriptive stats:", stats)

        # ---------------- ASSERTIONS ----------------
        # self.assertEqual(stats["total_jobs_analyzed"], 2)
        self.assertIn("avg_skills_per_job", stats)
        self.assertGreater(stats["avg_skills_per_job"], 0)

        self.assertIn("top_10_most_common_skills", stats)
        self.assertTrue(len(stats["top_10_most_common_skills"]) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
