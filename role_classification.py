from fastapi import FastAPI, APIRouter, Query
import pandas as pd
import numpy as np
import requests, os
from dotenv import load_dotenv
from datetime import datetime
import xgboost as xgb
import shap

# ------------------------------------------------------------
# INITIAL SETUP
# ------------------------------------------------------------
app = FastAPI(
    title="SkillLab Job Classifier API",
    root_path="/role-classification"
)

analysis_router = APIRouter(
    prefix="/api/analysis",
    tags=["SKILL Analysis"]
)

# Globals
JOB_MODEL = None
SKILL_INDEX = None
EXPLAINER = None

# @analysis_router.post("/jobs_emerging_train")
# def train_job_emerging_classifier(
#     keywords: str = Query(...),
#     max_pages: int = Query(8)
# ):
#     """
#     Trains an XGBoost model to classify jobs into Emerging vs Established.
#     Returns classification + SHAP impact values for EACH job.
#     """
#     global JOB_MODEL, SKILL_INDEX, EXPLAINER
#
#     load_dotenv()
#     API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
#
#     # Authenticate
#     res = requests.post(f"{API}/login", json={"username": USER, "password": PASS})
#     token = res.text.replace('"', "")
#     headers = {"Authorization": f"Bearer {token}"}
#
#     keywords_list = [k.strip() for k in keywords.split(",")]
#
#     # ------------------------------------------------------------
#     # Fetch jobs
#     # ------------------------------------------------------------
#     jobs = []
#     for page in range(1, max_pages + 1):
#         form = [
#             ("keywords_logic", "or"),
#             ("skill_ids_logic", "or"),
#             ("occupation_ids_logic", "or")
#         ]
#         for k in keywords_list:
#             form.append(("keywords", k))
#
#         res = requests.post(
#             f"{API}/jobs?page={page}&page_size=100",
#             headers=headers,
#             data=form
#         )
#         items = res.json().get("items", [])
#         if not items:
#             break
#         jobs.extend(items)
#
#     if not jobs:
#         return {"error": "No jobs found"}
#
#     # ------------------------------------------------------------
#     # Extract ESCO skill URIs
#     # ------------------------------------------------------------
#     skill_uris = []
#     job_titles = []
#
#     for job in jobs:
#         job_titles.append(job.get("title", "").lower())
#         for s in job.get("skills", []):
#             if isinstance(s, str) and s.startswith("http"):
#                 skill_uris.append(s)
#
#     unique_uris = sorted(set(skill_uris))
#
#     # ------------------------------------------------------------
#     # Map URI → Label
#     # ------------------------------------------------------------
#     all_esco = []
#     for page in range(1, 40):
#         r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers)
#         its = r.json().get("items", [])
#         if not its:
#             break
#         all_esco.extend(its)
#
#     id_to_label = {
#         s["id"]: s["label"].strip().lower()
#         for s in all_esco if "id" in s
#     }
#
#     # ------------------------------------------------------------
#     # GLOBAL SKILL RARITY FIX ✔
#     # ------------------------------------------------------------
#     all_skill_ids = [
#         s for job in jobs for s in job.get("skills", []) if s in id_to_label
#     ]
#     skill_global_counts = pd.Series(all_skill_ids).value_counts()
#
#     # ------------------------------------------------------------
#     # Build feature matrix
#     # ------------------------------------------------------------
#     job_skill_vectors = []
#     labels = []
#
#     SKILL_INDEX = sorted(set(id_to_label.get(u, u) for u in unique_uris))
#
#     for job in jobs:
#         title = job.get("title", "").lower()
#
#         job_skill_ids = [s for s in job.get("skills", []) if s in skill_global_counts]
#         job_skill_labels = [id_to_label[s] for s in job_skill_ids]
#
#         vec = [1 if skill in job_skill_labels else 0 for skill in SKILL_INDEX]
#         job_skill_vectors.append(vec)
#
#         # ------------------------------------------------------------
#         # ✔ FIXED AUTO-LABELING LOGIC
#         # ------------------------------------------------------------
#
#         # Rule 1: title keywords
#         if any(x in title for x in ["ai", "ml", "genai", "gpt", "llm", "cloud", "blockchain"]):
#             labels.append(1)
#
#         # Rule 2: global rarity of skills (correct version)
#         elif job_skill_ids and np.mean([skill_global_counts[s] for s in job_skill_ids]) < 3:
#             labels.append(1)
#
#         else:
#             labels.append(0)
#
#     X = np.array(job_skill_vectors)
#     y = np.array(labels)
#
#     # ------------------------------------------------------------
#     # Train XGBoost
#     # ------------------------------------------------------------
#     model = xgb.XGBClassifier(
#         n_estimators=120,
#         max_depth=5,
#         learning_rate=0.08,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         eval_metric="logloss"
#     )
#
#     model.fit(X, y)
#     JOB_MODEL = model
#
#     # SHAP
#     EXPLAINER = shap.TreeExplainer(model)
#
#     # ------------------------------------------------------------
#     # Diagnostics for each job
#     # ------------------------------------------------------------
#     results = []
#     probs = model.predict_proba(X)[:, 1]
#     shap_values = EXPLAINER.shap_values(X)
#
#     for i, job in enumerate(jobs):
#         title = job.get("title", "")
#         prob = float(probs[i])
#         label = "Emerging" if prob >= 0.5 else "Established"
#
#         shap_vec = shap_values[i]
#         idx_sorted = np.argsort(np.abs(shap_vec))[::-1][:5]
#
#         shap_expl = [
#             {"skill": SKILL_INDEX[idx], "impact": float(shap_vec[idx])}
#             for idx in idx_sorted
#         ]
#
#         results.append({
#             "job_title": title,
#             "classification": label,
#             "probability": round(prob, 4),
#             "explanation": shap_expl
#         })
#
#     return {
#         "message": "✅ Emerging job classifier trained.",
#         "jobs_used": len(jobs),
#         "skills_dim": len(SKILL_INDEX),
#         "positive_label_ratio": float(np.mean(y)),
#         "job_diagnostics": results
#     }

from fastapi import Query
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB


@analysis_router.post("/jobs_emergingdck_train")
def train_job_emerging_classifier(
    keywords: str = Query(...),
    max_pages: int = Query(8),
    model_type: str = Query("xgboost", description="Choose: xgboost, random_forest, logistic, svm, naive_bayes")
):
    """
    Trains a chosen ML model to classify jobs into Emerging vs Established.
    Options:
      - xgboost
      - random_forest
      - logistic
      - svm
    Returns classification + SHAP/coef explanations.
    """

    """
        Trains an XGBoost model to classify jobs into Emerging vs Established.
        Uses improved rule-based auto-labeling (titles + skills + rarity).
        Returns classification + SHAP impact values for EACH job.
        """
    global JOB_MODEL, SKILL_INDEX, EXPLAINER

    load_dotenv()
    API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USER = os.getenv("TRACKER_USERNAME", "")
    PASS = os.getenv("TRACKER_PASSWORD", "")

    # Authenticate
    res = requests.post(f"{API}/login", json={"username": USER, "password": PASS})
    token = res.text.replace('"', "")
    headers = {"Authorization": f"Bearer {token}"}

    keywords_list = [k.strip() for k in keywords.split(",")]

    # ------------------------------------------------------------
    # Fetch jobs
    # ------------------------------------------------------------
    jobs = []
    for page in range(1, max_pages + 1):
        form = [
            ("keywords_logic", "or"),
            ("skill_ids_logic", "or"),
            ("occupation_ids_logic", "or")
        ]
        for k in keywords_list:
            form.append(("keywords", k))

        res = requests.post(
            f"{API}/jobs?page={page}&page_size=100",
            headers=headers,
            data=form
        )
        items = res.json().get("items", [])
        if not items:
            break
        jobs.extend(items)

    if not jobs:
        return {"error": "No jobs found"}

    # ------------------------------------------------------------
    # Extract ESCO skill URIs
    # ------------------------------------------------------------
    skill_uris = []
    job_titles = []

    for job in jobs:
        job_titles.append(job.get("title", "").lower())
        for s in job.get("skills", []):
            if isinstance(s, str) and s.startswith("http"):
                skill_uris.append(s)

    unique_uris = sorted(set(skill_uris))

    # ------------------------------------------------------------
    # Map URI → Label
    # ------------------------------------------------------------
    all_esco = []
    for page in range(1, 40):
        r = requests.post(f"{API}/skills?page={page}&page_size=100", headers=headers)
        its = r.json().get("items", [])
        if not its:
            break
        all_esco.extend(its)

    id_to_label = {
        s["id"]: s["label"].strip().lower()
        for s in all_esco if "id" in s
    }

    # ------------------------------------------------------------
    # GLOBAL SKILL FREQUENCY
    # ------------------------------------------------------------
    all_skill_ids = [
        s for job in jobs for s in job.get("skills", []) if s in id_to_label
    ]
    skill_global_counts = pd.Series(all_skill_ids).value_counts()

    rare_threshold = len(jobs) * 0.15
    rare_skill_ids = skill_global_counts[skill_global_counts < rare_threshold].index

    SKILL_CATEGORIES = {
        "ai": [
            "ai", "ml", "learning", "neural", "vision", "nlp", "model",
            "predict", "classify", "inference", "cluster", "pattern"
        ],
        "cloud": [
            "cloud", "aws", "azure", "gcp", "docker", "kube", "kubernetes",
            "serverless", "virtual", "container", "deploy"
        ],
        "data": [
            "data", "etl", "clean", "analyse", "query", "sql", "warehouse",
            "pipeline", "extract", "load", "transform"
        ],
        "devops": [
            "devops", "ci", "cd", "git", "monitor", "config", "infra",
            "build", "deploy", "automate"
        ],
        "cyber": [
            "security", "cyber", "risk", "threat", "encrypt", "attack",
            "protect", "penetration", "firewall"
        ],
        "software": [
            "software", "code", "test", "debug", "frontend", "backend",
            "api", "design", "develop"
        ],
        "algorithms": [
            "algorithm", "math", "stats", "optim", "graph", "regress",
            "cluster", "logic"
        ],

        # -----------------------------------------
        # NEW DOMAINS YOU REQUESTED
        # -----------------------------------------

        "green": [
            "sustainable", "green", "co2", "carbon", "renewable",
            "energy", "solar", "wind", "hydrogen", "climate",
            "environment", "ecosystem", "emissions",  # Core sustainability
            "sustainable", "sustainability", "green", "environmental", "environment",
            "ecosystem", "ecology", "biodiversity", "conservation", "restoration",
            "circular economy", "recycling", "upcycling", "reuse", "resource efficiency",
            "sustainable materials", "eco-design", "lifecycle assessment",
            "agroecology", "soil health", "sustainable agriculture", "precision agriculture", "water conservation",
        ],

        "education": [
            "teaching", "learning theory", "pedagogy", "curriculum",
            "assessment", "educational", "instructional", "training"
        ],

        "society": [
            "social", "community", "governance", "policy", "ethics",
            "inclusion", "inequality", "participation", "democracy"
        ],

        "business": [
            "management", "leadership", "strategy", "marketing",
            "finance", "operations", "sales", "analytics"
        ],

        "health": [
            "medical", "health", "clinical", "patient",
            "diagnosis",
            "pharma",
            "biotech",
            "care",
            "hospital",
            "nursing",
            "therapy",
            "mental",
            "vaccine",
            "rehabilitation",
            "health services",
            "health management",
            "hospital",
            "medical devices",
            "bioinformatics",
            "genomics",
            "pathology",
            "immunology",
            "clinic",
            "ambulance",
            "emergency care",
            "primary care",
            "health policy"
        ],

        "manufacturing": [
            "industrial", "automation", "robotic", "manufacture",
            "production", "assembly", "lean", "quality control"
        ],
    }

    TITLE_KEYWORDS_EMERGING = [
        "ai", "machine learning", "ml", "deep learning",
        "data scientist", "genai", "gpt", "llm",
        "cloud", "aws", "azure", "gcp",
        "blockchain", "quantum", "cybersecurity",
        "robotics", "automation", "devops",
        "artificial intelligence development",
        "dall-e image generator",
        "artificial intelligence risk",
        "crewai",
        "artificial intelligence systems",
        "azure openai",
        "artificial general intelligence",
        "autogen",
        "artificial neural networks",
        "image captioning",
        "ai/ml inference",
        "image inpainting",
        "applications of artificial intelligence",
        "image super-resolution",
        "ai agents",
        "natural language generation (nlg)",
        "ai alignment",
        "large language modeling",
        "ai innovation",
        "language models",
        "ai research",
        "natural language understanding (nlu)",
        "ai safety",
        "natural language user interface",
        "attention mechanisms",
        "langchain",
        "adversarial machine learning",
        "langgraph",
        "agentic ai",
        "microsoft copilot",
        "agentic systems",
        "microsoft luis",
        "autoencoders",
        "prompt engineering",
        "association rule learning",
        "retrieval augmented generation",
        "activity recognition",
        "sentence transformers",
        "3d reconstruction",
        "operationalizing ai",
        "backpropagation",
        "supervised learning",
        "bagging techniques",
        "unsupervised learning",
        "bayesian belief networks",
        "transfer learning",
        "boltzmann machine",
        "zero shot learning",
        "classification and regression tree (cart)",
        "soft computing",
        "deeplearning4j",
        "sorting algorithm",
        "concept drift detection",
        "training datasets",
        "deep learning",
        "test datasets",
        "deep learning methods",
        "test retrieval systems",
        "deep reinforcement learning (drl)",
        "dlib (c++ library)",
        "computational intelligence",
        "topological data analysis (tda)",
        "convolutional neural networks",
        "swarm intelligence",
        "cognitive computing",
        "spiking neural networks",
        "collaborative filtering",
        "variational autoencoders",
        "ensemble methods",
        "sequence-to-sequence models (seq2seq)",
        "expectation maximization algorithm",
        "transformer (machine learning model)",
        "expert systems",
        "stable diffusion",
        "federated learning",
        "small language model",
        "few shot learning",
        "apache mahout",
        "gradient boosting",
        "apache mxnet",
        "gradient boosting machines (gbm)",
        "apache singa",
        "hidden markov model",
        "aforge",
        "incremental learning",
        "amazon forecast",
        "inference engine",
        "hyperparameter optimization",
        "chatgpt",
        "fuzzy set",
        "genetic algorithm",
        "genetic programming",
        "catboost (machine learning library)",
        "chainer (deep learning framework)",
        "cloud-native architecture",
        "edge computing",
        "internet of things (iot)",
        "digital twins",
        "full-stack observability",
        "kubernetes orchestration",
        "containerization",
        "serverless computing",
        "microservices architecture",
        "distributed systems",
        "blockchain architecture",
        "zero trust security",
        "cyber threat intelligence",
        "penetration testing",
        "cryptographic engineering",
        "5g network engineering",
        "wireless sensor networks",
        "autonomous systems",
        "robotic process automation (rpa)",
        "industrial automation",
        "energy-efficient computing",
        "sustainable computing",
        "green cloud optimization",
        "real-time data streaming",
        "event-driven architecture",
        "apache kafka",
        "data lake engineering",
        "data mesh",
        "extended reality (xr)",
        "augmented reality development"
    ]

    TITLE_KEYWORDS_ESTABLISHED = [
        "php", "oracle", "cobol", "mainframe",
        "crm", "helpdesk", "technician",
        "network administrator", "desktop support"
    ]

    EMERGING_SKILL_KEYWORDS = [
        "artificial intelligence development",
        "dall-e image generator",
        "artificial intelligence risk",
        "crewai",
        "artificial intelligence systems",
        "azure openai",
        "artificial general intelligence",
        "autogen",
        "artificial neural networks",
        "image captioning",
        "ai/ml inference",
        "image inpainting",
        "applications of artificial intelligence",
        "image super-resolution",
        "ai agents",
        "natural language generation (nlg)",
        "ai alignment",
        "large language modeling",
        "ai innovation",
        "language models",
        "ai research",
        "natural language understanding (nlu)",
        "ai safety",
        "natural language user interface",
        "attention mechanisms",
        "langchain",
        "adversarial machine learning",
        "langgraph",
        "agentic ai",
        "microsoft copilot",
        "agentic systems",
        "microsoft luis",
        "autoencoders",
        "prompt engineering",
        "association rule learning",
        "retrieval augmented generation",
        "activity recognition",
        "sentence transformers",
        "3d reconstruction",
        "operationalizing ai",
        "backpropagation",
        "supervised learning",
        "bagging techniques",
        "unsupervised learning",
        "bayesian belief networks",
        "transfer learning",
        "boltzmann machine",
        "zero shot learning",
        "classification and regression tree (cart)",
        "soft computing",
        "deeplearning4j",
        "sorting algorithm",
        "concept drift detection",
        "training datasets",
        "deep learning",
        "test datasets",
        "deep learning methods",
        "test retrieval systems",
        "deep reinforcement learning (drl)",
        "dlib (c++ library)",
        "computational intelligence",
        "topological data analysis (tda)",
        "convolutional neural networks",
        "swarm intelligence",
        "cognitive computing",
        "spiking neural networks",
        "collaborative filtering",
        "variational autoencoders",
        "ensemble methods",
        "sequence-to-sequence models (seq2seq)",
        "expectation maximization algorithm",
        "transformer (machine learning model)",
        "expert systems",
        "stable diffusion",
        "federated learning",
        "small language model",
        "few shot learning",
        "apache mahout",
        "gradient boosting",
        "apache mxnet",
        "gradient boosting machines (gbm)",
        "apache singa",
        "hidden markov model",
        "aforge",
        "incremental learning",
        "amazon forecast",
        "inference engine",
        "hyperparameter optimization",
        "chatgpt",
        "fuzzy set",
        "genetic algorithm",
        "genetic programming",
        "catboost (machine learning library)",
        "chainer (deep learning framework)",
        "cloud-native architecture",
        "edge computing",
        "internet of things (iot)",
        "digital twins",
        "full-stack observability",
        "kubernetes orchestration",
        "containerization",
        "serverless computing",
        "microservices architecture",
        "distributed systems",
        "blockchain architecture",
        "zero trust security",
        "cyber threat intelligence",
        "penetration testing",
        "cryptographic engineering",
        "5g network engineering",
        "wireless sensor networks",
        "autonomous systems",
        "robotic process automation (rpa)",
        "industrial automation",
        "energy-efficient computing",
        "sustainable computing",
        "green cloud optimization",
        "real-time data streaming",
        "event-driven architecture",
        "apache kafka",
        "data lake engineering",
        "data mesh",
        "extended reality (xr)",
        "augmented reality development"
    ]

    # ------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------
    job_skill_vectors = []
    labels = []

    SKILL_INDEX = sorted(set(id_to_label.get(u, u) for u in unique_uris))

    for job in jobs:
        title = job.get("title", "").lower()

        job_skill_ids = [s for s in job.get("skills", []) if s in id_to_label]
        job_skill_labels = [id_to_label[s] for s in job_skill_ids]

        # Binary vector
        vec = [1 if skill in job_skill_labels else 0 for skill in SKILL_INDEX]
        job_skill_vectors.append(vec)

        # ------------------------------------------------------------
        # NEW IMPROVED LABELING RULES
        # ------------------------------------------------------------

        # 1. Established override
        if any(kw in title for kw in TITLE_KEYWORDS_ESTABLISHED):
            labels.append(0)
            continue

        # 2. Title Emerging
        if any(kw in title for kw in TITLE_KEYWORDS_EMERGING):
            labels.append(1)
            continue

        # 3. Emerging skill keywords
        if any(any(em in skill for em in EMERGING_SKILL_KEYWORDS) for skill in job_skill_labels):
            labels.append(1)
            continue

        # 4. Rare skill IDs
        if any(s in rare_skill_ids for s in job_skill_ids):
            labels.append(1)
            continue

        # 5. Default → Established
        labels.append(0)

    # ===================================================================
    # 1. SELECT & TRAIN MODEL
    # ===================================================================
    X = np.array(job_skill_vectors)
    y = np.array(labels)

    model_type = model_type.lower()

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = shap.TreeExplainer(model)
        explanation_mode = "shap_tree"

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1
        )
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = shap.TreeExplainer(model)
        explanation_mode = "shap_tree"

    elif model_type == "logistic":
        model = LogisticRegression(
            max_iter=500,
            solver="liblinear"
        )
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = None
        explanation_mode = "linear_coef"

    elif model_type == "svm":
        model = LinearSVC()
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = None
        explanation_mode = "linear_coef"

    elif model_type == "naive_bayes":
        model = BernoulliNB()
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = None  # No SHAP for Naive Bayes
        explanation_mode = "linear_coef"

    else:
        return {"error": "Invalid model_type. Choose: xgboost, random_forest, logistic, svm, naive_bayes"}

    # ===================================================================
    # 2. GENERATE DIAGNOSTICS
    # ===================================================================
    results = []

    # Tree models (XGBoost / RF) handled separately later
    if model_type in ["xgboost", "random_forest"]:
        probs = model.predict_proba(X)[:, 1]
        shap_values = EXPLAINER.shap_values(X)

    else:
        # Preferred: use predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            # SVM fallback: convert decision function to probability
            decision = model.decision_function(X)
            probs = 1 / (1 + np.exp(-decision))

        shap_values = None

        # Global SHAP impact storage
    global_skill_impacts = {skill: 0.0 for skill in SKILL_INDEX}

    for i, job in enumerate(jobs):
        title = job.get("title", "")
        prob = float(probs[i])
        classification = "Emerging" if prob >= 0.5 else "Established"

        # ----------------------------------------------------
        # SHAP EXPLANATION (Tree Models)
        # ----------------------------------------------------
        # ----------------------------------------------------
        # SHAP EXPLANATION (Tree Models)
        # ----------------------------------------------------
        if explanation_mode == "shap_tree":
            shap_vec = shap_values[i]

            # --- Remove zero impact skills ---
            nonzero_indices = [
                idx for idx in range(len(shap_vec))
                if shap_vec[idx] != 0
            ]

            # Exclude unwanted recurring skills
            forbidden = {"adhere to ohsas 18001", "visual basic"}

            nonzero_indices = [
                idx for idx in nonzero_indices
                if SKILL_INDEX[idx].lower() not in forbidden
            ]

            # If nothing remains, skip explanation
            if nonzero_indices:
                idx_sorted = sorted(
                    nonzero_indices,
                    key=lambda idx: abs(shap_vec[idx]),
                    reverse=True
                )[:5]
            else:
                idx_sorted = []  # EMPTY → no explanation

            shap_expl = [
                {
                    "skill": SKILL_INDEX[idx],
                    "impact": float(shap_vec[idx])
                }
                for idx in idx_sorted
            ]

            # === Final Filtering of SHAP explanations ===

            def clean_explanation(shap_vec, skill_index, forbidden=None):
                if forbidden is None:
                    forbidden = set()

                cleaned = []
                for idx, impact in enumerate(shap_vec):
                    skill = skill_index[idx].lower()

                    # skip forbidden skills
                    if skill in forbidden:
                        continue

                    # skip zero or extremely tiny impacts
                    if impact == 0 or abs(impact) < 1e-12:
                        continue

                    cleaned.append({"skill": skill_index[idx], "impact": float(impact)})

                # sort by absolute impact
                cleaned = sorted(cleaned, key=lambda x: abs(x["impact"]), reverse=True)

                # return top 5
                return cleaned[:5]

            # Forbidden repeating noisy skills
            FORBIDDEN_SKILLS = {
                "adhere to ohsas 18001",
                "visual basic"
            }

            # Apply final cleaning
            shap_expl = clean_explanation(
                shap_vec,
                SKILL_INDEX,
                forbidden=FORBIDDEN_SKILLS
            )




        # ----------------------------------------------------
        # COEFFICIENT EXPLANATION (Linear Models)
        # ----------------------------------------------------
        # else:
        #     coefs = model.coef_.ravel()
        #     shap_vec = coefs * X[i]
        #
        #     idx_sorted = np.argsort(np.abs(shap_vec))[::-1][:5]
        #     shap_expl = [
        #         {"skill": SKILL_INDEX[idx], "impact": float(shap_vec[idx])}
        #         for idx in idx_sorted
        #     ]
        else:
            # LOGISTIC / LINEARSVC
            if hasattr(model, "coef_"):
                coefs = model.coef_.ravel()
                shap_vec = coefs * X[i]

            # NAIVE BAYES
            elif hasattr(model, "feature_log_prob_"):
                # difference between Emerging vs Established likelihoods
                class1 = model.feature_log_prob_[1]
                class0 = model.feature_log_prob_[0]
                shap_vec = (class1 - class0) * X[i]

            else:
                raise Exception(f"No explanation method available for model_type={model_type}")

            idx_sorted = np.argsort(np.abs(shap_vec))[::-1][:5]
            shap_expl = [
                {"skill": SKILL_INDEX[idx], "impact": float(shap_vec[idx])}
                for idx in idx_sorted
            ]

        # Accumulate global SHAP impacts
        for ex in shap_expl:
            global_skill_impacts[ex["skill"]] += ex["impact"]

        # --- New: Group positive vs negative ---
        positive_impacts = [
            s for s in shap_expl if s["impact"] > 0
        ]

        negative_impacts = [
            s for s in shap_expl if s["impact"] < 0
        ]

        top_positive = positive_impacts[:3]  # top 3 positive
        top_negative = negative_impacts[:3]  # top 3 negative

        # --- New: Emerging Score ---
        emerging_score = int(prob * 100)

        # --- Enhanced verdict ---
        verdict = (
            f"This job is {classification} with an Emerging score of {emerging_score}. "
            f"Top emerging signals: {[s['skill'] for s in top_positive]}. "
            f"Top established signals: {[s['skill'] for s in top_negative]}."
        )

        # TEXT SUMMARY
        most_pos = max(shap_expl, key=lambda x: x["impact"])
        most_neg = min(shap_expl, key=lambda x: x["impact"])

        text_summary = (
            f"Model used: {model_type}. "
            f"Job '{title}' classified as {classification} with probability {prob:.2f}. "
            f"Top positive indicator: {most_pos['skill']} ({most_pos['impact']:.3f}). "
            f"Top negative indicator: {most_neg['skill']} ({most_neg['impact']:.3f})."
        )

        # ----------------------------------------------------
        # RADAR PROFILE CALCULATION (CATEGORY-LEVEL IMPACT)
        # ----------------------------------------------------
        radar_profile = {cat: 0.0 for cat in SKILL_CATEGORIES}

        for s in shap_expl:
            skill_label = s["skill"]
            impact = s["impact"]

            for cat, keywords in SKILL_CATEGORIES.items():
                if any(k in skill_label for k in keywords):
                    # radar_profile[cat] += impact
                    radar_profile[cat] += abs(impact)

        # Normalize radar values (0–100 scale)
        max_val = max(abs(v) for v in radar_profile.values()) or 1
        radar_profile_normalized = {
            cat: int((v / max_val) * 100) for cat, v in radar_profile.items()
        }

        # Sort global emerging (positive) and established (negative) skills
        global_positive = sorted(
            [(skill, imp) for skill, imp in global_skill_impacts.items() if imp > 0],
            key=lambda x: x[1],
            reverse=True
        )[:15]

        global_negative = sorted(
            [(skill, imp) for skill, imp in global_skill_impacts.items() if imp < 0],
            key=lambda x: x[1]
        )[:15]  # lowest (most negative) first

        # Convert to JSON-friendly format
        global_top_emerging = [
            {"skill": skill, "global_impact": float(imp)}
            for skill, imp in global_positive
        ]

        global_top_established = [
            {"skill": skill, "global_impact": float(imp)}
            for skill, imp in global_negative
        ]

        # ----------------------------------------------------
        # EMOJI BADGE FOR EMERGING LEVEL
        # ----------------------------------------------------
        if emerging_score >= 90:
            emerging_badge = "🔥🔥🔥 Ultra Emerging"
        elif emerging_score >= 70:
            emerging_badge = "🔥🔥 Highly Emerging"
        elif emerging_score >= 50:
            emerging_badge = "🔥 Emerging"
        elif emerging_score >= 30:
            emerging_badge = "⚪ Neutral"
        else:
            emerging_badge = "📘 Established"

        # ----------------------------------------------------
        # CATEGORY BADGES (based on radar > 50)
        # ----------------------------------------------------
        category_badges = []
        if radar_profile_normalized.get("ai", 0) > 50: category_badges.append("🤖")
        if radar_profile_normalized.get("cloud", 0) > 50: category_badges.append("☁️")
        if radar_profile_normalized.get("data", 0) > 50: category_badges.append("📊")
        if radar_profile_normalized.get("devops", 0) > 50: category_badges.append("⚙️")
        if radar_profile_normalized.get("cyber", 0) > 50: category_badges.append("🔐")
        if radar_profile_normalized.get("software", 0) > 50: category_badges.append("🧰")
        if radar_profile_normalized.get("algorithms", 0) > 50: category_badges.append("📐")
        if radar_profile_normalized.get("green", 0) > 50: category_badges.append("🌿")
        if radar_profile_normalized.get("education", 0) > 50: category_badges.append("📘")
        if radar_profile_normalized.get("society", 0) > 50: category_badges.append("🤝")
        if radar_profile_normalized.get("business", 0) > 50: category_badges.append("💼")
        if radar_profile_normalized.get("health", 0) > 50: category_badges.append("🩺")
        if radar_profile_normalized.get("manufacturing", 0) > 50: category_badges.append("🏭")

        if not category_badges:
            category_badges = ["⚪"]  # neutral

        # ============================================================
        # 4. DESCRIPTIVE STATISTICS
        # ============================================================

        # Count classifications
        num_jobs = len(jobs)
        num_emerging = int(np.sum(y))
        num_established = num_jobs - num_emerging

        pct_emerging = round((num_emerging / num_jobs) * 100, 2)
        pct_established = round((num_established / num_jobs) * 100, 2)

        # Average number of skills per job
        avg_skills = np.mean([sum(vec) for vec in X])
        avg_skills = round(float(avg_skills), 2)

        # Top 10 most common skills in this sample
        skill_frequency = pd.Series(
            [s for job in jobs for s in job.get("skills", []) if s in id_to_label]
        ).value_counts().head(10)

        top_10_skills = [
            {"skill": id_to_label.get(skill, skill), "count": int(count)}
            for skill, count in skill_frequency.items()
        ]

        descriptive_stats = {
            "total_jobs_analyzed": num_jobs,
            "num_emerging": num_emerging,
            "num_established": num_established,
            "pct_emerging": pct_emerging,
            "pct_established": pct_established,
            "avg_skills_per_job": avg_skills,
            "top_10_most_common_skills": top_10_skills
        }

        results.append({
            "job_title": title,
            "classification": classification,
            "probability": round(prob, 4),
            "emerging_score": emerging_score,
            "top_emerging_drivers": top_positive,
            "top_established_drivers": top_negative,
            "explanation": shap_expl,
            "radar_profile": radar_profile_normalized,
            "summary": verdict,
            "emerging_badge": emerging_badge,
            "category_badges": category_badges,

        })

    # ===================================================================
    # 3. RETURN
    # ===================================================================
    return {
        "message": f"✅ {model_type.upper()} job classifier trained.",
        "model_type": model_type,
        "jobs_used": len(jobs),
        "skills_dim": len(SKILL_INDEX),
        "positive_label_ratio": float(np.mean(y)),
        "descriptive_statistics": descriptive_stats,
        "global_top_emerging_skills": global_top_emerging,
        "global_top_established_skills": global_top_established

    }


# ------------------------------------------------------------
# REGISTER ROUTER
# ------------------------------------------------------------
app.include_router(analysis_router)

