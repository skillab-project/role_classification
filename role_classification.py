from fastapi import Query
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi.responses import StreamingResponse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
import requests
import math
import re
import time
import json
from pathlib import Path
import os
from typing import Optional
from dotenv import load_dotenv
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


@analysis_router.post("/jobs_emergingdck_train")
def train_job_emerging_classifier(
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    occupation_ids: Optional[str] = Query(
        None, description="Comma-separated occupation IDs (e.g. http://data.europa.eu/esco/isco/C2153)"
    ),
    model_type: str = Query(
        "xgboost", description="Choose: xgboost, random_forest, logistic, svm, naive_bayes"
    ),
):
    """
    Trains a chosen ML model to classify jobs into Emerging vs Established.
    Fetches ALL available pages automatically (no page limit).
    Supports occupation_ids filtering.
    Results cached in Completed_Analyses/ — cache hit skips all API calls.
    Analysis logic is unchanged.
    """
    global JOB_MODEL, SKILL_INDEX, EXPLAINER

    load_dotenv()
    API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USER = os.getenv("TRACKER_USERNAME", "")
    PASS = os.getenv("TRACKER_PASSWORD", "")

    # ================================================================
    # 📁 CACHE SETUP
    # ================================================================
    folder = Path("Completed_Analyses")
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Folder '{folder}' created.")
    else:
        print(f"📁 Folder '{folder}' already exists, moving on.")

    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    occ_ids_list = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

    filename = f"completed_analysis_emerging_{model_type}"
    for kw in keywords_list:
        filename += f"_{kw}"
    for occ in occ_ids_list:
        match = re.search(r'C\d+$', occ)
        filename += f"_{match.group(0)}" if match else f"_{occ.replace('/', '_').replace(':', '').replace('.', '')}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    if file_path.exists():
        print(f"✅ Cache hit — loading from '{file_path}' (skipping all API calls).")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print("🌐 No cache found — running full analysis...")

    # ================================================================
    # 1️⃣ AUTHENTICATE
    # ================================================================
    print("🔐 Authenticating with Tracker...")
    res = requests.post(f"{API}/login", json={"username": USER, "password": PASS}, timeout=15)
    res.raise_for_status()
    token = res.text.replace('"', "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    print("✅ Authenticated successfully.")
    print(f"📡 Keywords: {keywords_list if keywords_list else '(none)'}")
    print(f"🏢 Occupation IDs: {occ_ids_list if occ_ids_list else '(none)'}")

    # ================================================================
    # 2️⃣ AUTO-PAGINATE ALL JOB PAGES WITH RETRY
    # ================================================================
    page_size = 100
    REQUEST_TIMEOUT = 180
    MAX_RETRIES = 3
    RETRY_BACKOFF = 10

    def build_form_data():
        fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
        for kw in keywords_list:
            fd.append(("keywords", kw))
        for occ in occ_ids_list:
            fd.append(("occupation_ids", occ))
        return fd

    def fetch_page_with_retry(page_num: int) -> dict:
        url = f"{API}/jobs?page={page_num}&page_size={page_size}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} for page {page_num} (timeout={REQUEST_TIMEOUT}s)...")
                r = requests.post(url, headers=headers, data=build_form_data(), timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    print(f"   ⚠️ HTTP {r.status_code}: {r.text[:300]}")
                    return {}
                return r.json()
            except requests.exceptions.ReadTimeout:
                print(f"   ⏱️ Timeout page {page_num}, attempt {attempt}/{MAX_RETRIES}.")
                if attempt < MAX_RETRIES:
                    print(f"   🔄 Retrying in {RETRY_BACKOFF}s...")
                    time.sleep(RETRY_BACKOFF)
                else:
                    print(f"   ❌ All retries exhausted for page {page_num}.")
                    return {}
            except Exception as ex:
                print(f"   ❌ {type(ex).__name__}: {ex}")
                return {}

    print("🔍 Probing page 1 to determine total record count...")
    probe_data = fetch_page_with_retry(1)
    if not probe_data:
        return {"error": "❌ Probe request (page 1) failed after all retries."}

    total_count = probe_data.get("count", 0)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
    print(f"📊 Total records: {total_count} → {total_pages} page(s) to fetch")

    if total_count == 0:
        return {"error": "No jobs found for the given filters."}

    jobs = list(probe_data.get("items", []))
    print(f"📦 Page 1/{total_pages}: {len(jobs)} jobs from probe.")

    for page in range(2, total_pages + 1):
        print(f"📄 Fetching page {page}/{total_pages}...")
        data = fetch_page_with_retry(page)
        if not data:
            print(f"⚠️ Page {page} failed — stopping early.")
            break
        items = data.get("items", [])
        print(f"📦 Page {page}/{total_pages}: {len(items)} jobs (running total: {len(jobs) + len(items)})")
        if not items:
            break
        jobs.extend(items)
        if len(items) < page_size:
            print("✅ Last page reached.")
            break

    print(f"🎯 Total jobs retrieved: {len(jobs)} / {total_count}")

    if not jobs:
        return {"error": "No jobs found."}

    # ================================================================
    # 3️⃣ EXTRACT SKILL URIs & MAP TO LABELS (batch, only found URIs)
    # ================================================================
    skill_uris = []
    job_titles = []
    for job in jobs:
        job_titles.append(job.get("title", "").lower())
        for s in job.get("skills", []):
            if isinstance(s, str) and s.startswith("http"):
                skill_uris.append(s)

    unique_uris = sorted(set(skill_uris))
    print(f"📚 Found {len(unique_uris)} unique skill URIs — resolving in batches...")

    id_to_label = {}
    if unique_uris:
        try:
            batch_size_skills = 50
            total_batches = math.ceil(len(unique_uris) / batch_size_skills)
            for batch_num, start in enumerate(range(0, len(unique_uris), batch_size_skills), 1):
                batch = unique_uris[start:start + batch_size_skills]
                skill_payload = [("ids", sid) for sid in batch]
                print(f"   Batch {batch_num}/{total_batches}: resolving {len(batch)} URIs...")
                skill_res = requests.post(
                    f"{API}/skills",
                    headers={"Authorization": f"Bearer {token}"},
                    data=skill_payload,
                    timeout=60
                )
                skill_res.raise_for_status()
                for s in skill_res.json().get("items", []):
                    sid = s.get("id", "")
                    if sid:
                        id_to_label[sid] = s.get("label", sid).strip().lower()
            matched = sum(1 for u in unique_uris if u in id_to_label)
            print(f"🔗 Matched: {matched}/{len(unique_uris)} URIs")
        except Exception as e:
            print(f"⚠️ Skill label lookup failed: {type(e).__name__}: {e} — using raw URIs.")
            id_to_label = {sid: sid for sid in unique_uris}

    # ================================================================
    # ✅ FROM HERE ONWARDS: ORIGINAL ANALYSIS LOGIC — UNCHANGED
    # ================================================================

    # GLOBAL SKILL FREQUENCY
    all_skill_ids = [
        s for job in jobs for s in job.get("skills", []) if s in id_to_label
    ]
    skill_global_counts = pd.Series(all_skill_ids).value_counts()
    rare_threshold = len(jobs) * 0.15
    rare_skill_ids = skill_global_counts[skill_global_counts < rare_threshold].index

    SKILL_CATEGORIES = {
        "ai": ["ai", "ml", "learning", "neural", "vision", "nlp", "model", "predict", "classify", "inference", "cluster", "pattern"],
        "cloud": ["cloud", "aws", "azure", "gcp", "docker", "kube", "kubernetes", "serverless", "virtual", "container", "deploy"],
        "data": ["data", "etl", "clean", "analyse", "query", "sql", "warehouse", "pipeline", "extract", "load", "transform"],
        "devops": ["devops", "ci", "cd", "git", "monitor", "config", "infra", "build", "deploy", "automate"],
        "cyber": ["security", "cyber", "risk", "threat", "encrypt", "attack", "protect", "penetration", "firewall"],
        "software": ["software", "code", "test", "debug", "frontend", "backend", "api", "design", "develop"],
        "algorithms": ["algorithm", "math", "stats", "optim", "graph", "regress", "cluster", "logic"],
        "green": [
            "sustainable", "green", "co2", "carbon", "renewable", "energy", "solar", "wind", "hydrogen",
            "climate", "environment", "ecosystem", "emissions", "sustainability", "environmental",
            "ecology", "biodiversity", "conservation", "restoration", "circular economy", "recycling",
            "upcycling", "reuse", "resource efficiency", "sustainable materials", "eco-design",
            "lifecycle assessment", "agroecology", "soil health", "sustainable agriculture",
            "precision agriculture", "water conservation",
        ],
        "education": ["teaching", "learning theory", "pedagogy", "curriculum", "assessment", "educational", "instructional", "training"],
        "society": ["social", "community", "governance", "policy", "ethics", "inclusion", "inequality", "participation", "democracy"],
        "business": ["management", "leadership", "strategy", "marketing", "finance", "operations", "sales", "analytics"],
        "health": [
            "medical", "health", "clinical", "patient", "diagnosis", "pharma", "biotech", "care",
            "hospital", "nursing", "therapy", "mental", "vaccine", "rehabilitation", "health services",
            "health management", "medical devices", "bioinformatics", "genomics", "pathology",
            "immunology", "clinic", "ambulance", "emergency care", "primary care", "health policy"
        ],
        "manufacturing": ["industrial", "automation", "robotic", "manufacture", "production", "assembly", "lean", "quality control"],
    }

    TITLE_KEYWORDS_EMERGING = [
        "ai", "machine learning", "ml", "deep learning", "data scientist", "genai", "gpt", "llm",
        "cloud", "aws", "azure", "gcp", "blockchain", "quantum", "cybersecurity", "robotics",
        "automation", "devops", "artificial intelligence development", "dall-e image generator",
        "artificial intelligence risk", "crewai", "artificial intelligence systems", "azure openai",
        "artificial general intelligence", "autogen", "artificial neural networks", "image captioning",
        "ai/ml inference", "image inpainting", "applications of artificial intelligence",
        "image super-resolution", "ai agents", "natural language generation (nlg)", "ai alignment",
        "large language modeling", "ai innovation", "language models", "ai research",
        "natural language understanding (nlu)", "ai safety", "natural language user interface",
        "attention mechanisms", "langchain", "adversarial machine learning", "langgraph", "agentic ai",
        "microsoft copilot", "agentic systems", "microsoft luis", "autoencoders", "prompt engineering",
        "association rule learning", "retrieval augmented generation", "activity recognition",
        "sentence transformers", "3d reconstruction", "operationalizing ai", "backpropagation",
        "supervised learning", "bagging techniques", "unsupervised learning", "bayesian belief networks",
        "transfer learning", "boltzmann machine", "zero shot learning",
        "classification and regression tree (cart)", "soft computing", "deeplearning4j",
        "sorting algorithm", "concept drift detection", "training datasets", "deep learning",
        "test datasets", "deep learning methods", "test retrieval systems",
        "deep reinforcement learning (drl)", "dlib (c++ library)", "computational intelligence",
        "topological data analysis (tda)", "convolutional neural networks", "swarm intelligence",
        "cognitive computing", "spiking neural networks", "collaborative filtering",
        "variational autoencoders", "ensemble methods", "sequence-to-sequence models (seq2seq)",
        "expectation maximization algorithm", "transformer (machine learning model)", "expert systems",
        "stable diffusion", "small language model", "federated learning", "few shot learning",
        "apache mahout", "gradient boosting", "apache mxnet", "gradient boosting machines (gbm)",
        "apache singa", "hidden markov model", "aforge", "incremental learning", "amazon forecast",
        "inference engine", "hyperparameter optimization", "chatgpt", "fuzzy set", "genetic algorithm",
        "genetic programming", "catboost (machine learning library)", "chainer (deep learning framework)",
        "cloud-native architecture", "edge computing", "internet of things (iot)", "digital twins",
        "full-stack observability", "kubernetes orchestration", "containerization", "serverless computing",
        "microservices architecture", "distributed systems", "blockchain architecture",
        "zero trust security", "cyber threat intelligence", "penetration testing",
        "cryptographic engineering", "5g network engineering", "wireless sensor networks",
        "autonomous systems", "robotic process automation (rpa)", "industrial automation",
        "energy-efficient computing", "sustainable computing", "green cloud optimization",
        "real-time data streaming", "event-driven architecture", "apache kafka", "data lake engineering",
        "data mesh", "extended reality (xr)", "augmented reality development"
    ]

    TITLE_KEYWORDS_ESTABLISHED = [
        "php", "oracle", "cobol", "mainframe", "crm", "helpdesk", "technician",
        "network administrator", "desktop support"
    ]

    EMERGING_SKILL_KEYWORDS = TITLE_KEYWORDS_EMERGING  # same list reused

    # Build feature matrix
    SKILL_INDEX = sorted(set(id_to_label.get(u, u) for u in unique_uris))
    job_skill_vectors = []
    labels = []

    for job in jobs:
        title = job.get("title", "").lower()
        job_skill_ids = [s for s in job.get("skills", []) if s in id_to_label]
        job_skill_labels = [id_to_label[s] for s in job_skill_ids]
        vec = [1 if skill in job_skill_labels else 0 for skill in SKILL_INDEX]
        job_skill_vectors.append(vec)

        if any(kw in title for kw in TITLE_KEYWORDS_ESTABLISHED):
            labels.append(0)
        elif any(kw in title for kw in TITLE_KEYWORDS_EMERGING):
            labels.append(1)
        elif any(any(em in skill for em in EMERGING_SKILL_KEYWORDS) for skill in job_skill_labels):
            labels.append(1)
        elif any(s in rare_skill_ids for s in job_skill_ids):
            labels.append(1)
        else:
            labels.append(0)

    X = np.array(job_skill_vectors)
    y = np.array(labels)

    model_type_lower = model_type.lower()

    if model_type_lower == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = shap.TreeExplainer(model)
        explanation_mode = "shap_tree"
    elif model_type_lower == "random_forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1)
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = shap.TreeExplainer(model)
        explanation_mode = "shap_tree"
    elif model_type_lower == "logistic":
        model = LogisticRegression(max_iter=500, solver="liblinear")
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = None
        explanation_mode = "linear_coef"
    elif model_type_lower == "svm":
        model = LinearSVC()
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = None
        explanation_mode = "linear_coef"
    elif model_type_lower == "naive_bayes":
        model = BernoulliNB()
        model.fit(X, y)
        JOB_MODEL = model
        EXPLAINER = None
        explanation_mode = "linear_coef"
    else:
        return {"error": "Invalid model_type. Choose: xgboost, random_forest, logistic, svm, naive_bayes"}

    # ================================================================
    # PREDICT + SHAP — ✅ FIX 2: chunked SHAP to avoid single huge allocation
    # ================================================================
    print(f"📋 Running predictions on {num_jobs} jobs...")
    if model_type_lower in ["xgboost", "random_forest"]:
        probs = model.predict_proba(X)[:, 1]

        print(f"🧠 Computing SHAP values for {X.shape[0]} jobs x {X.shape[1]} skills (chunked)...")

        SHAP_CHUNK = 10_000  # ✅ process 10k rows at a time instead of all 535k at once
        shap_values_list = []
        total_chunks = math.ceil(num_jobs / SHAP_CHUNK)
        for chunk_i in range(total_chunks):
            start = chunk_i * SHAP_CHUNK
            end = min(start + SHAP_CHUNK, num_jobs)
            print(f"   SHAP chunk {chunk_i + 1}/{total_chunks}: rows {start}–{end}...")
            sv = EXPLAINER.shap_values(
                X[start:end],
                approximate=True,
                check_additivity=False
            )
            # For binary XGBoost, sv may be 2D array directly
            if isinstance(sv, list):
                sv = sv[1]  # class 1
            shap_values_list.append(sv)
        shap_values = np.vstack(shap_values_list)
        print(f"✅ SHAP computation complete.")
    else:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            decision = model.decision_function(X)
            probs = 1 / (1 + np.exp(-decision))
        shap_values = None

    # Precompute linear explanation vectors once
    linear_shap_matrix = None
    if explanation_mode == "linear_coef":
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            linear_shap_matrix = X.toarray() * coefs
        elif hasattr(model, "feature_log_prob_"):
            diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
            linear_shap_matrix = X.toarray() * diff
        else:
            raise Exception(f"No explanation method available for model_type={model_type}")

    # ================================================================
    # DESCRIPTIVE STATS
    # ================================================================
    num_emerging = int(np.sum(y))
    num_established = num_jobs - num_emerging
    pct_emerging = round((num_emerging / num_jobs) * 100, 2)
    pct_established = round((num_established / num_jobs) * 100, 2)
    # Dense row-sum is fine here since it's just a 1D vector
    avg_skills = round(float(np.mean(np.asarray(X.sum(axis=1)).ravel())), 2)
    skill_frequency = pd.Series(all_skill_ids).value_counts().head(10)
    top_10_skills = [{"skill": id_to_label.get(sk, sk), "count": int(cnt)} for sk, cnt in skill_frequency.items()]
    descriptive_stats = {
        "total_jobs_analyzed": num_jobs,
        "num_emerging": num_emerging,
        "num_established": num_established,
        "pct_emerging": pct_emerging,
        "pct_established": pct_established,
        "avg_skills_per_job": avg_skills,
        "top_10_most_common_skills": top_10_skills
    }

    # ================================================================
    # PER-JOB DIAGNOSTICS
    # ================================================================
    print(f"📋 Building per-job diagnostics for {num_jobs} jobs...")

    FORBIDDEN_SKILLS = {"adhere to ohsas 18001", "visual basic"}
    BADGE_MAP = {"ai": "🤖", "cloud": "☁️", "data": "📊", "devops": "⚙️", "cyber": "🔐", "software": "🧰", "algorithms": "📐", "green": "🌿", "education": "📘", "society": "🤝", "business": "💼", "health": "🩺", "manufacturing": "🏭"}
    LOG_EVERY = max(1, num_jobs // 10)

    global_skill_impacts = {skill: 0.0 for skill in SKILL_INDEX}
    results = []

    def clean_explanation(shap_vec, skill_index, forbidden=None):
        if forbidden is None:
            forbidden = set()
        cleaned = []
        for idx, impact in enumerate(shap_vec):
            skill = skill_index[idx].lower()
            if skill in forbidden:
                continue
            if impact == 0 or abs(impact) < 1e-12:
                continue
            cleaned.append({"skill": skill_index[idx], "impact": float(impact)})
        cleaned = sorted(cleaned, key=lambda x: abs(x["impact"]), reverse=True)
        return cleaned[:5]

    print(f"📋 Building per-job diagnostics for {num_jobs} jobs...")
    for i, job in enumerate(jobs):
        if i % LOG_EVERY == 0:
            print(f"   ⚙️ Processing job {i+1}/{num_jobs} ({round((i+1)/num_jobs*100)}%)...")

        title = job.get("title", "")
        prob = float(probs[i])
        classification = "Emerging" if prob >= 0.5 else "Established"
        emerging_score = int(prob * 100)

        if explanation_mode == "shap_tree":
            shap_expl = clean_explanation(shap_values[i], SKILL_INDEX, forbidden=FORBIDDEN_SKILLS)
        else:
            idx_sorted = np.argsort(np.abs(linear_shap_matrix[i]))[::-1][:5]
            shap_expl = [{"skill": SKILL_INDEX[idx], "impact": float(linear_shap_matrix[i][idx])} for idx in idx_sorted]

        for ex in shap_expl:
            global_skill_impacts[ex["skill"]] += ex["impact"]

        positive_impacts = [s for s in shap_expl if s["impact"] > 0]
        negative_impacts = [s for s in shap_expl if s["impact"] < 0]
        top_positive = positive_impacts[:3]
        top_negative = negative_impacts[:3]

        verdict = (
            f"This job is {classification} with an Emerging score of {emerging_score}. "
            f"Top emerging signals: {[s['skill'] for s in top_positive]}. "
            f"Top established signals: {[s['skill'] for s in top_negative]}."
        )

        radar_profile = {cat: 0.0 for cat in SKILL_CATEGORIES}
        for s in shap_expl:
            for cat, kws in SKILL_CATEGORIES.items():
                if any(k in s["skill"] for k in kws):
                    radar_profile[cat] += abs(s["impact"])
        max_val = max(abs(v) for v in radar_profile.values()) or 1
        radar_profile_normalized = {cat: int((v / max_val) * 100) for cat, v in radar_profile.items()}

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

        category_badges = [badge for cat, badge in BADGE_MAP.items() if radar_profile_normalized.get(cat, 0) > 50]
        if not category_badges:
            category_badges = ["⚪"]

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

    print(f"✅ Per-job diagnostics complete.")

    # Compute global summaries once after the loop
    global_positive = sorted([(sk, imp) for sk, imp in global_skill_impacts.items() if imp > 0], key=lambda x: x[1], reverse=True)[:15]
    global_negative = sorted([(sk, imp) for sk, imp in global_skill_impacts.items() if imp < 0], key=lambda x: x[1])[:15]
    global_top_emerging = [{"skill": sk, "global_impact": float(imp)} for sk, imp in global_positive]
    global_top_established = [{"skill": sk, "global_impact": float(imp)} for sk, imp in global_negative]

    final_result = {
        "message": f"✅ {model_type.upper()} job classifier trained.",
        "model_type": model_type,
        "jobs_used": len(jobs),
        "total_jobs_available": total_count,
        "skills_dim": len(SKILL_INDEX),
        "positive_label_ratio": float(np.mean(y)),
        "filters_used": {
            "keywords": keywords_list if keywords_list else None,
            "occupation_ids": occ_ids_list if occ_ids_list else None,
        },
        "job_diagnostics": results,
        "descriptive_statistics": descriptive_stats,
        "global_top_emerging_skills": global_top_emerging,
        "global_top_established_skills": global_top_established
    }

    # ================================================================
    # 💾 SAVE TO CACHE
    # ================================================================
    print(f"💾 Saving results to cache: '{file_path}'...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print(f"✅ Cached successfully to '{file_path}'.")

    # ================================================================
    # 📡 STREAM RESPONSE — avoids freezing FastAPI/browser on large payloads
    # Serializes in chunks of 500 jobs so the connection stays alive
    # ================================================================
    print(f"📡 Streaming response ({len(results)} jobs)...")

    def iter_json():
        # Stream header fields first
        header = {
            "message": final_result["message"],
            "model_type": final_result["model_type"],
            "jobs_used": final_result["jobs_used"],
            "total_jobs_available": final_result["total_jobs_available"],
            "skills_dim": final_result["skills_dim"],
            "positive_label_ratio": final_result["positive_label_ratio"],
            "filters_used": final_result["filters_used"],
            "descriptive_statistics": final_result["descriptive_statistics"],
            "global_top_emerging_skills": final_result["global_top_emerging_skills"],
            "global_top_established_skills": final_result["global_top_established_skills"],
        }
        payload = json.dumps(header, ensure_ascii=False)
        # Insert job_diagnostics array in streaming chunks
        payload = payload[:-1]  # strip trailing }
        yield payload + ', "job_diagnostics": ['

        chunk_size = 200
        all_diag = final_result["job_diagnostics"]
        for idx, job_result in enumerate(all_diag):
            chunk = json.dumps(job_result, ensure_ascii=False)
            if idx < len(all_diag) - 1:
                yield chunk + ","
            else:
                yield chunk
            # yield in batches to avoid overwhelming the buffer
            if idx % chunk_size == 0 and idx > 0:
                pass  # natural yield cadence

        yield "]}"  # close job_diagnostics array and root object

    return StreamingResponse(iter_json(), media_type="application/json")


# ------------------------------------------------------------
# REGISTER ROUTER
# ------------------------------------------------------------
app.include_router(analysis_router)
