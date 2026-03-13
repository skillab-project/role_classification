import os
import re
import math
import time
import json
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
import datetime
from scipy import sparse

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

def get_lock_status(lock_path: Path):
    """
    Checks if a lock is active and not stale (older than 30 mins).
    Returns (is_locked, message)
    """
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text())
            started_at = datetime.datetime.fromisoformat(lock_data["started_at"])
            elapsed = (datetime.datetime.now() - started_at).total_seconds()
            
            # If lock is less than 30 minutes old, it's active
            if elapsed < 1800: 
                return True, f"Analysis already in progress (started {int(elapsed // 60)}m ago)."
            else:
                # Lock is stale, remove it
                lock_path.unlink()
        except:
            lock_path.unlink()
    return False, None

def create_lock(lock_path: Path):
    lock_path.write_text(json.dumps({
        "started_at": datetime.datetime.now().isoformat()
    }))

def remove_lock(lock_path: Path):
    if lock_path.exists():
        lock_path.unlink()

@analysis_router.post("/jobs_emergingdck_train")
def train_job_emerging_classifier(
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    occupation_ids: Optional[str] = Query(
        None, description="Comma-separated occupation IDs"
    ),
    model_type: str = Query(
        "xgboost", description="Choose: xgboost, random_forest, logistic"
    ),
):
    load_dotenv()
    API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USER = os.getenv("TRACKER_USERNAME", "")
    PASS = os.getenv("TRACKER_PASSWORD", "")
    
    model_type_lower = model_type.lower() # Defined early to prevent NameError

    # ================================================================
    # 📁 CACHE SETUP
    # ================================================================
    folder = Path("Completed_Analyses")
    folder.mkdir(parents=True, exist_ok=True)

    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    occ_ids_list = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

    # Generate base filename
    clean_name = f"emerging_{model_type_lower}"
    for kw in keywords_list: clean_name += f"_{kw}"
    for occ in occ_ids_list:
        match = re.search(r'C\d+$', occ)
        clean_name += f"_{match.group(0)}" if match else f"_{occ.replace('/', '_').replace(':', '').replace('.', '')}"
    
    file_path = folder / f"completed_analysis_{clean_name}.json"
    lock_path = folder / f"lock_{clean_name}.lock"

    # 1. Check if already finished
    if file_path.exists():
        print(f"✅ Cache hit — streaming from '{file_path}'...")
        def stream_file():
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 1024): yield chunk
        return StreamingResponse(stream_file(), media_type="application/json")

    # 2. Check if currently running
    lock_path = folder / f"lock_{clean_name}.lock"
    is_locked, lock_msg = get_lock_status(lock_path)
    if is_locked:
        return {"status": "processing", "message": lock_msg}

    # 3. Create lock
    create_lock(lock_path)

    try:
        # ================================================================
        # 1️⃣ DATA FETCHING
        # ================================================================
        res = requests.post(f"{API}/login", json={"username": USER, "password": PASS}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded"}

        def fetch_page(p):
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
            for kw in keywords_list: fd.append(("keywords", kw))
            for occ in occ_ids_list: fd.append(("occupation_ids", occ))
            r = requests.post(f"{API}/jobs?page={p}&page_size=100", headers=headers, data=fd, timeout=180)
            return r.json() if r.status_code == 200 else {}

        probe_data = fetch_page(1)
        total_count = probe_data.get("count", 0)
        if total_count == 0: return {"error": "No jobs found for the given filters."}
        
        jobs = list(probe_data.get("items", []))
        total_pages = math.ceil(total_count / 100)
        print(f"📊 Fetching {total_pages} pages...")

        for page in range(2, total_pages + 1):
            print(f"📦 Progress: {page}/{total_pages} pages fetched.")
            data = fetch_page(page)
            items = data.get("items", [])
            if not items: break
            jobs.extend(items)
            if page % 500 == 0: print(f"📦 Progress: {len(jobs)}/{total_count}")

        # ================================================================
        # 2️⃣ SKILL RESOLUTION
        # ================================================================
        all_skill_uris = [s for j in jobs for s in j.get("skills", []) if isinstance(s, str) and s.startswith("http")]
        unique_uris = sorted(set(all_skill_uris))
        id_to_label = {}
        for i in range(0, len(unique_uris), 50):
            batch = unique_uris[i:i + 50]
            s_res = requests.post(f"{API}/skills", headers={"Authorization": f"Bearer {token}"}, data=[("ids", sid) for sid in batch], timeout=60)
            for s in s_res.json().get("items", []):
                sid = s.get("id")
                label = s.get("label") or sid # Safety for AttributeError: NoneType has no attribute strip
                id_to_label[sid] = str(label).strip().lower()

        # ================================================================
        # 3️⃣ LOGIC CONSTANTS
        # ================================================================
        SKILL_CATEGORIES = {
            "ai": ["ai", "ml", "learning", "neural", "vision", "nlp", "model", "predict", "classify", "inference", "cluster", "pattern"],
            "cloud": ["cloud", "aws", "azure", "gcp", "docker", "kube", "kubernetes", "serverless", "virtual", "container", "deploy"],
            "data": ["data", "etl", "clean", "analyse", "query", "sql", "warehouse", "pipeline", "extract", "load", "transform"],
            "devops": ["devops", "ci", "cd", "git", "monitor", "config", "infra", "build", "deploy", "automate"],
            "cyber": ["security", "cyber", "risk", "threat", "encrypt", "attack", "protect", "penetration", "firewall"],
            "software": ["software", "code", "test", "debug", "frontend", "backend", "api", "design", "develop"],
            "algorithms": ["algorithm", "math", "stats", "optim", "graph", "regress", "cluster", "logic"],
            "green": ["sustainable", "green", "co2", "carbon", "renewable", "energy", "solar", "wind", "climate"],
            "education": ["teaching", "pedagogy", "curriculum", "assessment", "educational"],
            "society": ["social", "community", "governance", "policy", "ethics"],
            "business": ["management", "leadership", "strategy", "marketing", "finance"],
            "health": ["medical", "health", "clinical", "patient", "diagnosis", "pharma"],
            "manufacturing": ["industrial", "automation", "robotic", "manufacture", "production"]
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
        
        EMERGING_SKILL_KEYWORDS = TITLE_KEYWORDS_EMERGING

        # ================================================================
        # 4️⃣ BUILDING MATRIX AND LABELS (The Filling Loop)
        # ================================================================
        SKILL_INDEX = sorted(set(id_to_label.get(u, u) for u in unique_uris))
        skill_to_idx = {name: i for i, name in enumerate(SKILL_INDEX)}
        
        all_skill_ids_resolved = [s for s in all_skill_uris if s in id_to_label]
        skill_global_counts = pd.Series(all_skill_ids_resolved).value_counts()
        top_10_skills = [{"skill": id_to_label.get(sk, sk), "count": int(cnt)} for sk, cnt in skill_global_counts.head(10).items()]
        
        rare_threshold = len(jobs) * 0.15
        rare_skill_ids = set(skill_global_counts[skill_global_counts < rare_threshold].index)

        print(f"🏗️ Building sparse matrix ({len(jobs)} x {len(SKILL_INDEX)})...")
        X_sparse = sparse.lil_matrix((len(jobs), len(SKILL_INDEX)), dtype=np.int8)
        labels = []

        for i, job in enumerate(jobs):
            title = job.get("title", "").lower()
            job_skill_ids = [s for s in job.get("skills", []) if s in id_to_label]
            
            for s_id in job_skill_ids:
                s_lab = id_to_label[s_id]
                if s_lab in skill_to_idx:
                    X_sparse[i, skill_to_idx[s_lab]] = 1
            
            if any(kw in title for kw in TITLE_KEYWORDS_ESTABLISHED): labels.append(0)
            elif any(kw in title for kw in TITLE_KEYWORDS_EMERGING): labels.append(1)
            elif any(s in rare_skill_ids for s in job_skill_ids): labels.append(1)
            else: labels.append(0)

        y = np.array(labels)
        X_sparse = X_sparse.tocsr() 

        # ================================================================
        # 5️⃣ MODEL TRAINING (With Subsampling for speed)
        # ================================================================
        model = None 
        explainer = None
        mode = None

        TRAIN_SIZE = min(100000, len(jobs))
        print(f"🚂 Training {model_type} on {TRAIN_SIZE} samples...")
        
        indices = np.arange(len(jobs))
        np.random.shuffle(indices)
        train_idx = indices[:TRAIN_SIZE]

        if model_type_lower == "xgboost":
            model = xgb.XGBClassifier(n_estimators=100, max_depth=4, tree_method="hist", n_jobs=-1)
            model.fit(X_sparse[train_idx], y[train_idx])
            explainer = shap.TreeExplainer(model)
            mode = "shap"
        elif model_type_lower == "random_forest":
            model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
            model.fit(X_sparse[train_idx], y[train_idx])
            explainer = shap.TreeExplainer(model)
            mode = "shap"
        elif model_type_lower == "logistic":
            model = LogisticRegression(max_iter=500, solver="liblinear")
            model.fit(X_sparse[train_idx], y[train_idx])
            mode = "linear"
        else:
            remove_lock(lock_path)
            return {"error": "Invalid model_type. Choose: xgboost, random_forest, logistic, svm, naive_bayes"}

        # ================================================================
        # 6️⃣ STREAMING GENERATOR
        # ================================================================
        def response_generator():
            num_jobs = len(jobs)
            global_skill_impacts = {skill: 0.0 for skill in SKILL_INDEX}
            
            # Predict probabilities
            probs = model.predict_proba(X_sparse)[:, 1]

            coefs = None
            if mode == "linear":
                coefs = model.coef_.ravel()

            BADGE_MAP = {"ai": "🤖", "cloud": "☁️", "data": "📊", "green": "🌿", "software": "🧰", "cyber": "🔐", "devops": "⚙️"}

            # FIX: Corrected variable name X -> X_sparse and sum logic
            header = {
                "message": f"✅ {model_type.upper()} trained.",
                "model_type": model_type,
                "jobs_used": num_jobs,
                "total_jobs_available": total_count,
                "skills_dim": len(SKILL_INDEX),
                "positive_label_ratio": float(np.mean(y)),
                "filters_used": {"keywords": keywords_list, "occupation_ids": occ_ids_list},
                "descriptive_statistics": {
                    "total_jobs_analyzed": num_jobs,
                    "pct_emerging": round(float(np.mean(y) * 100), 2),
                    "avg_skills_per_job": round(float(X_sparse.getnnz(axis=1).mean()), 2),
                    "top_10_most_common_skills": top_10_skills
                }
            }
            yield json.dumps(header, ensure_ascii=False)[:-1] + ', "job_diagnostics": ['

            BATCH_SIZE = 1000 
            for start in range(0, num_jobs, BATCH_SIZE):
                end = min(start + BATCH_SIZE, num_jobs)
                X_batch = X_sparse[start:end]
                
                # FIX: Handle Sparse matrix logic for impacts
                if mode == "shap":
                    # SHAP usually requires dense input for TreeExplainer
                    batch_impacts = explainer.shap_values(X_batch.toarray(), check_additivity=False)
                else:
                    # For linear, we use multiply for element-wise impact then convert to dense
                    batch_impacts = X_batch.multiply(coefs).toarray()

                for i in range(end - start):
                    g_idx = start + i
                    job_imp = batch_impacts[i]
                    top_idx = np.argsort(np.abs(job_imp))[::-1][:5]
                    
                    shap_expl = []
                    radar = {c: 0.0 for c in SKILL_CATEGORIES}
                    for idx in top_idx:
                        val = float(job_imp[idx])
                        if abs(val) < 1e-10: continue
                        name = SKILL_INDEX[idx]
                        shap_expl.append({"skill": name, "impact": val})
                        global_skill_impacts[name] += val
                        for cat, kws in SKILL_CATEGORIES.items():
                            if any(k in name for k in kws): radar[cat] += abs(val)

                    score = int(probs[g_idx] * 100)
                    max_val = max(radar.values()) if radar.values() else 0
                    max_r = max_val or 1
                    r_norm = {k: int((v/max_r)*100) for k, v in radar.items()}

                    job_diag = {
                        "job_title": jobs[g_idx].get("title", ""),
                        "classification": "Emerging" if score >= 50 else "Established",
                        "probability": round(float(probs[g_idx]), 4),
                        "emerging_score": score,
                        "explanation": shap_expl,
                        "radar_profile": r_norm,
                        "category_badges": [BADGE_MAP[c] for c in BADGE_MAP if r_norm.get(c, 0) > 50]
                    }
                    yield json.dumps(job_diag, ensure_ascii=False) + ("," if g_idx < num_jobs - 1 else "")

            # Footer...
            pos = sorted([(s, i) for s, i in global_skill_impacts.items() if i > 0], key=lambda x: x[1], reverse=True)[:15]
            neg = sorted([(s, i) for s, i in global_skill_impacts.items() if i < 0], key=lambda x: x[1])[:15]
            footer = {
                "global_top_emerging_skills": [{"skill": s, "global_impact": float(i)} for s, i in pos],
                "global_top_established_skills": [{"skill": s, "global_impact": float(i)} for s, i in neg]
            }
            yield '], ' + json.dumps(footer, ensure_ascii=False)[1:]

        # ================================================================
        # 7️⃣ EXECUTION
        # ================================================================
        def stream_and_cache():
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    for chunk in response_generator():
                        f.write(chunk)
                        yield chunk
                # Success! Remove the lock
                remove_lock(lock_path)
            except Exception as e:
                # Failure during streaming! Remove lock and partial file
                remove_lock(lock_path)
                if file_path.exists(): file_path.unlink()
                raise e

        return StreamingResponse(stream_and_cache(), media_type="application/json")
    except Exception as e:
        remove_lock(lock_path)
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}


app.include_router(analysis_router)