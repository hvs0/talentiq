import json
import math
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from db import init_db, upsert_candidate, get_all_candidates, delete_candidate, count_candidates, bulk_insert, get_db, create_job, get_all_jobs, get_job, add_to_shortlist, remove_from_shortlist, get_shortlist, delete_job
from parsers import parse_csv, parse_resume_text

app = FastAPI(title="TalentIQ Ranker API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CANDIDATES = []
CANDIDATES_BY_ID = {}
DATA_PATH = Path("/data/candidates.jsonl")
SAMPLE_PATH = Path("/data/sample_candidates.json")


def load_candidates():
    global CANDIDATES, CANDIDATES_BY_ID
    paths = [
        Path("/data/candidates.jsonl"),
        Path("/data/sample_candidates.json"),
        Path("./data/candidates.jsonl"),
        Path("./data/sample_candidates.json"),
    ]
    path = None
    for p in paths:
        if p.exists():
            path = p
            break

    if not path:
        print("WARNING: No candidate data found!")
        return

    print(f"Loading candidates from {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)...")

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            CANDIDATES = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            CANDIDATES = json.load(f)

    CANDIDATES_BY_ID = {c["candidate_id"]: c for c in CANDIDATES}
    print(f"Loaded {len(CANDIDATES)} candidates")


JD_CRITICAL_SKILLS = {
    "ndcg", "mrr", "map", "precision@k", "recall@k", "hit rate",
    "learning to rank", "ltr", "lambdamart", "xgboost ranker",
    "ranking evaluation", "relevance ranking", "search ranking",
    "information retrieval", "retrieval", "ranking", "search",
    "recommendation", "candidate matching", "candidate ranking",
    "embeddings", "sentence-transformers", "vector database", "vector search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "semantic search", "bge", "e5",
    "search infrastructure", "search backend", "ranking pipeline",
    "retrieval system", "recommendation system",
}

CORE_SKILLS = {
    "nlp", "python", "pytorch", "tensorflow",
    "transformers", "huggingface", "bert", "gpt",
    "rag", "retrieval augmented generation", "langchain",
    "llm", "fine-tuning", "lora", "qlora", "peft",
    "xgboost", "lightgbm", "recommendation systems",
}

HR_TECH_COMPANIES = {
    "linkedin", "eightfold", "darwinbox", "beamery", "seekout",
    "naukri", "foundit", "instahyre", "greenhouse", "lever",
    "workday", "hirevue", "phenom", "icims", "jobvite",
    "redrob", "hireez", "gem", "ashby", "rippling",
}

AI_ML_SKILLS = {
    "machine learning", "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "transformers", "pytorch", "tensorflow", "keras", "scikit-learn",
    "huggingface", "spacy", "nltk", "gensim",
    "llm", "large language models", "gpt", "bert", "t5",
    "fine-tuning", "lora", "qlora", "peft", "rlhf",
    "rag", "retrieval augmented generation", "langchain",
    "embeddings", "sentence-transformers", "vector search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "recommendation systems", "ranking", "search", "information retrieval",
    "xgboost", "lightgbm", "catboost",
    "mlops", "mlflow", "wandb", "weights & biases",
    "model deployment", "inference optimization", "onnx",
    "data science", "feature engineering", "model evaluation",
    "image classification", "object detection", "speech recognition",
    "text classification", "sentiment analysis",
    "generative ai", "stable diffusion", "diffusion models",
    "cnn", "rnn", "lstm", "gan",
    "openai", "anthropic", "groq",
    "prompt engineering", "nlu", "tts", "asr",
}

IRRELEVANT_SKILLS = {
    "computer vision", "image classification", "object detection",
    "stable diffusion", "diffusion models", "cnn", "gan",
    "speech recognition", "asr", "tts", "yolo",
    "image segmentation", "face detection", "face recognition",
    "video processing", "optical character recognition",
}

GENERIC_BUZZWORDS = {
    "machine learning", "deep learning", "ai", "artificial intelligence",
    "data science", "neural networks", "llm", "large language models",
}

STRONGEST_TITLES = [
    "search engineer", "ranking engineer", "recommendation engineer",
    "retrieval engineer", "nlp engineer", "senior search engineer",
    "senior nlp engineer", "senior ranking engineer",
    "information retrieval engineer", "relevance engineer",
]

STRONG_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer", "senior machine learning engineer",
    "staff ml engineer", "staff ai engineer", "principal ml engineer",
    "applied scientist",
    "senior data scientist", "lead data scientist",
]

WEAK_TITLES = [
    "data scientist", "deep learning engineer", "computer vision engineer",
    "junior ml engineer", "junior data scientist", "applied ml engineer",
    "applied machine learning engineer", "research engineer", "research scientist",
    "ai research engineer", "ml scientist",
]

MODERATE_TITLES = [
    "software engineer", "senior software engineer", "backend engineer",
    "data engineer", "platform engineer", "tech lead", "analytics engineer",
]

NON_ENGINEERING_TITLES = [
    "hr manager", "marketing manager", "sales executive", "accountant",
    "content writer", "graphic designer", "civil engineer",
    "mechanical engineer", "operations manager", "customer support",
    "project manager", "business analyst",
]

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "l&t infotech",
    "deloitte", "ey", "pwc", "kpmg",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "new delhi", "gurgaon", "gurugram",
    "hyderabad", "bangalore", "bengaluru", "mumbai", "chennai", "delhi ncr",
}


def normalize_skill(s):
    s = s.lower().strip()
    if ':' in s:
        s = s.split(':', 1)[-1].strip()
    return re.sub(r'[^a-z0-9\s/&+#.-]', '', s)


def title_match_score(title):
    t = title.lower().strip()
    for s in STRONGEST_TITLES:
        if s in t:
            return 1.0
    for s in STRONG_TITLES:
        if s in t:
            return 0.7
    for s in WEAK_TITLES:
        if s in t:
            return 0.3
    for m in MODERATE_TITLES:
        if m in t:
            return 0.25
    for n in NON_ENGINEERING_TITLES:
        if n in t:
            return 0.0
    return 0.15


def career_relevance_score(career_history):
    if not career_history:
        return 0.0
    total_months_retrieval = 0
    total_months_ai = 0
    has_shipped = False
    has_hrtech = False
    has_production_retrieval = False
    retrieval_kw = {"search", "ranking", "retrieval", "recommendation", "embeddings", "vector", "semantic", "relevance", "ndcg", "mrr", "information retrieval", "candidate matching", "search infrastructure"}
    production_kw = {"production", "scale", "million", "real-time", "serving", "infrastructure", "backend", "pipeline", "large-scale", "distributed"}
    ai_kw = {"ai", "ml", "machine learning", "deep learning", "nlp", "data science", "neural", "model"}
    ship_kw = {"deployed", "shipped", "production", "scale", "users", "real-time", "pipeline", "serving", "launched", "built", "end-to-end"}

    for job in career_history:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        duration = job.get("duration_months", 0)
        is_retrieval = any(k in title or k in desc for k in retrieval_kw)
        is_ai = any(k in title or k in desc for k in ai_kw)
        is_production = any(k in desc for k in production_kw)
        if is_retrieval:
            total_months_retrieval += duration
        if is_ai:
            total_months_ai += duration
        if any(k in desc for k in ship_kw) and (is_retrieval or is_ai):
            has_shipped = True
        if is_retrieval and is_production:
            has_production_retrieval = True
        if any(h in company for h in HR_TECH_COMPANIES):
            has_hrtech = True

    score = min(total_months_retrieval / 36, 1.0) * 0.40
    score += min(total_months_ai / 60, 1.0) * 0.15
    if has_shipped:
        score += 0.20
    if has_production_retrieval:
        score += 0.15
    if has_hrtech:
        score += 0.10
    return min(score, 1.0)


def skill_match_score(skills, candidate=None):
    if not skills:
        skills = []
    extra_skill_text = ""
    if candidate:
        p = candidate.get("profile", {})
        extra_skill_text = f"{p.get('summary','')} {p.get('headline','')}".lower()
        for s in candidate.get("skills", []):
            extra_skill_text += " " + s.get("name", "").lower()

    ai_count = 0
    critical_count = 0
    irrelevant_count = 0
    weighted = 0.0
    total_w = 0.0
    for skill in skills:
        name = normalize_skill(skill.get("name", ""))
        is_critical = any(cs in name or name in cs for cs in JD_CRITICAL_SKILLS)
        is_core = any(cs in name or name in cs for cs in CORE_SKILLS)
        is_ai = any(ai in name or name in ai for ai in AI_ML_SKILLS)
        is_irrelevant = any(ir in name or name in ir for ir in IRRELEVANT_SKILLS)
        is_buzzword = any(bw in name or name in bw for bw in GENERIC_BUZZWORDS)

        if is_irrelevant:
            irrelevant_count += 1
            continue
        if not (is_ai or is_core or is_critical):
            continue

        ai_count += 1
        if is_critical:
            critical_count += 1

        prof_w = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
        pw = prof_w.get(skill.get("proficiency", "beginner"), 0.3)
        trust = 1.0
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)
        if endorsements > 20: trust += 0.2
        elif endorsements > 5: trust += 0.1
        if duration > 24: trust += 0.2
        elif duration > 12: trust += 0.1
        if skill.get("proficiency") == "expert" and duration == 0: trust *= 0.3
        if skill.get("proficiency") in ("advanced", "expert") and endorsements == 0: trust *= 0.5

        if is_critical:
            weight = 3.0
        elif is_core:
            weight = 1.5
        elif is_buzzword:
            weight = 0.3
        else:
            weight = 1.0
        weighted += pw * trust * weight
        total_w += weight

    if extra_skill_text and ai_count < 3:
        for sk in JD_CRITICAL_SKILLS:
            if sk in extra_skill_text and len(sk) > 3:
                critical_count += 1
                ai_count += 1
                weighted += 1.2
                total_w += 3.0
        for sk in CORE_SKILLS:
            if sk in extra_skill_text and len(sk) > 3:
                ai_count += 1
                weighted += 0.4
                total_w += 1.0

    if total_w == 0:
        return 0.0, 0, critical_count
    score = min(weighted / total_w, 1.0)
    if irrelevant_count > 3 and critical_count == 0:
        score *= 0.5
    return score, ai_count, critical_count


def extract_experience_range(jd_text: str):
    jd = jd_text.lower()
    patterns = [
        r'(\d+\.?\d*)\s*[-\u2013to]+\s*(\d+\.?\d*)\s*(?:years?|yrs?)',
        r'(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
        r'(?:experience|exp)[:\s]*(\d+\.?\d*)\s*[-\u2013to]+\s*(\d+\.?\d*)\s*(?:years?|yrs?)',
        r'(?:experience|exp)[:\s]*(\d+\.?\d*)\s*(?:years?|yrs?)',
    ]
    for pat in patterns:
        m = re.search(pat, jd)
        if m:
            if m.lastindex == 2:
                return float(m.group(1)), float(m.group(2))
            else:
                val = float(m.group(1))
                return val, val + 2
    return None, None


def experience_fit_score(years, min_exp, max_exp):
    if years is None:
        return 0.3
    if min_exp is not None and max_exp is not None:
        if min_exp <= years <= max_exp:
            return 1.0
        elif years < min_exp:
            diff = min_exp - years
            if diff <= 0.5:
                return 0.8
            elif diff <= 1:
                return 0.5
            elif diff <= 2:
                return 0.3
            return 0.1
        else:
            diff = years - max_exp
            if diff <= 1:
                return 0.7
            elif diff <= 2:
                return 0.5
            elif diff <= 4:
                return 0.3
            return 0.1
    if 5 <= years <= 9: return 1.0
    elif 4 <= years < 5: return 0.8
    elif 9 < years <= 12: return 0.7
    elif 3 <= years < 4: return 0.5
    return 0.3


def location_score(location, country, relocate):
    loc = (location or "").lower()
    ctry = (country or "").lower()
    if ctry == "india":
        if any(c in loc for c in {"pune", "noida", "delhi", "ncr", "gurgaon", "gurugram"}):
            return 1.0
        if any(c in loc for c in PREFERRED_LOCATIONS):
            return 0.85
        return 0.7 if relocate else 0.5
    return 0.4 if relocate else 0.2


def behavioral_score(signals):
    score = 0.0
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            days = (date(2025, 6, 1) - datetime.strptime(last_active, "%Y-%m-%d").date()).days
            if days < 7: score += 0.20
            elif days < 30: score += 0.17
            elif days < 60: score += 0.13
            elif days < 90: score += 0.08
            elif days < 180: score += 0.03
        except ValueError:
            pass
    if signals.get("open_to_work_flag"): score += 0.10
    score += signals.get("recruiter_response_rate", 0) * 0.15
    rt = signals.get("avg_response_time_hours", 72)
    if rt < 4: score += 0.08
    elif rt < 12: score += 0.06
    elif rt < 24: score += 0.04
    score += signals.get("interview_completion_rate", 0) * 0.10
    score += (signals.get("profile_completeness_score", 0) / 100) * 0.07
    gh = signals.get("github_activity_score", -1)
    if gh >= 0: score += (gh / 100) * 0.10
    np_days = signals.get("notice_period_days", 90)
    if np_days <= 15: score += 0.08
    elif np_days <= 30: score += 0.06
    elif np_days <= 60: score += 0.03
    if signals.get("verified_email"): score += 0.02
    if signals.get("verified_phone"): score += 0.02
    if signals.get("linkedin_connected"): score += 0.03
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5: score += 0.05
    elif saved >= 2: score += 0.03
    return min(score, 1.0)


def text_relevance_score(candidate, jd_keywords=None):
    profile = candidate.get("profile", {})
    all_text = (profile.get("summary", "") + " " + profile.get("headline", "")).lower()
    for job in candidate.get("career_history", []):
        all_text += " " + job.get("description", "").lower() + " " + job.get("title", "").lower()

    top_tier = ["ndcg", "mrr", "learning to rank", "ltr", "lambdamart",
                "precision@k", "recall@k", "ranking evaluation",
                "candidate matching", "search ranking", "relevance ranking"]
    critical = ["ranking system", "retrieval", "search system", "recommendation",
                "embeddings", "vector", "faiss", "semantic search",
                "information retrieval", "relevance",
                "pinecone", "weaviate", "milvus", "qdrant", "opensearch",
                "sentence-transformers", "bge", "e5",
                "search infrastructure", "search backend", "ranking pipeline"]
    production = ["production", "large-scale", "million users", "real-time",
                  "serving", "distributed", "infrastructure", "scale"]
    hrtech = ["hr-tech", "talent", "candidate", "recruiting", "hiring",
              "job matching", "resume", "applicant"]
    high = ["fine-tuning", "rag", "evaluation", "llm", "language model",
            "a/b test", "deployed"]
    medium = ["python", "pytorch", "tensorflow", "nlp", "natural language",
              "data pipeline", "mlops", "shipped", "built", "end-to-end"]
    low_value = ["computer vision", "object detection", "image classification",
                 "diffusion", "gan", "speech recognition", "cnn", "yolo",
                 "image segmentation", "face detection"]

    if jd_keywords:
        for kw in jd_keywords:
            if kw.lower() in all_text:
                high.append(kw.lower())

    score = sum(0.12 for t in top_tier if t in all_text)
    score += sum(0.07 for t in critical if t in all_text)
    score += sum(0.04 for t in production if t in all_text)
    score += sum(0.03 for t in hrtech if t in all_text)
    score += sum(0.03 for t in high if t in all_text)
    score += sum(0.01 for t in medium if t in all_text)
    score -= sum(0.03 for t in low_value if t in all_text)
    return max(min(score, 1.0), 0.0)


def detect_honeypot(candidate):
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    flags = 0
    expert_zero = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if expert_zero >= 3: flags += 2
    title = profile.get("current_title", "").lower()
    is_non_eng = any(t in title for t in NON_ENGINEERING_TITLES)
    ai_count = sum(1 for s in skills if any(ai in normalize_skill(s.get("name", "")) or normalize_skill(s.get("name", "")) in ai for ai in AI_ML_SKILLS))
    if is_non_eng and ai_count >= 8: flags += 2
    if ai_count >= 10 and not any(any(k in j.get("description", "").lower() for k in {"ml", "ai", "machine learning", "model", "neural"}) for j in career):
        flags += 2
    return flags >= 3


def generic_skill_match(skills, jd_text, candidate=None):
    jd_lower = jd_text.lower()
    jd_words = set(re.findall(r'\b[a-z][a-z0-9+#.-]{1,}\b', jd_lower))
    matched = 0
    total_jd_skills = max(len(jd_words), 1)
    for skill in skills:
        name = normalize_skill(skill.get("name", ""))
        if any(w in name or name in w for w in jd_words if len(w) > 2):
            matched += 1
    if candidate:
        p = candidate.get("profile", {})
        text = f"{p.get('summary','')} {p.get('headline','')} {p.get('current_title','')}".lower()
        for w in jd_words:
            if len(w) > 3 and w in text:
                matched += 0.5
    return min(matched / max(total_jd_skills * 0.3, 1), 1.0)


def rank_candidates_for_jd(jd_text: str, top_n: int = 100):
    jd_lower = jd_text.lower()
    jd_keywords = [w for w in re.findall(r'\b[a-z]{3,}\b', jd_lower)
                   if w not in {"the", "and", "for", "that", "with", "this", "are", "from", "will", "have", "been", "not", "but"}]

    ai_jd_keywords = {"machine learning", "deep learning", "nlp", "ai engineer", "ml engineer",
                      "neural", "transformers", "pytorch", "tensorflow", "llm", "embeddings",
                      "computer vision", "reinforcement learning", "rag", "fine-tuning"}
    is_ai_jd = any(k in jd_lower for k in ai_jd_keywords)

    min_exp, max_exp = extract_experience_range(jd_text)

    scored = []
    for candidate in CANDIDATES:
        if detect_honeypot(candidate):
            continue

        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        exp_sc = experience_fit_score(profile.get("years_of_experience"), min_exp, max_exp)
        loc_sc = location_score(profile.get("location", ""), profile.get("country", ""), signals.get("willing_to_relocate", False))
        behav_sc = behavioral_score(signals)

        if is_ai_jd:
            title_sc = title_match_score(profile.get("current_title", ""))
            career_sc = career_relevance_score(candidate.get("career_history", []))
            skill_sc, ai_count, critical_count = skill_match_score(candidate.get("skills", []), candidate)
            text_sc = text_relevance_score(candidate, jd_keywords)

            if min_exp is None or min_exp >= 3:
                if title_sc == 0.0 and skill_sc < 0.1 and career_sc < 0.1:
                    continue

            exp_weight = 10.0 if (min_exp is not None) else 5.0
            composite = (
                title_sc * 20.0 + career_sc * 15.0 + skill_sc * 15.0 +
                text_sc * 18.0 + behav_sc * 5.0 + exp_sc * exp_weight + loc_sc * 2.0
            )

            composite += critical_count * 10.0

            skills_list = candidate.get("skills", [])
            irr_count = sum(1 for s in skills_list if any(ir in normalize_skill(s.get("name", "")) for ir in IRRELEVANT_SKILLS))
            composite -= irr_count * 5.0

            if title_sc == 0.0 and career_sc < 0.2:
                composite *= 0.1
            if title_sc >= 1.0 and critical_count >= 2:
                composite += 15.0
            elif title_sc >= 0.7 and critical_count >= 1:
                composite += 5.0

            composite += signals.get("recruiter_response_rate", 0) * 1.0
        else:
            gen_skill_sc = generic_skill_match(candidate.get("skills", []), jd_text, candidate)
            text_sc = text_relevance_score(candidate, jd_keywords)

            if gen_skill_sc == 0 and text_sc == 0:
                continue

            exp_weight = 30.0 if (min_exp is not None) else 10.0
            composite = (
                gen_skill_sc * 35.0 + exp_sc * exp_weight + text_sc * 15.0 +
                behav_sc * 10.0 + loc_sc * 3.0
            )

            composite += signals.get("recruiter_response_rate", 0) * 2.0

        if min_exp is not None and max_exp is not None:
            yrs = profile.get("years_of_experience", 0) or 0
            allowed_max = max_exp + max(1, max_exp * 0.5)
            if yrs > allowed_max:
                continue

        last_active = signals.get("last_active_date", "")
        if last_active:
            try:
                days = (date(2025, 6, 1) - datetime.strptime(last_active, "%Y-%m-%d").date()).days
                if days > 180 and signals.get("recruiter_response_rate", 0) < 0.1:
                    composite *= 0.5
            except ValueError:
                pass

        scored.append((composite, candidate, {
            "title": round(title_sc if is_ai_jd else 0, 3),
            "career": round(career_sc if is_ai_jd else 0, 3),
            "skills": round((skill_sc if is_ai_jd else gen_skill_sc), 3),
            "text": round(text_sc, 3),
            "behavioral": round(behav_sc, 3),
            "experience": round(exp_sc, 3),
            "location": round(loc_sc, 3),
        }))

    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top = scored[:top_n]

    if not top:
        return []

    max_s, min_s = top[0][0], top[-1][0]
    rng = max_s - min_s if max_s != min_s else 1.0

    results = []
    for rank, (raw, cand, breakdown) in enumerate(top, 1):
        norm_score = round(0.40 + 0.59 * ((raw - min_s) / rng), 4)
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        skills = cand.get("skills", [])
        ai_skills = [s["name"] for s in skills if any(ai in normalize_skill(s.get("name", "")) or normalize_skill(s.get("name", "")) in ai for ai in AI_ML_SKILLS)]

        results.append({
            "rank": rank,
            "candidate_id": cand["candidate_id"],
            "score": norm_score,
            "breakdown": breakdown,
            "profile": {
                "name": profile.get("anonymized_name", ""),
                "title": profile.get("current_title", ""),
                "company": profile.get("current_company", ""),
                "years_exp": profile.get("years_of_experience", 0),
                "location": profile.get("location", ""),
                "country": profile.get("country", ""),
                "headline": profile.get("headline", ""),
                "summary": profile.get("summary", ""),
            },
            "ai_skills": ai_skills[:10],
            "signals": {
                "github_score": signals.get("github_activity_score", -1),
                "response_rate": signals.get("recruiter_response_rate", 0),
                "notice_period": signals.get("notice_period_days", 0),
                "open_to_work": signals.get("open_to_work_flag", False),
                "last_active": signals.get("last_active_date", ""),
                "saved_by_recruiters": signals.get("saved_by_recruiters_30d", 0),
            },
            "career_history": cand.get("career_history", [])[:3],
            "education": cand.get("education", []),
        })

    return results


class RankRequest(BaseModel):
    jd_text: str
    top_n: int = 100


class RankResponse(BaseModel):
    candidates: list
    total_candidates: int
    filtered_candidates: int
    runtime_seconds: float


@app.on_event("startup")
async def startup():
    init_db()
    load_candidates()
    db_candidates = get_all_candidates()
    for c in db_candidates:
        c.pop("_source", None)
        c.pop("_created", None)
        if c["candidate_id"] not in CANDIDATES_BY_ID:
            CANDIDATES.append(c)
            CANDIDATES_BY_ID[c["candidate_id"]] = c
    print(f"Loaded {len(CANDIDATES)} candidates ({len(db_candidates)} from DB)")


@app.get("/health")
def health():
    return {"status": "ok", "candidates_loaded": len(CANDIDATES)}


@app.get("/")
def root():
    return {
        "app": "TalentIQ — Intelligent Candidate Ranking API",
        "candidates_loaded": len(CANDIDATES),
        "endpoints": {
            "POST /api/rank": "Rank candidates against a JD",
            "POST /api/upload": "Upload candidates JSON/JSONL file",
            "GET /api/stats": "Get system stats",
            "GET /api/candidate/{id}": "Get candidate details",
            "GET /docs": "Swagger UI",
        }
    }


@app.post("/api/rank")
def rank_endpoint(req: RankRequest):
    if not CANDIDATES:
        raise HTTPException(500, "No candidates loaded. Upload a file first via POST /api/upload")
    if not req.jd_text.strip():
        raise HTTPException(400, "JD text is required")

    t0 = time.time()
    results = rank_candidates_for_jd(req.jd_text, req.top_n)
    runtime = round(time.time() - t0, 2)

    return {
        "candidates": results,
        "total_candidates": len(CANDIDATES),
        "runtime_seconds": runtime,
    }


@app.post("/api/upload")
async def upload_candidates(file: UploadFile = File(...)):
    global CANDIDATES, CANDIDATES_BY_ID
    content = await file.read()
    text = content.decode("utf-8")

    try:
        if file.filename.endswith(".jsonl"):
            CANDIDATES = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            CANDIDATES = data if isinstance(data, list) else [data]
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    CANDIDATES_BY_ID = {c["candidate_id"]: c for c in CANDIDATES}
    return {"message": f"Loaded {len(CANDIDATES)} candidates", "count": len(CANDIDATES)}


@app.get("/api/candidate/{candidate_id}")
def get_candidate(candidate_id: str):
    cand = CANDIDATES_BY_ID.get(candidate_id)
    if not cand:
        raise HTTPException(404, "Candidate not found")
    return cand


@app.get("/api/stats")
def get_stats():
    return {
        "total_candidates": len(CANDIDATES),
        "ready": len(CANDIDATES) > 0,
    }


@app.get("/api/candidates")
def list_candidates(page: int = 1, limit: int = 50, search: str = ""):
    if search:
        q = search.lower()
        filtered = []
        for c in CANDIDATES:
            p = c.get("profile", {})
            skills_text = " ".join(s.get("name", "").lower() for s in c.get("skills", []))
            career_text = " ".join(j.get("company", "").lower() + " " + j.get("title", "").lower() for j in c.get("career_history", []))
            searchable = f"{c.get('candidate_id','')} {p.get('anonymized_name','')} {p.get('current_title','')} {p.get('headline','')} {p.get('current_company','')} {p.get('current_industry','')} {p.get('location','')} {p.get('country','')} {p.get('years_of_experience','')} {skills_text} {career_text}".lower()
            if q in searchable:
                filtered.append(c)
            if len(filtered) >= 200:
                break
        all_cands = filtered
    else:
        all_cands = CANDIDATES

    total = len(all_cands)
    start = (page - 1) * limit
    items = all_cands[start:start + limit]
    return {"candidates": items, "total": total, "page": page, "pages": (total + limit - 1) // limit}


@app.put("/api/candidate/{candidate_id}")
def update_candidate(candidate_id: str, data: dict):
    data["candidate_id"] = candidate_id
    upsert_candidate(candidate_id, data, "edited")
    CANDIDATES_BY_ID[candidate_id] = data
    for i, c in enumerate(CANDIDATES):
        if c.get("candidate_id") == candidate_id:
            CANDIDATES[i] = data
            break
    else:
        CANDIDATES.append(data)
    return {"message": "Updated", "candidate_id": candidate_id}


@app.delete("/api/candidate/{candidate_id}")
def remove_candidate(candidate_id: str):
    delete_candidate(candidate_id)
    CANDIDATES_BY_ID.pop(candidate_id, None)
    for i, c in enumerate(CANDIDATES):
        if c.get("candidate_id") == candidate_id:
            CANDIDATES.pop(i)
            break
    return {"message": "Deleted", "candidate_id": candidate_id}


@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")
    try:
        candidates = parse_csv(content)
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")

    if not candidates:
        raise HTTPException(400, "No candidates found in CSV")

    bulk_insert(candidates, "csv")
    for c in candidates:
        CANDIDATES.append(c)
        CANDIDATES_BY_ID[c["candidate_id"]] = c

    return {"message": f"Imported {len(candidates)} candidates from CSV", "count": len(candidates), "candidates": candidates}


@app.post("/api/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or ""

    if filename.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        except ImportError:
            text = content.decode("utf-8", errors="ignore")
    elif filename.lower().endswith(".docx"):
        try:
            from docx import Document
            import io as _io
            doc = Document(_io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            text = content.decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "Could not extract text from file")

    candidate = parse_resume_text(text, filename)

    import os
    os.makedirs("/data/resumes", exist_ok=True)
    ext = Path(filename).suffix or ".pdf"
    resume_path = f"/data/resumes/{candidate['candidate_id']}{ext}"
    with open(resume_path, "wb") as f:
        f.write(content)
    candidate["_resume_file"] = f"{candidate['candidate_id']}{ext}"

    upsert_candidate(candidate["candidate_id"], candidate, "resume")
    CANDIDATES.append(candidate)
    CANDIDATES_BY_ID[candidate["candidate_id"]] = candidate

    return {"message": "Resume parsed successfully", "candidate": candidate}


class SaveJobRequest(BaseModel):
    title: str
    jd_text: str
    jd_id: Optional[str] = None
    selected_candidates: list


@app.post("/api/jobs")
def save_job(req: SaveJobRequest):
    from db import get_db as _get_db

    jd_normalized = " ".join(req.jd_text.split()).strip().lower()

    conn = _get_db()
    existing = None
    rows = conn.execute("SELECT job_id, jd_text FROM jobs").fetchall()
    for row in rows:
        existing_norm = " ".join(row[1].split()).strip().lower()
        if existing_norm == jd_normalized:
            existing = row[0]
            break
    if not existing and req.jd_id:
        row = conn.execute("SELECT job_id FROM jobs WHERE title LIKE ?", (req.jd_id + '%',)).fetchone()
        if row:
            existing = row[0]
    conn.close()

    if existing:
        for c in req.selected_candidates:
            add_to_shortlist(existing, c["candidate_id"], c.get("rank", 0), c.get("score", 0))
        shortlist = get_shortlist(existing)
        return {"job_id": existing, "message": f"Updated — now {len(shortlist)} total candidates in this JD"}
    else:
        job_id = create_job(req.title, req.jd_text)
        for c in req.selected_candidates:
            add_to_shortlist(job_id, c["candidate_id"], c.get("rank", 0), c.get("score", 0))
        return {"job_id": job_id, "message": f"New JD saved with {len(req.selected_candidates)} candidates"}


@app.get("/api/jobs")
def list_jobs():
    jobs = get_all_jobs()
    for j in jobs:
        shortlist = get_shortlist(j["job_id"])
        j["shortlist_count"] = len(shortlist)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job_detail(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    shortlist = get_shortlist(job_id)
    enriched = []
    for s in shortlist:
        cand = CANDIDATES_BY_ID.get(s["candidate_id"])
        if cand:
            p = cand.get("profile", {})
            skills = [sk.get("name", "") for sk in cand.get("skills", []) if any(ai in sk.get("name", "").lower() for ai in ("ml", "ai", "python", "deep", "nlp", "pytorch", "tensorflow", "llm", "data", "search", "vector"))]
            enriched.append({**s, "profile": p, "skills": skills[:8]})
        else:
            enriched.append(s)
    job["shortlist"] = enriched
    return job


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: int):
    delete_job(job_id)
    return {"message": "Deleted"}


@app.post("/api/jobs/{job_id}/shortlist")
def add_candidate_to_shortlist(job_id: int, data: dict):
    add_to_shortlist(job_id, data["candidate_id"], data.get("rank", 0), data.get("score", 0))
    return {"message": "Added"}


@app.delete("/api/jobs/{job_id}/shortlist/{candidate_id}")
def remove_candidate_from_shortlist(job_id: int, candidate_id: str):
    remove_from_shortlist(job_id, candidate_id)
    return {"message": "Removed"}


from fastapi.responses import FileResponse

@app.get("/api/resume/{candidate_id}")
def get_resume(candidate_id: str):
    import os, glob
    resume_dir = "/data/resumes"
    if not os.path.exists(resume_dir):
        raise HTTPException(404, "No resumes uploaded")
    matches = glob.glob(f"{resume_dir}/{candidate_id}.*")
    if not matches:
        raise HTTPException(404, "No resume found for this candidate")
    filepath = matches[0]
    ext = Path(filepath).suffix.lower()
    media_types = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".doc": "application/msword", ".txt": "text/plain"}
    media_type = media_types.get(ext, "application/octet-stream")
    from starlette.responses import Response
    with open(filepath, "rb") as f:
        content = f.read()
    return Response(content=content, media_type=media_type, headers={"Content-Disposition": "inline"})


@app.get("/api/resume/{candidate_id}/exists")
def check_resume(candidate_id: str):
    import os, glob
    matches = glob.glob(f"/data/resumes/{candidate_id}.*")
    return {"has_resume": len(matches) > 0, "filename": os.path.basename(matches[0]) if matches else ""}
