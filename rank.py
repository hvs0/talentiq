
import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

CORE_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "vector search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "retrieval", "ranking", "search", "recommendation",
    "nlp", "natural language processing", "information retrieval",
    "python", "machine learning", "deep learning", "neural networks",
    "transformers", "huggingface", "pytorch", "tensorflow",
    "llm", "large language models", "fine-tuning", "lora", "qlora", "peft",
    "rag", "retrieval augmented generation",
    "bert", "gpt", "bge", "e5",
    "ndcg", "mrr", "evaluation", "a/b testing",
    "learning to rank", "xgboost", "lightgbm",
    "langchain", "langraph",
    "mlops", "model deployment", "inference",
    "distributed systems", "data pipelines",
    "sql", "spark", "airflow",
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
    "xgboost", "lightgbm", "catboost", "random forest",
    "mlops", "mlflow", "wandb", "weights & biases",
    "model deployment", "inference optimization", "onnx", "tensorrt",
    "data science", "feature engineering", "model evaluation",
    "image classification", "object detection", "speech recognition",
    "text classification", "sentiment analysis", "named entity recognition",
    "question answering", "text generation", "summarization",
    "generative ai", "stable diffusion", "diffusion models",
    "attention mechanism", "self-attention", "multi-head attention",
    "cnn", "rnn", "lstm", "gru", "gan",
    "bayesian", "statistical modeling", "time series",
    "anomaly detection", "clustering", "dimensionality reduction",
    "openai", "anthropic", "claude", "groq",
    "prompt engineering", "prompt tuning",
    "nlu", "nli", "tts", "asr",
}

STRONG_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer", "senior machine learning engineer",
    "staff ml engineer", "staff ai engineer", "principal ml engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "applied scientist", "research engineer", "ml scientist",
    "data scientist", "senior data scientist", "lead data scientist",
    "deep learning engineer", "computer vision engineer",
]

MODERATE_TITLES = [
    "software engineer", "senior software engineer", "backend engineer",
    "senior backend engineer", "full stack engineer", "data engineer",
    "senior data engineer", "platform engineer", "infrastructure engineer",
    "tech lead", "engineering manager", "solutions architect",
    "analytics engineer", "devops engineer",
]

NON_ENGINEERING_TITLES = [
    "hr manager", "marketing manager", "sales executive", "accountant",
    "content writer", "graphic designer", "civil engineer",
    "mechanical engineer", "operations manager", "customer support",
    "project manager", "business analyst", "product manager",
    "recruiter", "teacher", "professor", "consultant",
    "legal", "finance", "admin",
]

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "l&t infotech",
    "ltimindtree", "hexaware", "persistent", "zensar", "cyient",
    "virtusa", "birlasoft", "sonata software", "coforge",
    "deloitte", "ey", "pwc", "kpmg", "mckinsey", "bcg", "bain",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "new delhi", "gurgaon", "gurugram",
    "hyderabad", "bangalore", "bengaluru", "mumbai", "chennai",
    "delhi ncr", "ncr",
}

INDIA_LOCATIONS = PREFERRED_LOCATIONS | {
    "kolkata", "ahmedabad", "jaipur", "lucknow", "indore",
    "chandigarh", "kochi", "thiruvananthapuram", "coimbatore",
    "nagpur", "bhopal", "visakhapatnam", "patna", "vadodara",
}

def normalize_skill(s):
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s/&+#.-]', '', s)
    return s

def title_match_score(title):
    t = title.lower().strip()
    for strong in STRONG_TITLES:
        if strong in t:
            return 1.0
    for mod in MODERATE_TITLES:
        if mod in t:
            return 0.4
    for non_eng in NON_ENGINEERING_TITLES:
        if non_eng in t:
            return 0.0
    return 0.2

def career_relevance_score(career_history):
    if not career_history:
        return 0.0

    total_months_ai = 0
    total_months_product = 0
    has_product_company = False
    has_shipped_systems = False

    ai_keywords = {"ai", "ml", "machine learning", "deep learning", "nlp",
                   "data science", "search", "ranking", "recommendation",
                   "retrieval", "embeddings", "neural", "model"}
    shipping_keywords = {"deployed", "shipped", "production", "scale", "users",
                         "real-time", "pipeline", "serving", "a/b test",
                         "launched", "built", "designed", "implemented",
                         "end-to-end", "system"}

    for job in career_history:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        duration = job.get("duration_months", 0)
        company_size = job.get("company_size", "")

        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_ai_role = any(k in title or k in desc for k in ai_keywords)
        has_shipping = any(k in desc for k in shipping_keywords)

        if is_ai_role:
            total_months_ai += duration
        if not is_consulting:
            total_months_product += duration
            has_product_company = True
        if has_shipping and is_ai_role:
            has_shipped_systems = True

    score = 0.0
    score += min(total_months_ai / 60, 1.0) * 0.4
    score += min(total_months_product / 72, 1.0) * 0.25
    if has_shipped_systems:
        score += 0.2
    if has_product_company:
        score += 0.15

    return min(score, 1.0)

def skill_match_score(skills):
    if not skills:
        return 0.0, 0

    ai_skill_count = 0
    weighted_score = 0.0
    total_weight = 0.0

    for skill in skills:
        name = normalize_skill(skill.get("name", ""))
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration_months = skill.get("duration_months", 0)

        is_ai = any(ai_s in name or name in ai_s for ai_s in AI_ML_SKILLS)
        is_core = any(cs in name or name in cs for cs in CORE_SKILLS)

        if not (is_ai or is_core):
            continue

        ai_skill_count += 1

        prof_w = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
        pw = prof_w.get(proficiency, 0.3)

        trust = 1.0
        if endorsements > 20:
            trust += 0.2
        elif endorsements > 5:
            trust += 0.1
        if duration_months > 24:
            trust += 0.2
        elif duration_months > 12:
            trust += 0.1
        if proficiency == "expert" and duration_months == 0:
            trust *= 0.3
        if proficiency in ("advanced", "expert") and endorsements == 0:
            trust *= 0.5

        weight = 1.5 if is_core else 1.0
        weighted_score += pw * trust * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0, 0

    normalized = weighted_score / total_weight
    return min(normalized, 1.0), ai_skill_count

def experience_fit_score(years_exp):
    if years_exp is None:
        return 0.3
    if 5 <= years_exp <= 9:
        return 1.0
    elif 4 <= years_exp < 5:
        return 0.8
    elif 9 < years_exp <= 12:
        return 0.7
    elif 3 <= years_exp < 4:
        return 0.5
    elif 12 < years_exp <= 15:
        return 0.5
    elif years_exp < 3:
        return 0.2
    else:
        return 0.3

def location_score(location, country, willing_to_relocate):
    loc = (location or "").lower()
    ctry = (country or "").lower()

    if ctry == "india":
        if any(city in loc for city in {"pune", "noida", "delhi", "ncr", "gurgaon", "gurugram"}):
            return 1.0
        if any(city in loc for city in PREFERRED_LOCATIONS):
            return 0.85
        if any(city in loc for city in INDIA_LOCATIONS):
            return 0.7 if willing_to_relocate else 0.5
        return 0.6 if willing_to_relocate else 0.4
    else:
        if willing_to_relocate:
            return 0.4
        return 0.2

def behavioral_score(signals):
    score = 0.0

    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_since = (date(2025, 6, 1) - last_dt).days
            if days_since < 7:
                score += 0.20
            elif days_since < 30:
                score += 0.17
            elif days_since < 60:
                score += 0.13
            elif days_since < 90:
                score += 0.08
            elif days_since < 180:
                score += 0.03
        except ValueError:
            score += 0.05

    if signals.get("open_to_work_flag"):
        score += 0.10

    rr = signals.get("recruiter_response_rate", 0)
    score += rr * 0.15

    rt = signals.get("avg_response_time_hours", 72)
    if rt < 4:
        score += 0.08
    elif rt < 12:
        score += 0.06
    elif rt < 24:
        score += 0.04
    elif rt < 48:
        score += 0.02

    icr = signals.get("interview_completion_rate", 0)
    score += icr * 0.10

    pcs = signals.get("profile_completeness_score", 0)
    score += (pcs / 100) * 0.07

    gh = signals.get("github_activity_score", -1)
    if gh >= 0:
        score += (gh / 100) * 0.10

    np_days = signals.get("notice_period_days", 90)
    if np_days <= 15:
        score += 0.08
    elif np_days <= 30:
        score += 0.06
    elif np_days <= 60:
        score += 0.03

    if signals.get("verified_email"):
        score += 0.02
    if signals.get("verified_phone"):
        score += 0.02
    if signals.get("linkedin_connected"):
        score += 0.03

    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        score += 0.05
    elif saved >= 2:
        score += 0.03
    elif saved >= 1:
        score += 0.01

    return min(score, 1.0)

def detect_honeypot(candidate):
    flags = 0

    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0)

    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 3:
        flags += 2

    for job in career:
        duration = job.get("duration_months", 0)
        start = job.get("start_date", "")
        if duration > 120:
            flags += 1

    title = profile.get("current_title", "").lower()
    is_non_eng = any(t in title for t in NON_ENGINEERING_TITLES)
    ai_skill_names = [normalize_skill(s["name"]) for s in skills]
    ai_count = sum(1 for s in ai_skill_names if any(
        ai in s or s in ai for ai in AI_ML_SKILLS
    ))
    if is_non_eng and ai_count >= 8:
        flags += 2

    if ai_count >= 10 and not any(
        any(k in job.get("description", "").lower()
            for k in {"ml", "ai", "machine learning", "model", "neural"})
        for job in career
    ):
        flags += 2

    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if years_exp > 0 and total_career_months > 0:
        career_years = total_career_months / 12
        if abs(career_years - years_exp) > 5:
            flags += 1

    return flags >= 3

def text_relevance_score(candidate):
    profile = candidate.get("profile", {})
    summary = (profile.get("summary", "") + " " + profile.get("headline", "")).lower()
    career = candidate.get("career_history", [])

    all_text = summary
    for job in career:
        all_text += " " + job.get("description", "").lower()
        all_text += " " + job.get("title", "").lower()

    high_value_terms = [
        "ranking system", "retrieval", "search system", "recommendation",
        "embeddings", "vector", "faiss", "semantic", "production",
        "deployed", "scale", "real users", "a/b test",
        "fine-tuning", "llm", "language model", "rag",
        "evaluation", "ndcg", "precision", "recall",
        "product company", "startup", "series",
    ]

    medium_value_terms = [
        "machine learning", "deep learning", "neural",
        "python", "pytorch", "tensorflow",
        "nlp", "natural language", "text",
        "data pipeline", "infrastructure", "mlops",
        "shipped", "built", "designed", "implemented",
        "end-to-end", "system design",
    ]

    score = 0.0
    for term in high_value_terms:
        if term in all_text:
            score += 0.06
    for term in medium_value_terms:
        if term in all_text:
            score += 0.03

    return min(score, 1.0)

def compute_composite_score(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    if detect_honeypot(candidate):
        return -999.0, "Honeypot detected: impossible profile signals."

    title_sc = title_match_score(profile.get("current_title", ""))
    career_sc = career_relevance_score(candidate.get("career_history", []))
    skill_sc, ai_count = skill_match_score(candidate.get("skills", []))
    exp_sc = experience_fit_score(profile.get("years_of_experience"))
    loc_sc = location_score(
        profile.get("location", ""),
        profile.get("country", ""),
        signals.get("willing_to_relocate", False)
    )
    behav_sc = behavioral_score(signals)
    text_sc = text_relevance_score(candidate)

    composite = (
        title_sc * 22.0 +
        career_sc * 23.0 +
        skill_sc * 18.0 +
        text_sc * 12.0 +
        behav_sc * 10.0 +
        exp_sc * 5.0 +
        loc_sc * 3.0
    )

    if title_sc == 0.0 and career_sc < 0.2:
        composite *= 0.1

    if title_sc >= 0.8 and career_sc >= 0.6:
        composite += 5.0

    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_since = (date(2025, 6, 1) - last_dt).days
            if days_since > 180 and signals.get("recruiter_response_rate", 0) < 0.1:
                composite *= 0.5
        except ValueError:
            pass

    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        ai_assessments = [v for k, v in assessments.items()
                         if any(ai in k.lower() for ai in AI_ML_SKILLS)]
        if ai_assessments:
            composite += (sum(ai_assessments) / len(ai_assessments)) / 100 * 3.0

    gh = signals.get("github_activity_score", -1)
    if gh > 0:
        composite += (gh / 100) * 2.0

    saved = signals.get("saved_by_recruiters_30d", 0)
    composite += min(saved / 10, 1.0) * 1.5

    rr = signals.get("recruiter_response_rate", 0)
    composite += rr * 2.0

    np_days = signals.get("notice_period_days", 90)
    composite += max(0, (90 - np_days) / 90) * 1.5

    composite += math.log1p(ai_count) * 1.0

    return composite, None

def generate_reasoning(candidate, score):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "Unknown")
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    ai_skills = []
    for s in skills:
        name = normalize_skill(s.get("name", ""))
        if any(ai in name or name in ai for ai in AI_ML_SKILLS):
            ai_skills.append(s.get("name", ""))

    rr = signals.get("recruiter_response_rate", 0)
    gh = signals.get("github_activity_score", -1)
    np_days = signals.get("notice_period_days", 0)
    open_to_work = signals.get("open_to_work_flag", False)

    ai_roles = [j for j in career if any(
        k in j.get("title", "").lower() or k in j.get("description", "").lower()
        for k in {"ai", "ml", "machine learning", "data scien", "nlp", "search", "ranking"}
    )]

    parts = []
    parts.append(f"{title} at {company} with {years:.1f} yrs exp")

    if ai_skills:
        top_skills = ai_skills[:4]
        parts.append(f"AI skills: {', '.join(top_skills)}")

    if ai_roles:
        parts.append(f"{len(ai_roles)} AI/ML roles in career")

    if location and country:
        parts.append(f"{location}, {country}")

    signals_parts = []
    if rr > 0.5:
        signals_parts.append(f"response rate {rr:.0%}")
    if gh >= 50:
        signals_parts.append(f"GitHub score {gh:.0f}")
    if np_days <= 30:
        signals_parts.append(f"{np_days}d notice")
    if open_to_work:
        signals_parts.append("open to work")

    if signals_parts:
        parts.append("; ".join(signals_parts))

    reasoning = "; ".join(parts)
    return reasoning[:250]

def fast_prefilter(candidate):
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title_sc = title_match_score(title)

    skill_names = " ".join(normalize_skill(s.get("name", "")) for s in skills)
    has_ai_skill = any(ai in skill_names for ai in [
        "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
        "ai", "neural", "llm", "embeddings", "transformers", "bert", "gpt",
        "data science", "ml", "rag", "langchain", "search", "ranking",
        "recommendation", "faiss", "vector", "huggingface", "fine-tuning",
        "model", "inference", "scikit", "xgboost", "spacy",
    ])

    career_text = " ".join(
        j.get("title", "") + " " + j.get("description", "")
        for j in career
    ).lower()
    has_ai_career = any(k in career_text for k in [
        "machine learning", "ai", "ml engineer", "data scien",
        "nlp", "deep learning", "neural", "search engineer",
        "ranking", "recommendation", "retrieval",
    ])

    if title_sc >= 0.8:
        return True
    if title_sc >= 0.4 and (has_ai_skill or has_ai_career):
        return True
    if has_ai_skill and has_ai_career:
        return True
    if title_sc >= 0.2 and has_ai_skill and has_ai_career:
        return True

    return False

def rank_candidates(candidates_path, output_path, top_n=100):
    print(f"[1/4] Loading candidates from {candidates_path}...")
    t0 = time.time()

    candidates_path = Path(candidates_path)
    is_jsonl = candidates_path.suffix == ".jsonl"
    is_json = candidates_path.suffix == ".json"

    candidates = []
    if is_jsonl:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif is_json:
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        import gzip
        with gzip.open(candidates_path, "rt", encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f if line.strip()]

    t1 = time.time()
    print(f"    Loaded {len(candidates)} candidates in {t1-t0:.1f}s")

    print("[2/4] Pre-filtering candidates...")
    filtered = [(c, i) for i, c in enumerate(candidates) if fast_prefilter(c)]
    t2 = time.time()
    print(f"    {len(filtered)} candidates passed pre-filter in {t2-t1:.1f}s")

    if len(filtered) < 500:
        print("    Warning: Few candidates passed filter, using all candidates...")
        filtered = [(c, i) for i, c in enumerate(candidates)]

    print("[3/4] Scoring candidates...")
    scored = []
    honeypot_count = 0

    for candidate, idx in filtered:
        score, honeypot_reason = compute_composite_score(candidate)
        if honeypot_reason:
            honeypot_count += 1
            continue
        scored.append((score, candidate))

    t3 = time.time()
    print(f"    Scored {len(scored)} candidates in {t3-t2:.1f}s")
    print(f"    Detected {honeypot_count} honeypots")

    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top_candidates = scored[:top_n]

    print(f"[4/4] Generating top {top_n} ranking...")
    if top_candidates:
        max_score = top_candidates[0][0]
        min_score = top_candidates[-1][0] if len(top_candidates) > 1 else 0
        score_range = max_score - min_score if max_score != min_score else 1.0

    rows = []
    for rank, (raw_score, candidate) in enumerate(top_candidates, 1):
        cid = candidate["candidate_id"]
        reasoning = generate_reasoning(candidate, raw_score)
        normalized = 0.40 + 0.59 * ((raw_score - min_score) / score_range)
        score_rounded = round(normalized, 4)
        rows.append((cid, rank, score_rounded, reasoning))

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in rows:
            writer.writerow([cid, rank, score, reasoning])

    t4 = time.time()
    print(f"\nDone! Output written to {output_path}")
    print(f"Total runtime: {t4-t0:.1f}s")
    print(f"Top 5 candidates:")
    for cid, rank, score, reasoning in rows[:5]:
        print(f"  #{rank} {cid} (score={score:.4f}): {reasoning[:80]}...")

    return rows

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Candidate Ranking for Redrob Hackathon"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl or sample_candidates.json"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output CSV path"
    )
    parser.add_argument(
        "--top", type=int, default=100,
        help="Number of top candidates to output (default: 100)"
    )
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.top)

if __name__ == "__main__":
    main()
