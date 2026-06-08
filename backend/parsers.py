import csv
import io
import re
import uuid


def generate_candidate_id():
    return f"CAND_{uuid.uuid4().hex[:7].upper()}"


def parse_csv(content: str) -> list:
    reader = csv.DictReader(io.StringIO(content))
    candidates = []
    for row in reader:
        norm = {k.strip().lower().replace(" ", "_"): v.strip() for k, v in row.items() if k and v}
        if norm:
            cand = build_candidate_from_flat(norm)
            candidates.append(cand)
    return candidates


def parse_resume_text(text: str, filename: str = "") -> dict:
    lines = text.strip().split("\n")
    name = lines[0].strip() if lines else "Unknown"

    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    phone_match = re.search(r'[\+]?[\d\s\-\(\)]{10,}', text)
    exp_match = re.search(r'(\d+\.?\d*)\s*(?:\+\s*)?(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)', text, re.I)
    years_exp = float(exp_match.group(1)) if exp_match else 0

    skills_section = re.search(r'(?:skills|technologies|tech stack)[:\s]*(.+?)(?:\n\n|\Z)', text, re.I | re.S)
    skills = []
    if skills_section:
        skill_text = skills_section.group(1)
        raw_skills = re.split(r'[,|•·\n]', skill_text)
        skills = [{"name": s.strip(), "proficiency": "intermediate", "endorsements": 0, "duration_months": 12}
                  for s in raw_skills if s.strip() and len(s.strip()) < 40]

    title_match = re.search(r'(?:title|role|position|designation)[:\s]*(.+)', text, re.I)
    title = title_match.group(1).strip() if title_match else ""
    if not title and len(lines) > 1:
        title = lines[1].strip() if not re.search(r'@|\.com|phone|\d{10}', lines[1]) else ""

    company_match = re.search(r'(?:company|organization|employer|at)\s*[:\s]*(.+)', text, re.I)
    company = company_match.group(1).strip() if company_match else ""

    loc_match = re.search(r'(?:location|city|address)[:\s]*(.+)', text, re.I)
    location = loc_match.group(1).strip() if loc_match else ""

    career = []
    exp_section = re.search(r'(?:experience|work history|employment)[:\s]*(.+?)(?:education|skills|projects|\Z)', text, re.I | re.S)
    if exp_section:
        jobs = re.findall(r'(.+?)\s*(?:at|@|-)\s*(.+?)(?:\n|$)', exp_section.group(1))
        for job_title, job_company in jobs[:5]:
            career.append({
                "company": job_company.strip(),
                "title": job_title.strip(),
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": len(career) == 0,
                "industry": "",
                "company_size": "51-200",
                "description": ""
            })

    candidate = {
        "candidate_id": generate_candidate_id(),
        "profile": {
            "anonymized_name": name,
            "headline": title,
            "summary": text[:500],
            "location": location,
            "country": "India",
            "years_of_experience": years_exp,
            "current_title": title,
            "current_company": company,
            "current_company_size": "51-200",
            "current_industry": ""
        },
        "career_history": career if career else [{
            "company": company or "Unknown",
            "title": title or "Unknown",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": int(years_exp * 12) if years_exp else 24,
            "is_current": True,
            "industry": "",
            "company_size": "51-200",
            "description": text[:200]
        }],
        "education": [],
        "skills": skills[:20],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 50,
            "signup_date": "2024-01-01",
            "last_active_date": "2025-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 0,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24,
            "skill_assessment_scores": {},
            "connection_count": 0,
            "endorsements_received": 0,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 10, "max": 30},
            "preferred_work_mode": "flexible",
            "willing_to_relocate": True,
            "github_activity_score": -1,
            "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": -1,
            "verified_email": bool(email_match),
            "verified_phone": bool(phone_match),
            "linkedin_connected": False
        }
    }
    return candidate


def build_candidate_from_flat(row: dict) -> dict:
    cid = row.get("candidate_id", row.get("id", generate_candidate_id()))
    name = row.get("name", row.get("full_name", row.get("candidate_name", "Unknown")))
    title = row.get("title", row.get("current_title", row.get("job_title", row.get("designation", ""))))
    company = row.get("company", row.get("current_company", row.get("organization", "")))
    location = row.get("location", row.get("city", ""))
    country = row.get("country", "India")
    years = float(row.get("years_of_experience", row.get("experience", row.get("yrs_exp", 0))) or 0)
    headline = row.get("headline", row.get("tagline", title))
    summary = row.get("summary", row.get("about", row.get("bio", "")))
    industry = row.get("industry", row.get("current_industry", ""))

    skills_raw = row.get("skills", row.get("key_skills", row.get("technical_skills", "")))
    skills = []
    if skills_raw:
        for s in re.split(r'[,|;]', skills_raw):
            s = s.strip()
            if s:
                skills.append({"name": s, "proficiency": "intermediate", "endorsements": 0, "duration_months": 12})

    notice = int(row.get("notice_period", row.get("notice_period_days", 30)) or 30)

    return {
        "candidate_id": cid if cid.startswith("CAND_") else generate_candidate_id(),
        "profile": {
            "anonymized_name": name,
            "headline": headline,
            "summary": summary,
            "location": location,
            "country": country,
            "years_of_experience": years,
            "current_title": title,
            "current_company": company,
            "current_company_size": row.get("company_size", "51-200"),
            "current_industry": industry
        },
        "career_history": [{
            "company": company or "Unknown",
            "title": title or "Unknown",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": int(years * 12) if years else 24,
            "is_current": True,
            "industry": industry,
            "company_size": row.get("company_size", "51-200"),
            "description": summary[:300] if summary else ""
        }],
        "education": [],
        "skills": skills[:20],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 60,
            "signup_date": "2024-01-01",
            "last_active_date": "2025-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 0,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24,
            "skill_assessment_scores": {},
            "connection_count": 0,
            "endorsements_received": 0,
            "notice_period_days": notice,
            "expected_salary_range_inr_lpa": {"min": 10, "max": 30},
            "preferred_work_mode": row.get("work_mode", "flexible"),
            "willing_to_relocate": row.get("relocate", "yes").lower() in ("yes", "true", "1"),
            "github_activity_score": -1,
            "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": -1,
            "verified_email": True,
            "verified_phone": False,
            "linkedin_connected": False
        }
    }
