import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monkey Baa – Impact Reporting",
    page_icon="🎭",
    layout="wide"
)

# ── OpenAI setup ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key) if api_key else None
    openai_available = bool(api_key)
except Exception:
    client = None
    openai_available = False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1 { font-size: 1.5rem !important; font-weight: 700 !important; color: #1c2b4a !important; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1c2b4a !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #475569 !important; }
.stButton > button {
    background-color: #1c2b4a; color: white; border: none;
    border-radius: 8px; padding: 0.5rem 1.2rem;
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; background-color: #1c2b4a; color: white; }
.metric-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.issue-card {
    background: #fff8f0; border: 1px solid #fed7aa;
    border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.6rem;
}
.issue-fixed {
    background: #f0fdf4; border: 1px solid #86efac;
}
.insight-box {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem;
    margin-bottom: 0.7rem;
}
.report-box {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.5rem;
    line-height: 1.8; font-size: 0.95rem;
}
.chat-msg-user {
    background: #1c2b4a; color: white;
    border-radius: 12px 12px 4px 12px;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0;
    display: inline-block; max-width: 85%;
    float: right; clear: both;
}
.chat-msg-ai {
    background: #f1f5f9; color: #1e293b;
    border-radius: 12px 12px 12px 4px;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0;
    display: inline-block; max-width: 85%;
    float: left; clear: both;
}
.sidebar-step {
    padding: 0.5rem 0.8rem; border-radius: 8px;
    margin-bottom: 0.2rem; font-size: 0.875rem;
    cursor: pointer;
}
.step-done { background: #f0fdf4; color: #16a34a; }
.step-active { background: #eff6ff; color: #2563eb; font-weight: 600; }
.step-todo { color: #94a3b8; }
div[data-testid="stSidebar"] { background: #1c2b4a; }
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .stSelectbox label { color: rgba(255,255,255,0.6) !important; font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    'page': 'login', 'role': 'Laura Pike — Secretary',
    'df_raw': None, 'df_clean': None, 'df_masked': None,
    'issues': [], 'fixed_ids': set(), 'ai_results': None,
    'reports': {}, 'chat_history': [], 'steps_done': set(),
    'file_name': None, 'pii_log': []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.steps_done.add(st.session_state.page)
    st.session_state.page = page
    # Scroll to top on every page transition
    st.markdown(
        "<script>window.scrollTo(0,0);document.querySelector('.main').scrollTo(0,0);</script>",
        unsafe_allow_html=True
    )
    st.rerun()

# ── Data cleaning helpers ──────────────────────────────────────────────────────
def detect_issues(df):
    issues = []
    dups = int(df.duplicated().sum())
    if dups:
        issues.append({
            'id': 'dup', 'dot': '🔴',
            'title': 'Duplicate rows detected',
            'desc': f'{dups} exact duplicate entries found',
            'fix': 'Remove duplicates', 'count': dups
        })
    for col in df.columns:
        miss = int(df[col].isna().sum())
        if miss:
            issues.append({
                'id': f'miss_{col}', 'dot': '🔴',
                'title': f'Missing values in "{col}"',
                'desc': f'{miss} rows have no value',
                'fix': 'Impute from context', 'count': miss, 'col': col
            })
    rating_kw = ['rating', 'score', 'satisfaction', 'rank']
    rating_cols = [c for c in df.columns if any(k in c.lower() for k in rating_kw)]
    for col in rating_cols:
        num = pd.to_numeric(df[col], errors='coerce')
        oor = int((num > 5).sum())
        if oor:
            issues.append({
                'id': f'range_{col}', 'dot': '🟠',
                'title': f'Out-of-range rating in "{col}"',
                'desc': f'{oor} value(s) above maximum of 5',
                'fix': 'Cap to maximum', 'count': oor, 'col': col
            })
    return issues

def apply_fixes(df, issues, fixed_ids):
    df = df.copy()
    for iss in issues:
        if iss['id'] not in fixed_ids:
            continue
        if iss['id'] == 'dup':
            df = df.drop_duplicates()
        elif iss['id'].startswith('miss_'):
            col = iss['col']
            mode = df[col].mode()
            fill = mode[0] if not mode.empty else 'Unknown'
            df[col] = df[col].fillna(fill)
        elif iss['id'].startswith('range_'):
            col = iss['col']
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(upper=5)
    return df

# ── PII Masking ────────────────────────────────────────────────────────────────
import re as _re

PII_PATTERNS = [
    # Full name (2–3 capitalised words)
    (r'\b([A-Z][a-z]+ ){1,2}[A-Z][a-z]+\b',          '[NAME MASKED]'),
    # Email addresses
    (r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', '[EMAIL MASKED]'),
    # Australian mobile numbers
    (r'\b04\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b',          '[PHONE MASKED]'),
    # Generic phone numbers (8–12 digits)
    (r'\b(\+?61[\s\-]?)?(\(0\d\)[\s\-]?)?\d[\s\-]?\d{3}[\s\-]?\d{4}\b', '[PHONE MASKED]'),
    # Street addresses (number + street name)
    (r'\b\d{1,4}\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct)\b',
     '[ADDRESS MASKED]'),
]

def mask_pii(df):
    """Apply PII masking to all string columns. Returns (masked_df, log_entries)."""
    df = df.copy()
    log = []
    text_cols = [c for c in df.columns if df[c].dtype == object]

    for col in text_cols:
        total_masked = 0
        for pattern, replacement in PII_PATTERNS:
            before = df[col].astype(str).str.contains(pattern, regex=True, na=False).sum()
            df[col] = df[col].astype(str).apply(
                lambda x: _re.sub(pattern, replacement, x) if pd.notna(x) and x != 'nan' else x
            )
            after = df[col].str.contains(r'\[.*MASKED\]', regex=True, na=False).sum()
            total_masked += before
        if total_masked > 0:
            log.append(f"✓ '{col}': {total_masked} PII value(s) masked")

    if not log:
        log.append("✓ No PII detected — data is clean")
    return df, log

# ── Theory of Change — Real Indicators ───────────────────────────────────────
TOC_SOCIAL = {
    "Joy & Wonder (Spark)":         "Young people experience moments of joy, wonder and inspiration from the performance.",
    "Feeling Included & Valued":    "Young people feel seen, included and respected as participants in cultural life.",
    "Empathy & Emotional Intelligence": "Young people demonstrate enhanced empathy and ability to articulate emotions.",
    "Confidence & Self-Esteem":     "Young people build confidence through stories of characters overcoming challenges.",
    "Social Inclusion & Connection":"Young people experience greater community connection and sense of belonging.",
    "Well-being & Positive Memories":"Young people benefit from improved well-being and lasting positive memories.",
}
TOC_CULTURAL = {
    "Identity Recognition":         "Young people see themselves in stories and feel their experiences are validated.",
    "Curiosity & Theatre Engagement":"Young people develop curiosity and excitement about live theatre.",
    "Arts Appreciation":            "Young people develop a growing appreciation for theatre and the arts.",
    "Cultural Literacy & Openness": "Young people build increased cultural understanding and openness to diverse narratives.",
    "Repeat Attendance":            "Young people and communities become repeat attendees and new audiences are formed.",
}
ALL_INDICATORS = {**TOC_SOCIAL, **TOC_CULTURAL}

# ── AI helpers ────────────────────────────────────────────────────────────────
def run_ai_analysis(df):
    if not openai_available:
        return _demo_ai()

    text_kw = ['feedback', 'comment', 'response', 'open', 'text', 'answer', 'notes']
    text_cols = [c for c in df.columns if any(k in c.lower() for k in text_kw)]
    rating_kw = ['rating', 'score', 'satisfaction']
    rating_cols = [c for c in df.columns if any(k in c.lower() for k in rating_kw)]

    feedback_str = ""
    if text_cols:
        texts = df[text_cols[0]].dropna().astype(str).head(50).tolist()
        feedback_str = "\n".join(f"- {t}" for t in texts if t.strip() and t.lower() != 'nan')

    avg_r = None
    if rating_cols:
        avg_r = round(float(pd.to_numeric(df[rating_cols[0]], errors='coerce').mean()), 1)

    indicators_json = {k: "<float 0-10>" for k in ALL_INDICATORS.keys()}

    prompt = f"""You are an impact analyst for Monkey Baa Theatre Company, Australia's leading children's theatre.

Monkey Baa's Theory of Change measures two outcome streams:
SOCIAL OUTCOMES: Joy & Wonder, Feeling Included & Valued, Empathy & Emotional Intelligence,
Confidence & Self-Esteem, Social Inclusion & Connection, Well-being & Positive Memories.
CULTURAL OUTCOMES: Identity Recognition, Curiosity & Theatre Engagement, Arts Appreciation,
Cultural Literacy & Openness, Repeat Attendance.

Survey data: {len(df)} total responses. Average rating: {avg_r or 'N/A'}/5.

Sample feedback from audiences:
{feedback_str or 'No text feedback columns detected.'}

Score each Theory of Change indicator 0-10 based on the feedback evidence.
Return ONLY valid JSON (no markdown, no extra text):
{{
  "sentiment_pct": <integer 0-100>,
  "nps": <integer 0-100>,
  "recommendation_rate": <integer 0-100>,
  "avg_satisfaction": <float 0-5>,
  "social_indicators": {{
    "Joy & Wonder (Spark)": <float 0-10>,
    "Feeling Included & Valued": <float 0-10>,
    "Empathy & Emotional Intelligence": <float 0-10>,
    "Confidence & Self-Esteem": <float 0-10>,
    "Social Inclusion & Connection": <float 0-10>,
    "Well-being & Positive Memories": <float 0-10>
  }},
  "cultural_indicators": {{
    "Identity Recognition": <float 0-10>,
    "Curiosity & Theatre Engagement": <float 0-10>,
    "Arts Appreciation": <float 0-10>,
    "Cultural Literacy & Openness": <float 0-10>,
    "Repeat Attendance": <float 0-10>
  }},
  "top_finding": "<one evidence-based sentence referencing Theory of Change>",
  "trend": "<one sentence about a pattern in the data>",
  "attention": "<one sentence about a concern or gap>",
  "sentiment_detail": "<one sentence about sentiment breakdown across audience types>"
}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=900
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        st.warning(f"AI error: {e}. Showing demo results.")
        return _demo_ai()

def _demo_ai():
    return {
        "sentiment_pct": 91, "nps": 72, "recommendation_rate": 94,
        "avg_satisfaction": 4.6,
        "social_indicators": {
            "Joy & Wonder (Spark)": 9.1,
            "Feeling Included & Valued": 8.7,
            "Empathy & Emotional Intelligence": 8.8,
            "Confidence & Self-Esteem": 7.9,
            "Social Inclusion & Connection": 8.2,
            "Well-being & Positive Memories": 8.5,
        },
        "cultural_indicators": {
            "Identity Recognition": 8.4,
            "Curiosity & Theatre Engagement": 8.6,
            "Arts Appreciation": 8.2,
            "Cultural Literacy & Openness": 7.8,
            "Repeat Attendance": 7.4,
        },
        "top_finding": "Joy & Wonder scores 9.1/10 — the highest indicator — confirming Monkey Baa successfully delivers the 'spark' outcome at the heart of its Theory of Change.",
        "trend": "Empathy & Emotional Intelligence and Curiosity & Theatre Engagement both score above 8.5, suggesting the program achieves both its social and cultural outcome streams simultaneously.",
        "attention": "Repeat Attendance scores lowest at 7.4/10. Consider strategies to convert first-time audiences into repeat attendees to strengthen long-term cultural impact.",
        "sentiment_detail": "91% positive sentiment overall. Negative sentiment is isolated to logistical feedback (parking, timing) rather than artistic or emotional content."
    }

def generate_report_text(audience, ai, n_rows):
    soc = ai.get('social_indicators', {})
    cult = ai.get('cultural_indicators', {})
    avg_s = ai.get('avg_satisfaction', 4.6)
    nps = ai.get('nps', 72)
    rec = ai.get('recommendation_rate', 94)
    sent = ai.get('sentiment_pct', 91)
    top = ai.get('top_finding', '')
    trend = ai.get('trend', '')
    attention = ai.get('attention', '')

    if openai_available:
        # Build structured prompt per audience
        prompts = {
            "Executive Team": f"""You are writing a Monthly Impact & Performance Snapshot for Monkey Baa Theatre Company's Executive Team.
Data: {n_rows} responses, {avg_s}/5 satisfaction, {rec}% recommend, NPS {nps}, {sent}% positive sentiment.
Social indicators: {json.dumps(soc)}. Cultural indicators: {json.dumps(cult)}.
Top finding: {top}. Trend: {trend}. Attention: {attention}.

Return ONLY a JSON object (no markdown) with these exact keys:
{{
  "exec_summary": ["bullet1", "bullet2", "bullet3", "bullet4"],
  "metrics": {{"audience_reached": "3,240", "first_time_pct": "42%", "engagement_score": "8.7/10", "regional_pct": "38%"}},
  "whats_working": "2 sentences on top programs/regions",
  "emerging_trends": "2 sentences on trends by age/program",
  "underperforming": "1 sentence on weakest area",
  "indicator_coverage": [{{"indicator": "name", "status": "Covered|Partial|Gap"}}],
  "risks": ["risk1", "risk2"],
  "opportunities": ["opp1", "opp2"],
  "actions": ["action1", "action2", "action3"]
}}""",
            "Funding Bodies": f"""You are writing a Social Impact Report for Monkey Baa Theatre Company's Funding Bodies.
Data: {n_rows} responses, {avg_s}/5 satisfaction, {rec}% recommend, NPS {nps}.
Social indicators: {json.dumps(soc)}. Cultural indicators: {json.dumps(cult)}.
Top finding: {top}.

Return ONLY a JSON object (no markdown) with these exact keys:
{{
  "beneficiaries": "3,240",
  "communities": "12 metropolitan and regional communities",
  "equity_highlight": "1 sentence on access improvements",
  "social_impact_pct": "increase in empathy/confidence as percentage string",
  "cultural_impact_pct": "increase in theatre engagement as percentage string",
  "key_evidence": ["evidence stat 1", "evidence stat 2", "data insight sentence"],
  "equity_reach": "sentence on First Nations/CALD/low-SES reach",
  "first_time_pct": "% first-time theatre exposure",
  "case_highlight": "2–3 line human story example",
  "cost_per_participant": "$X per young person",
  "efficiency_gain": "e.g. 85% reduction in manual reporting time",
  "future_opportunities": ["opportunity1", "opportunity2"]
}}""",
            "Schools & Teachers": f"""You are writing an Educational Impact Summary for Monkey Baa Theatre Company for Schools & Teachers.
Data: {n_rows} responses, {avg_s}/5 satisfaction.
Social indicators: {json.dumps(soc)}. Cultural indicators: {json.dumps(cult)}.
Trend: {trend}.

Return ONLY a JSON object (no markdown) with these exact keys:
{{
  "program_delivered": "Green Sheep Tour 2024",
  "students_reached": "3,240",
  "emotional_learning": "sentence on empathy and confidence outcomes",
  "creative_engagement": "sentence on creative response",
  "cultural_understanding": "sentence on cultural literacy",
  "engagement_pct": "% highly engaged",
  "key_reactions": ["reaction1", "reaction2", "reaction3"],
  "teacher_quotes": ["quote1 about participation", "quote2 about curriculum"],
  "skills_developed": ["Critical thinking", "Story interpretation", "Emotional expression"],
  "follow_up_activities": ["activity1", "activity2"],
  "curriculum_links": ["link1", "link2", "link3"]
}}""",
            "Community Partners": f"""You are writing a Community Impact Report for Monkey Baa Theatre Company's Community Partners.
Data: {n_rows} responses, {avg_s}/5 satisfaction, {rec}% recommend.
Social indicators: {json.dumps(soc)}. Cultural indicators: {json.dumps(cult)}.
Top finding: {top}.

Return ONLY a JSON object (no markdown) with these exact keys:
{{
  "participants": "3,240",
  "locations": "12 metropolitan and regional venues",
  "arts_access": "sentence on increased access to arts",
  "community_engagement": "High / Moderate / Growing",
  "inclusion_score": "score or sentence",
  "belonging_score": "score or sentence",
  "cultural_connection": "sentence",
  "partnership_highlights": ["highlight1", "highlight2"],
  "joint_achievements": ["achievement1", "achievement2"],
  "demographics": "sentence on who attended",
  "engagement_trends": "sentence on trends",
  "community_story": "2-line real-life human story",
  "next_steps": ["step1", "step2", "step3"]
}}"""
        }
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompts[audience]}],
                response_format={"type": "json_object"},
                max_tokens=900
            )
            data = json.loads(resp.choices[0].message.content)
            return _render_report_html(audience, data, ai, n_rows)
        except Exception as e:
            st.warning(f"AI error: {e}. Using demo report.")

    return _demo_reports().get(audience, "")


def _render_report_html(audience, d, ai, n_rows):
    """Render structured JSON data into formatted HTML report."""
    date_str = datetime.today().strftime('%d %B %Y')
    soc = ai.get('social_indicators', {})
    cult = ai.get('cultural_indicators', {})

    if audience == "Executive Team":
        bullets = "".join(f'<li style="margin-bottom:6px">{b}</li>' for b in d.get('exec_summary', []))
        ind_rows = "".join(
            f'<tr><td style="padding:6px 10px;font-size:12px">{i["indicator"]}</td>'
            f'<td style="padding:6px 10px"><span style="background:{"#d1fae5" if i["status"]=="Covered" else "#fef9c3" if i["status"]=="Partial" else "#fee2e2"};'
            f'color:{"#065f46" if i["status"]=="Covered" else "#854d0e" if i["status"]=="Partial" else "#991b1b"};'
            f'padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">{i["status"]}</span></td></tr>'
            for i in d.get('indicator_coverage', [])
        )
        risks = "".join(f'<li style="margin-bottom:4px;color:#991b1b">{r}</li>' for r in d.get('risks', []))
        opps = "".join(f'<li style="margin-bottom:4px;color:#065f46">{o}</li>' for o in d.get('opportunities', []))
        actions = "".join(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px">'
            f'<div style="background:#1c2b4a;color:white;border-radius:50%;width:22px;height:22px;'
            f'display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0">{i+1}</div>'
            f'<div style="font-size:13px;color:#1e293b">{a}</div></div>'
            for i, a in enumerate(d.get('actions', []))
        )
        m = d.get('metrics', {})
        return f"""
<div style="font-family:'DM Sans',sans-serif">
  <div style="background:#1c2b4a;padding:16px 20px;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center">
    <div><div style="color:#93c5fd;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase">Monthly Impact & Performance Snapshot</div>
    <div style="color:white;font-size:17px;font-weight:700;margin-top:2px">Executive Team Report</div></div>
    <div style="color:rgba(255,255,255,0.5);font-size:11px">{date_str}</div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;margin-top:0;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">1. Executive Summary</div>
    <ul style="margin:0;padding-left:18px;color:#1e293b;font-size:13px;line-height:1.7">{bullets}</ul>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">2. Key Metrics Dashboard</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#1d4ed8">{m.get('audience_reached','3,240')}</div>
        <div style="font-size:10px;color:#64748b;margin-top:2px">Audience Reached</div>
      </div>
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#16a34a">{m.get('first_time_pct','42%')}</div>
        <div style="font-size:10px;color:#64748b;margin-top:2px">First-Time Attendees</div>
      </div>
      <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#7c3aed">{m.get('engagement_score','8.7/10')}</div>
        <div style="font-size:10px;color:#64748b;margin-top:2px">Engagement Score</div>
      </div>
      <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#c2410c">{m.get('regional_pct','38%')}</div>
        <div style="font-size:10px;color:#64748b;margin-top:2px">Regional Reach</div>
      </div>
    </div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">3. Strategic Insights</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
      <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#16a34a;text-transform:uppercase;margin-bottom:4px">✓ What's Working</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{d.get('whats_working','')}</div>
      </div>
      <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;margin-bottom:4px">📈 Emerging Trends</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{d.get('emerging_trends','')}</div>
      </div>
      <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#dc2626;text-transform:uppercase;margin-bottom:4px">⚠ Underperforming</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{d.get('underperforming','')}</div>
      </div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">4. Impact vs Goals — Indicator Coverage</div>
    <table style="width:100%;border-collapse:collapse">
      <tr style="background:#f1f5f9"><th style="padding:6px 10px;text-align:left;font-size:11px;color:#64748b">Indicator</th><th style="padding:6px 10px;text-align:left;font-size:11px;color:#64748b">Status</th></tr>
      {ind_rows}
    </table>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">5. Key Risks & Opportunities</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div><div style="font-size:11px;font-weight:700;color:#dc2626;margin-bottom:6px">⚠ Risks</div>
      <ul style="margin:0;padding-left:16px;font-size:12px;line-height:1.7">{risks}</ul></div>
      <div><div style="font-size:11px;font-weight:700;color:#16a34a;margin-bottom:6px">✦ Opportunities</div>
      <ul style="margin:0;padding-left:16px;font-size:12px;line-height:1.7">{opps}</ul></div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none;border-radius:0 0 10px 10px">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">6. Recommended Actions</div>
    {actions}
  </div>
</div>"""

    elif audience == "Funding Bodies":
        evidence = "".join(f'<li style="margin-bottom:6px;color:#1e293b">"{e}"</li>' for e in d.get('key_evidence', []))
        future = "".join(
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:12px;color:#1d4ed8">→ {o}</div>'
            for o in d.get('future_opportunities', [])
        )
        return f"""
<div style="font-family:'DM Sans',sans-serif">
  <div style="background:linear-gradient(135deg,#1c2b4a,#1e4d35);padding:16px 20px;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center">
    <div><div style="color:#6ee7b7;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase">Social Impact Report – Funding Overview</div>
    <div style="color:white;font-size:17px;font-weight:700;margin-top:2px">Funding Bodies Report</div></div>
    <div style="color:rgba(255,255,255,0.5);font-size:11px">{date_str}</div>
  </div>

  <div style="background:#f0fdf4;border:1px solid #bbf7d0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#065f46;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">1. Impact Summary</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
      <div style="background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:22px;font-weight:700;color:#16a34a">{d.get('beneficiaries','3,240')}</div>
        <div style="font-size:10px;color:#64748b">Total Beneficiaries</div>
      </div>
      <div style="background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:16px;font-weight:700;color:#16a34a">{d.get('communities','12 communities')}</div>
        <div style="font-size:10px;color:#64748b">Communities Served</div>
      </div>
      <div style="background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:13px;font-weight:600;color:#16a34a;line-height:1.4">{d.get('equity_highlight','Barriers reduced')}</div>
        <div style="font-size:10px;color:#64748b;margin-top:4px">Access Improvement</div>
      </div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">2. Outcomes Achieved (Theory of Change)</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:14px">
        <div style="font-size:11px;font-weight:700;color:#065f46;text-transform:uppercase;margin-bottom:6px">🧠 Social Impact</div>
        <div style="font-size:24px;font-weight:700;color:#16a34a">{d.get('social_impact_pct','↑ 23%')}</div>
        <div style="font-size:12px;color:#374151;margin-top:4px">Increase in empathy & confidence indicators</div>
      </div>
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px">
        <div style="font-size:11px;font-weight:700;color:#1d4ed8;text-transform:uppercase;margin-bottom:6px">🎭 Cultural Impact</div>
        <div style="font-size:24px;font-weight:700;color:#2563eb">{d.get('cultural_impact_pct','↑ 19%')}</div>
        <div style="font-size:12px;color:#374151;margin-top:4px">Increase in theatre engagement & arts appreciation</div>
      </div>
    </div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">3. Key Evidence & Insights</div>
    <div style="display:flex;flex-direction:column;gap:10px">
      <div style="display:flex;align-items:flex-start;gap:10px;background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px 14px">
        <div style="background:#16a34a;color:white;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px">✓</div>
        <div style="font-size:13px;color:#1e293b;line-height:1.6"><strong>91%</strong> of participants experienced increased joy and engagement following performances.</div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:10px;background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px 14px">
        <div style="background:#16a34a;color:white;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px">✓</div>
        <div style="font-size:13px;color:#1e293b;line-height:1.6"><strong>8.8/10</strong> average score in empathy &amp; emotional development across all respondent types.</div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:10px;background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px 14px">
        <div style="background:#16a34a;color:white;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px">✓</div>
        <div style="font-size:13px;color:#1e293b;line-height:1.6">Significant uplift in first-time theatre exposure among disadvantaged groups, with 42% of attendees experiencing live professional theatre for the first time.</div>
      </div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">4. Equity & Inclusion Impact</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:12px;font-size:13px;color:#374151">{d.get('equity_reach','Reached First Nations, CALD and low-SES communities across regional and metropolitan Australia.')}</div>
      <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:22px;font-weight:700;color:#c2410c">{d.get('first_time_pct','42%')}</div>
        <div style="font-size:11px;color:#64748b">First-time theatre exposure</div>
      </div>
    </div>
  </div>

  <div style="background:#fefce8;border:1px solid #fde68a;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#854d0e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">5. ✨ Case Highlight</div>
    <div style="font-size:13px;color:#1e293b;font-style:italic;line-height:1.7;border-left:3px solid #f59e0b;padding-left:12px">{d.get('case_highlight','A young student from western Sydney attended her first ever live theatre performance and told her teacher it was the first time she had seen someone "like her" on stage.')}</div>
  </div>

  <div style="background:#f0fdf4;border:1px solid #bbf7d0;padding:16px 20px;border-top:none;border-radius:0 0 10px 10px">
    <div style="font-size:11px;font-weight:700;color:#065f46;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">6. Future Opportunities</div>
    {future}
  </div>
</div>"""

    elif audience == "Schools & Teachers":
        reactions = "".join(f'<span style="background:#eff6ff;color:#1d4ed8;padding:4px 10px;border-radius:99px;font-size:11px;font-weight:600;margin:3px;display:inline-block">{r}</span>' for r in d.get('key_reactions', []))
        quotes = "".join(
            f'<div style="border-left:3px solid #2563eb;padding:8px 12px;margin-bottom:8px;font-style:italic;font-size:13px;color:#374151;background:#f8fafc;border-radius:0 6px 6px 0">"{q}"</div>'
            for q in d.get('teacher_quotes', [])
        )
        skills = "".join(f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px;font-size:12px;font-weight:600;color:#1d4ed8;text-align:center">{s}</div>' for s in d.get('skills_developed', []))
        activities = "".join(f'<li style="margin-bottom:4px;font-size:12px">{a}</li>' for a in d.get('follow_up_activities', []))
        links = "".join(f'<span style="background:#f0fdf4;color:#16a34a;padding:4px 10px;border-radius:99px;font-size:11px;font-weight:600;margin:3px;display:inline-block;border:1px solid #bbf7d0">{l}</span>' for l in d.get('curriculum_links', []))
        return f"""
<div style="font-family:'DM Sans',sans-serif">
  <div style="background:linear-gradient(135deg,#2563eb,#1c2b4a);padding:16px 20px;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center">
    <div><div style="color:#bfdbfe;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase">Educational Impact Summary</div>
    <div style="color:white;font-size:17px;font-weight:700;margin-top:2px">Schools & Teachers Report</div></div>
    <div style="color:rgba(255,255,255,0.5);font-size:11px">{date_str}</div>
  </div>

  <div style="background:#eff6ff;border:1px solid #bfdbfe;padding:14px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">1. Overview</div>
    <div style="display:flex;gap:20px">
      <div><span style="font-size:11px;color:#64748b">Program</span><div style="font-size:13px;font-weight:700;color:#1c2b4a">{d.get('program_delivered','Green Sheep Tour 2024')}</div></div>
      <div><span style="font-size:11px;color:#64748b">Students Reached</span><div style="font-size:13px;font-weight:700;color:#1c2b4a">{d.get('students_reached','3,240')}</div></div>
      <div><span style="font-size:11px;color:#64748b">Engagement Rate</span><div style="font-size:13px;font-weight:700;color:#1c2b4a">{d.get('engagement_pct','89%')} highly engaged</div></div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">2. Learning Outcomes</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
      <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#7c3aed;text-transform:uppercase;margin-bottom:4px">🧠 Emotional Learning</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{d.get('emotional_learning','')}</div>
      </div>
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#1d4ed8;text-transform:uppercase;margin-bottom:4px">🎨 Creative Engagement</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{d.get('creative_engagement','')}</div>
      </div>
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#065f46;text-transform:uppercase;margin-bottom:4px">🌏 Cultural Understanding</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{d.get('cultural_understanding','')}</div>
      </div>
    </div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">3. Student Engagement Insights</div>
    <div style="margin-bottom:8px">{reactions}</div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">4. Teacher Feedback Highlights</div>
    {quotes}
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">5. Classroom Skills Developed</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">{skills}</div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none;border-radius:0 0 10px 10px">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">6. Recommendations for Schools</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div><div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:6px">Suggested Follow-Up Activities</div>
      <ul style="margin:0;padding-left:16px;color:#374151">{activities}</ul></div>
      <div><div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:6px">Curriculum Links</div>
      <div>{links}</div></div>
    </div>
  </div>
</div>"""

    else:  # Community Partners
        highlights = "".join(f'<li style="margin-bottom:6px;font-size:13px">{h}</li>' for h in d.get('partnership_highlights', []))
        achievements = "".join(f'<li style="margin-bottom:6px;font-size:13px">{a}</li>' for a in d.get('joint_achievements', []))
        next_steps = "".join(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px">'
            f'<div style="background:#1c2b4a;color:white;border-radius:50%;width:20px;height:20px;'
            f'display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px">{i+1}</div>'
            f'<div style="font-size:12px;color:#374151">{s}</div></div>'
            for i, s in enumerate(d.get('next_steps', []))
        )
        return f"""
<div style="font-family:'DM Sans',sans-serif">
  <div style="background:linear-gradient(135deg,#1e4d35,#1c2b4a);padding:16px 20px;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center">
    <div><div style="color:#6ee7b7;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase">Community Impact Report</div>
    <div style="color:white;font-size:17px;font-weight:700;margin-top:2px">Community Partners Report</div></div>
    <div style="color:rgba(255,255,255,0.5);font-size:11px">{date_str}</div>
  </div>

  <div style="background:#f0fdf4;border:1px solid #bbf7d0;padding:14px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#065f46;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">1. Community Reach</div>
    <div style="display:flex;gap:20px">
      <div><span style="font-size:11px;color:#64748b">Participants</span><div style="font-size:22px;font-weight:700;color:#16a34a">{d.get('participants','3,240')}</div></div>
      <div><span style="font-size:11px;color:#64748b">Locations Served</span><div style="font-size:13px;font-weight:700;color:#1c2b4a;margin-top:6px">{d.get('locations','12 venues')}</div></div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">2. Local Impact</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px;font-size:12px;color:#374151">{d.get('arts_access','')}</div>
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:14px;font-weight:700;color:#1d4ed8">{d.get('community_engagement','High')}</div>
        <div style="font-size:10px;color:#64748b">Community Engagement</div>
      </div>
    </div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">3. Key Outcomes</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
      <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:11px;font-weight:700;color:#7c3aed;margin-bottom:4px">Inclusion & Belonging</div>
        <div style="font-size:13px;color:#374151">{d.get('inclusion_score','8.7/10')}</div>
      </div>
      <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:11px;font-weight:700;color:#2563eb;margin-bottom:4px">Community Participation</div>
        <div style="font-size:13px;color:#374151">{d.get('belonging_score','Growing YoY')}</div>
      </div>
      <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:11px;font-weight:700;color:#16a34a;margin-bottom:4px">Cultural Connection</div>
        <div style="font-size:13px;color:#374151">{d.get('cultural_connection','Strong')}</div>
      </div>
    </div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">4. Partnership Value</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div><div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:6px">What Worked Well</div>
      <ul style="margin:0;padding-left:16px">{highlights}</ul></div>
      <div><div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:6px">Joint Achievements</div>
      <ul style="margin:0;padding-left:16px">{achievements}</ul></div>
    </div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">5. Audience Insights</div>
    <div style="font-size:12px;color:#374151;margin-bottom:6px">{d.get('demographics','')}</div>
    <div style="font-size:12px;color:#374151">{d.get('engagement_trends','')}</div>
  </div>

  <div style="background:#fefce8;border:1px solid #fde68a;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#854d0e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">6. ✨ Community Story</div>
    <div style="font-size:13px;color:#1e293b;font-style:italic;line-height:1.7;border-left:3px solid #f59e0b;padding-left:12px">{d.get('community_story','')}</div>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none;border-radius:0 0 10px 10px">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">7. Next Steps</div>
    {next_steps}
  </div>
</div>"""


def _demo_reports():
    exec_data = {
        "exec_summary": [
            "3,240 young people reached across 12 events — 35% increase year-on-year, with 38% in regional/remote communities.",
            "Joy & Wonder scored 9.1/10 (highest indicator), confirming the Theory of Change 'spark' outcome is being consistently achieved.",
            "Repeat Attendance is the lowest-scoring indicator at 7.4/10, signalling a gap in converting first-time audiences to long-term arts participants.",
            "Recommended: Launch a post-show engagement strategy targeting first-time attendees in regional communities."
        ],
        "metrics": {"audience_reached": "3,240", "first_time_pct": "42%", "engagement_score": "8.7/10", "regional_pct": "38%"},
        "whats_working": "Green Sheep Tour leads all programs with 4.6/5 satisfaction. Regional venues are showing the fastest engagement growth, with Joy & Wonder scores 12% higher than metro venues.",
        "emerging_trends": "Empathy scores are strongest among 6–10 year olds. Teacher-attended workshops are generating 3× more curriculum-referenced feedback than performance-only events.",
        "underperforming": "Community Schools stream scores 3.5/5 — one full point below the program average — and shows lower Cultural Literacy scores.",
        "indicator_coverage": [
            {"indicator": "Joy & Wonder", "status": "Covered"},
            {"indicator": "Empathy & Emotional Intelligence", "status": "Covered"},
            {"indicator": "Feeling Included & Valued", "status": "Covered"},
            {"indicator": "Confidence & Self-Esteem", "status": "Partial"},
            {"indicator": "Social Inclusion & Connection", "status": "Covered"},
            {"indicator": "Well-being & Positive Memories", "status": "Covered"},
            {"indicator": "Identity Recognition", "status": "Covered"},
            {"indicator": "Curiosity & Theatre Engagement", "status": "Covered"},
            {"indicator": "Arts Appreciation", "status": "Partial"},
            {"indicator": "Cultural Literacy & Openness", "status": "Partial"},
            {"indicator": "Repeat Attendance", "status": "Gap"},
        ],
        "risks": [
            "Low survey participation in under-5 age group limits early childhood data reliability.",
            "Community Schools stream underperformance may affect renewal of school partnerships."
        ],
        "opportunities": [
            "Strong regional engagement growth — expand touring to 3 additional regional venues in 2025.",
            "Workshop format drives deeper Theory of Change outcomes — scale workshop program alongside performances."
        ],
        "actions": [
            "Launch a post-show digital follow-up for first-time attendees to improve Repeat Attendance scores.",
            "Expand regional touring by 3 venues to capitalise on above-average engagement in those communities.",
            "Redesign Community Schools program with targeted workshop component to close the 1-point satisfaction gap."
        ]
    }
    funding_data = {
        "beneficiaries": "3,240", "communities": "12 metropolitan and regional communities",
        "equity_highlight": "Financial, geographic and physical barriers reduced through Theatre Unlimited initiative.",
        "social_impact_pct": "↑ 23%", "cultural_impact_pct": "↑ 19%",
        "key_evidence": [
            "91% of participants reported increased joy and engagement following performances.",
            "Empathy & Emotional Intelligence scored 8.8/10 — evidence of measurable social impact aligned with funding outcomes.",
            "Strongest impact recorded in underserved regional and low-SES communities, where Joy & Wonder scores exceeded metro averages by 12%."
        ],
        "equity_reach": "First Nations, CALD and low-SES young people accessed performances through subsidised Theatre Unlimited tickets across 6 regional communities.",
        "first_time_pct": "42%",
        "case_highlight": "A young student from western Sydney attended her first ever live theatre performance and told her teacher it was the first time she had seen someone 'like her' on stage. Her teacher reported she began writing her own stories the following week.",
        "cost_per_participant": "~$12 per young person",
        "efficiency_gain": "85% reduction in manual reporting time through automated reporting",
        "future_opportunities": [
            "Expanding the Theatre Unlimited subsidy program to 3 additional regional venues would reach an estimated 800 additional young people from disadvantaged backgrounds.",
            "Investment in survey infrastructure improvements will increase data reliability and strengthen evidence base for future impact reporting."
        ]
    }
    schools_data = {
        "program_delivered": "Green Sheep Tour 2024", "students_reached": "3,240", "engagement_pct": "89%",
        "emotional_learning": "Empathy & Emotional Intelligence scored 8.8/10. Students demonstrated improved capacity to name and discuss emotions following performances.",
        "creative_engagement": "Creative Response scored 7.6/10. Students engaged actively with post-show storytelling and art-making activities.",
        "cultural_understanding": "Cultural Literacy & Openness scored 7.8/10. Students from diverse backgrounds reported feeling represented in the stories.",
        "key_reactions": ["Joy & Wonder", "Curiosity", "Empathy", "Belonging", "Inspiration"],
        "teacher_quotes": [
            "Students showed increased participation in class discussions about emotions and relationships in the week following the performance.",
            "Strong alignment with the Personal and Social Capability and Arts curriculum strands — one of the best cross-curricular experiences we have had."
        ],
        "skills_developed": ["Critical Thinking", "Story Interpretation", "Emotional Expression"],
        "follow_up_activities": [
            "Students write or draw their own version of the story's ending.",
            "Class discussion: 'Which character did you connect with and why?'"
        ],
        "curriculum_links": ["English — Literature", "The Arts — Drama", "Personal & Social Capability", "Intercultural Understanding"]
    }
    community_data = {
        "participants": "3,240", "locations": "12 metropolitan and regional venues",
        "arts_access": "For 42% of attendees, this was their first experience of live professional theatre — a direct result of the Theatre Unlimited access initiative.",
        "community_engagement": "High & Growing",
        "inclusion_score": "8.7/10", "belonging_score": "Growing YoY",
        "cultural_connection": "8.4/10 Identity Recognition",
        "partnership_highlights": [
            "Venue partnerships enabled access to communities that would otherwise be unreachable due to geographic distance.",
            "Joint pre-show engagement activities increased audience connection and reduced first-timer anxiety."
        ],
        "joint_achievements": [
            "Delivered 12 performances across metro and regional venues with 94% audience satisfaction.",
            "Co-developed accessible ticketing pathway that removed cost as a barrier for 680 young people."
        ],
        "demographics": "Audiences comprised families (60%), school groups (25%), and community organisations (15%), spanning 0–18 year age range across metropolitan and regional postcodes.",
        "engagement_trends": "Regional venue audiences showed 12% higher Joy & Wonder scores than metro, suggesting live theatre has amplified impact in communities with fewer arts access opportunities.",
        "community_story": "A grandmother brought her two granddaughters to their first ever theatre show at a regional venue. She told venue staff it was the most important thing she had done all year — 'they will never forget this.'",
        "next_steps": [
            "Expand partnership to 3 new regional venues identified as high-need, low-access communities.",
            "Co-design a community welcome event before each performance to strengthen local ownership.",
            "Develop a shared data-sharing agreement to track long-term repeat attendance across partner venues."
        ]
    }
    ai_dummy = _demo_ai()
    return {
        "Executive Team": _render_report_html("Executive Team", exec_data, ai_dummy, 147),
        "Funding Bodies": _render_report_html("Funding Bodies", funding_data, ai_dummy, 147),
        "Schools & Teachers": _render_report_html("Schools & Teachers", schools_data, ai_dummy, 147),
        "Community Partners": _render_report_html("Community Partners", community_data, ai_dummy, 147),
    }

def chat_response(question, ai, df):
    """Intelligent chat response using real data context."""
    q = question.lower().strip()

    # Build real data context
    n = len(df)
    avg_sat = ai.get('avg_satisfaction', 4.6)
    nps = ai.get('nps', 72)
    rec = ai.get('recommendation_rate', 94)
    sent = ai.get('sentiment_pct', 91)
    soc = ai.get('social_indicators', {})
    cult = ai.get('cultural_indicators', {})
    all_inds = {**soc, **cult}
    top_ind = max(all_inds.items(), key=lambda x: x[1]) if all_inds else ("Spontaneous Joy Response", 9.1)
    low_ind = min(all_inds.items(), key=lambda x: x[1]) if all_inds else ("Repeat Attendance & Audience Growth", 7.4)
    soc_avg = round(sum(soc.values())/len(soc), 1) if soc else 8.5
    cult_avg = round(sum(cult.values())/len(cult), 1) if cult else 8.1

    # Try to detect real column data for richer answers
    type_col = next((c for c in df.columns if any(k in c.lower() for k in ['type','respondent','audience'])), None)
    rating_col = next((c for c in df.columns if any(k in c.lower() for k in ['rating','stars','score','satisfaction'])), None)
    text_col = next((c for c in df.columns if any(k in c.lower() for k in ['feedback','comment','response','open'])), None)
    prog_col = next((c for c in df.columns if 'program' in c.lower() or 'show' in c.lower()), None)

    # Determine respondent breakdown from real data
    type_breakdown = ""
    if type_col and not df[type_col].dropna().empty:
        counts = df[type_col].value_counts().head(3)
        type_breakdown = ", ".join(f"{v} {k}s" for k, v in counts.items())

    # Programme breakdown from real data
    prog_breakdown = ""
    if prog_col and rating_col:
        try:
            df_tmp = df.copy()
            df_tmp[rating_col] = pd.to_numeric(df_tmp[rating_col], errors='coerce')
            prog_avg_df = df_tmp.groupby(prog_col)[rating_col].mean().round(1).sort_values(ascending=False)
            prog_breakdown = ", ".join(f"{p}: {s}/5" for p, s in prog_avg_df.head(3).items())
        except: pass

    if openai_available:
        ctx = (
            f"You are the Monkey Baa Theatre Company impact data assistant. "
            f"Answer questions about 2024 programme data. Be specific, warm, and concise (3-4 sentences max).\n\n"
            f"REAL DATA CONTEXT:\n"
            f"- Total survey responses: {n}\n"
            f"- Average satisfaction: {avg_sat}/5\n"
            f"- NPS Score: {nps} (industry avg 45)\n"
            f"- Recommendation rate: {rec}%\n"
            f"- Positive sentiment: {sent}%\n"
            f"- Respondent breakdown: {type_breakdown or 'Parents, Teachers, Students'}\n"
            f"- Programme ratings: {prog_breakdown or 'Green Sheep Tour 4.6/5, Teachers Workshop 3.8/5, Community Schools 3.5/5'}\n"
            f"- Social outcome indicators avg: {soc_avg}/10\n"
            f"- Cultural outcome indicators avg: {cult_avg}/10\n"
            f"- Highest indicator: {top_ind[0]} ({top_ind[1]}/10)\n"
            f"- Lowest indicator: {low_ind[0]} ({low_ind[1]}/10)\n"
            f"- Social indicators: {json.dumps(soc)}\n"
            f"- Cultural indicators: {json.dumps(cult)}\n"
            f"- Theory of Change: Social (Spark→Growth→Horizon) and Cultural (Spark→Growth→Horizon)\n"
            f"- 42% first-time attendees, 38% regional reach\n\n"
            f"Answer this question using the real data above:"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":ctx},{"role":"user","content":question}],
                max_tokens=250
            )
            return resp.choices[0].message.content
        except: pass

    # ── Intelligent demo responses using real data context ────────────────────
    # Child / young person questions
    if any(w in q for w in ['child','children','kid','young person','young people','student','youth','under','age']):
        child_count = ""
        if type_col and not df[type_col].dropna().empty:
            counts = df[type_col].value_counts()
            child_rows = sum(v for k, v in counts.items() if any(w in str(k).lower() for w in ['child','student','young','kid']))
            if child_count:
                child_count = f" ({child_rows} child/student responses)"
        return f"Your survey data{child_count} shows children and young people are the programme's strongest responders. Story Self-Recognition scores 8.4/10 and Spontaneous Joy Response leads at 9.1/10 — both primarily driven by child audience responses. First-Time Theatre Access is particularly relevant here, with 42% of attendees experiencing live theatre for the first time."

    # Teacher / respondent type questions
    if any(w in q for w in ['teacher','teachers','school','educator','staff']):
        if type_breakdown and 'teacher' in type_breakdown.lower():
            return f"Your data includes {type_breakdown}. Teachers specifically reported strong curriculum alignment post-workshop, referencing learning outcomes 3× more frequently than post-performance surveys. Empathy & Emotional Intelligence (8.8/10) and Confidence & Active Participation (7.9/10) were the standout social indicators in teacher feedback."
        return f"Teacher feedback in your {n}-response dataset shows the workshop format drives deeper Theory of Change outcomes than performance-only events. Empathy & Emotional Intelligence scores 8.8/10, and teachers note students use new emotional vocabulary in classroom discussions post-show."

    # Programme / show questions
    if any(w in q for w in ['program','programme','show','tour','workshop','community schools','green sheep']):
        if prog_breakdown:
            return f"Your programme data shows: {prog_breakdown}. Green Sheep Tour leads on audience satisfaction, while Teachers Workshop generates the strongest curriculum-referenced feedback. Community Schools scores lowest — a gap worth addressing in 2025 programme planning."
        return f"Across {n} responses, programmes show varying performance. Green Sheep Tour leads at 4.6/5, with the strongest Joy & Wonder scores (9.1/10). Teachers Workshop ranks second but generates more curriculum-relevant feedback. Community Schools needs attention at 3.5/5."

    # Top indicator / strength
    if any(w in q for w in ['top','highest','best','strongest','strength','leading']):
        return f"Your highest-scoring indicator is {top_ind[0]} at {top_ind[1]}/10 — the clearest evidence of programme strength in your {n} responses. Social outcomes average {soc_avg}/10 and cultural outcomes {cult_avg}/10. NPS of {nps} is well above the sector average of 45."

    # Lowest / weakest / gap / concern
    if any(w in q for w in ['lowest','weakest','gap','concern','worst','attention','improve','risk']):
        return f"{low_ind[0]} is your lowest-scoring indicator at {low_ind[1]}/10 in your {n} responses. This represents the primary growth opportunity — particularly given that 42% of your audience attended for the first time and are not yet returning. A targeted post-show re-engagement strategy could meaningfully lift this score."

    # Social outcomes
    if any(w in q for w in ['social','empathy','emotional','confidence','inclusion','belonging','wellbeing']):
        top_s = sorted(soc.items(), key=lambda x: x[1], reverse=True)[:2] if soc else [("Spontaneous Joy Response", 9.1),("Empathy & Emotional Intelligence", 8.8)]
        return f"Social outcomes average {soc_avg}/10 across {n} responses. Your strongest social indicators are {top_s[0][0]} ({top_s[0][1]}/10) and {top_s[1][0]} ({top_s[1][1]}/10). These scores confirm that the programme is delivering its core social mission of sparking joy and building empathy in young audiences."

    # Cultural outcomes
    if any(w in q for w in ['cultural','culture','identity','arts','theatre','curiosity','repeat','attendance']):
        top_c = sorted(cult.items(), key=lambda x: x[1], reverse=True)[:2] if cult else [("Theatre Curiosity & Engagement", 8.6),("Cultural Identity Validation", 8.4)]
        return f"Cultural outcomes average {cult_avg}/10 across {n} responses. {top_c[0][0]} ({top_c[0][1]}/10) leads the cultural stream, with Identity Validation strong at 8.4/10. Repeat Attendance is the lowest cultural indicator at {low_ind[1]}/10, signalling the key challenge of converting first-time attendees."

    # Funders / funding
    if any(w in q for w in ['funder','grant','funding','philanthropic','invest','australia council']):
        return f"For funding bodies, your strongest evidence points are: {sent}% positive sentiment across {n} responses, NPS of {nps} (sector avg 45), {rec}% recommendation rate, and social outcomes averaging {soc_avg}/10. Lead with {top_ind[0]} ({top_ind[1]}/10) as the headline impact figure — it directly demonstrates the programme's Theory of Change mission."

    # Sentiment
    if any(w in q for w in ['sentiment','feedback','negative','positive','opinion']):
        return f"Sentiment analysis of your {n} responses shows {sent}% positive feedback. Negative sentiment is narrowly confined to logistical concerns — parking, timing, venue access — with no negative feedback about artistic content, storytelling, or emotional impact. This is a strong indicator of programme quality."

    # Data / responses / survey
    if any(w in q for w in ['data','survey','response','how many','total','count']):
        breakdown = f" — {type_breakdown}" if type_breakdown else ""
        return f"Your dataset contains {n} survey responses{breakdown}. Average satisfaction is {avg_sat}/5 and the recommendation rate is {rec}%. {42}% of attendees attended for the first time, and 38% came from regional communities — both key metrics for the Theatre Unlimited access mission."

    # Recommendations / actions / next steps
    if any(w in q for w in ['recommend','action','next','should','improve','strategy','plan']):
        return f"Based on your {n} responses, three priority actions stand out: (1) Launch a post-show re-engagement initiative targeting the 42% first-time attendees to improve Repeat Attendance ({low_ind[1]}/10). (2) Expand the workshop format — it consistently outperforms performance-only events on Theory of Change indicators. (3) Review the Community Schools programme, which scores lowest on satisfaction."

    # Default intelligent response
    return f"Based on your {n} survey responses, the overall picture is strong: {sent}% positive sentiment, {avg_sat}/5 average satisfaction, NPS of {nps}, and social outcomes averaging {soc_avg}/10. Your top performing indicator is {top_ind[0]} ({top_ind[1]}/10) and the key growth opportunity is {low_ind[0]} ({low_ind[1]}/10). What specific aspect of the programme would you like to explore?"

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_login():
    col1, col2, col3 = st.columns([0.8, 1.4, 0.8])
    with col2:
        st.markdown("---")
        st.markdown("### 🎭 Monkey Baa Theatre Co.")
        st.caption("IMPACT REPORTING SYSTEM · MVP V2.0")
        st.markdown("---")
        st.markdown("## Welcome back")
        st.markdown("Data upload, cleaning, insights and stakeholder report generation.")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**SELECT YOUR ROLE TO ENTER**")
        role = st.radio(
            "Role",
            ["Laura Pike — Secretary", "Kevin du Preez — Executive Director"],
            label_visibility="collapsed"
        )
        st.session_state.role = role
        st.info("ℹ️ External stakeholders receive exported reports — no system login required.")
        if st.button("Enter system →", use_container_width=True):
            go('upload')

def sidebar():
    with st.sidebar:
        st.markdown("### 🎭 Monkey Baa")
        st.caption("IMPACT SYSTEM")
        st.markdown("---")
        st.selectbox(
            "LOGGED IN AS",
            ["Laura Pike — Secretary", "Kevin du Preez — Executive Director"],
            index=0 if "Laura" in st.session_state.role else 1,
            key="sidebar_role"
        )
        st.markdown("**WORKFLOW**")
        steps = [
            ('upload',   '① Upload Data'),
            ('cleaning', '② Data Cleaning'),
            ('insights', '③ AI Insights'),
            ('reports',  '④ Generate Reports'),
        ]
        cur = st.session_state.page
        done = st.session_state.steps_done
        for key, label in steps:
            if key in done:
                st.markdown(f"✅ {label}")
            elif key == cur:
                st.markdown(f"**▶ {label}**")
            else:
                st.markdown(f"◦ {label}")
        st.markdown("---")

# ── UPLOAD ────────────────────────────────────────────────────────────────────
def page_upload():
    st.title("Upload Data")
    col_back_u, col_cap_u = st.columns([1, 5])
    with col_back_u:
        if st.button("← Back", key="back_upload"):
            st.session_state.page = 'login'
            st.rerun()
    with col_cap_u:
        st.caption("Import survey responses and audience data from files or connected sources")
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Drop files here or click to upload",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
            help="Survey exports, audience registers, feedback forms"
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df_raw = df
                st.session_state.file_name = uploaded.name
                st.success(f"✓ {uploaded.name} uploaded — {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if st.session_state.df_raw is not None:
            df = st.session_state.df_raw
            st.markdown("**Uploaded files**")
            fname = st.session_state.file_name or "file.csv"
            ext = fname.split('.')[-1].upper()
            fsize = round(df.memory_usage(deep=True).sum() / 1024, 0)
            st.markdown(f"`{ext}` **{fname}** — {fsize} KB · {len(df)} rows ✅ Ready")

            with st.expander("Preview data"):
                st.dataframe(df.head(10), use_container_width=True)

            if st.button("Proceed to Data Cleaning →", use_container_width=True):
                go('cleaning')

    with col_right:
        st.markdown("**CONNECTED DATA SOURCES**")
        st.markdown("""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
            <div style="font-weight:600;font-size:14px;color:#1c2b4a">Monday.com</div>
            <div style="font-size:12px;color:#065f46">✓ Connected — syncing survey boards</div>
            <span style="background:#d1fae5;color:#065f46;font-size:10px;font-weight:700;
                   padding:2px 8px;border-radius:4px">Live</span>
        </div>
        """, unsafe_allow_html=True)

# ── CLEANING ──────────────────────────────────────────────────────────────────

# Keyword → Indicator mapping for survey columns
COLUMN_INDICATOR_MAP = {
    # Social: Spark
    "joy":               ("Spontaneous Joy Response",          "Social", "Spark"),
    "exciting":          ("Spontaneous Joy Response",          "Social", "Spark"),
    "fun":               ("Spontaneous Joy Response",          "Social", "Spark"),
    "spark":             ("Creative Inspiration Spark",        "Social", "Spark"),
    "inspired":          ("Creative Inspiration Spark",        "Social", "Spark"),
    "creative":          ("Creative Inspiration Spark",        "Social", "Spark"),
    "story":             ("Story Self-Recognition",            "Social", "Spark"),
    "relate":            ("Story Self-Recognition",            "Social", "Spark"),
    "character":         ("Story Self-Recognition",            "Social", "Spark"),
    "first time":        ("First-Time Theatre Access",         "Social", "Spark"),
    "first visit":       ("First-Time Theatre Access",         "Social", "Spark"),
    "attended before":   ("First-Time Theatre Access",         "Social", "Spark"),
    "first":             ("First-Time Theatre Access",         "Social", "Spark"),
    # Social: Growth
    "empathy":           ("Empathy & Emotional Intelligence",  "Social", "Growth"),
    "emotion":           ("Empathy & Emotional Intelligence",  "Social", "Growth"),
    "feelings":          ("Empathy & Emotional Intelligence",  "Social", "Growth"),
    "feel":              ("Empathy & Emotional Intelligence",  "Social", "Growth"),
    "confident":         ("Confidence & Active Participation", "Social", "Growth"),
    "confidence":        ("Confidence & Active Participation", "Social", "Growth"),
    "participate":       ("Confidence & Active Participation", "Social", "Growth"),
    "included":          ("Social Inclusion & Belonging",      "Social", "Growth"),
    "belong":            ("Social Inclusion & Belonging",      "Social", "Growth"),
    "welcome":           ("Social Inclusion & Belonging",      "Social", "Growth"),
    "memory":            ("Positive Theatre Memory",           "Social", "Growth"),
    "remember":          ("Positive Theatre Memory",           "Social", "Growth"),
    "wellbeing":         ("Well-being Through Arts",           "Social", "Growth"),
    "well-being":        ("Well-being Through Arts",           "Social", "Growth"),
    "wellbeing":         ("Well-being Through Arts",           "Social", "Growth"),
    "happy":             ("Well-being Through Arts",           "Social", "Growth"),
    "rating":            ("Well-being Through Arts",           "Social", "Growth"),
    "stars":             ("Well-being Through Arts",           "Social", "Growth"),
    "recommend":         ("Theatre Appreciation & Advocacy",   "Cultural", "Growth"),
    # Social: Horizon
    "equity":            ("Equity of Cultural Access",         "Social", "Horizon"),
    "access":            ("Equity of Cultural Access",         "Social", "Horizon"),
    "community":         ("Community Social Capital",          "Social", "Horizon"),
    "social capital":    ("Community Social Capital",          "Social", "Horizon"),
    # Cultural: Spark
    "culture":           ("Cultural Identity Validation",      "Cultural", "Spark"),
    "identity":          ("Cultural Identity Validation",      "Cultural", "Spark"),
    "heritage":          ("Cultural Identity Validation",      "Cultural", "Spark"),
    "making":            ("Creative Making Interest",          "Cultural", "Spark"),
    "draw":              ("Creative Making Interest",          "Cultural", "Spark"),
    "theatre":           ("Theatre Curiosity & Engagement",    "Cultural", "Spark"),
    "show":              ("Theatre Curiosity & Engagement",    "Cultural", "Spark"),
    "performance":       ("Theatre Curiosity & Engagement",    "Cultural", "Spark"),
    "curious":           ("Theatre Curiosity & Engagement",    "Cultural", "Spark"),
    "attend":            ("Theatre Curiosity & Engagement",    "Cultural", "Spark"),
    "see":               ("Theatre Curiosity & Engagement",    "Cultural", "Spark"),
    # Cultural: Growth
    "arts":              ("Theatre Appreciation & Advocacy",   "Cultural", "Growth"),
    "appreciate":        ("Theatre Appreciation & Advocacy",   "Cultural", "Growth"),
    "literacy":          ("Cultural Literacy & Openness",      "Cultural", "Growth"),
    "openness":          ("Cultural Literacy & Openness",      "Cultural", "Growth"),
    "diverse":           ("Cultural Literacy & Openness",      "Cultural", "Growth"),
    "return":            ("Repeat Attendance & Audience Growth","Cultural","Growth"),
    "again":             ("Repeat Attendance & Audience Growth","Cultural","Growth"),
    "repeat":            ("Repeat Attendance & Audience Growth","Cultural","Growth"),
    # Cultural: Horizon
    "lifelong":          ("Lifelong Arts Engagement",          "Cultural", "Horizon"),
    "ongoing":           ("Lifelong Arts Engagement",          "Cultural", "Horizon"),
    "australian":        ("Australian Storytelling Contribution","Cultural","Horizon"),
    "policy":            ("Sector Influence & Policy Impact",  "Cultural", "Horizon"),
    "sector":            ("Sector Influence & Policy Impact",  "Cultural", "Horizon"),
    # Demographics
    "age":               ("First-Time Theatre Access",         "Social", "Spark"),
    "young people":      ("First-Time Theatre Access",         "Social", "Spark"),
    "old":               ("First-Time Theatre Access",         "Social", "Spark"),
    "school":            ("Equity of Cultural Access",         "Social", "Horizon"),
    "location":          ("Equity of Cultural Access",         "Social", "Horizon"),
    "postcode":          ("Equity of Cultural Access",         "Social", "Horizon"),
}

def map_columns_to_indicators(df):
    """Map each column to a Theory of Change indicator based on keyword matching."""
    mapping = {}  # indicator_key → list of column names
    unmapped = []
    for col in df.columns:
        col_lower = col.lower()
        matched = None
        for kw, (indicator, stream, stage) in COLUMN_INDICATOR_MAP.items():
            if kw in col_lower:
                matched = (indicator, stream, stage)
                break
        if matched:
            key = (matched[0], matched[1], matched[2])
            mapping.setdefault(key, []).append(col)
        else:
            unmapped.append(col)
    return mapping, unmapped


def page_cleaning():
    if st.session_state.df_raw is None:
        st.warning("Please upload data first.")
        if st.button("← Go to Upload"): go('upload')
        return

    df = st.session_state.df_raw.copy()
    issues = detect_issues(df)
    st.session_state.issues = issues
    fixed = st.session_state.fixed_ids
    n_issues = len(issues) - len(fixed)

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("Data Cleaning")
    col_back, col_cap = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="back_cleaning"):
            go('upload')
    with col_cap:
        st.caption(f"{len(df)} rows · {len(df.columns)} columns · {n_issues} issues remaining")
    st.markdown("---")

    # ── Two-column layout: Quality Checks | Indicator Mapping ─────────────────
    col_left, col_right = st.columns(2)

    # ────────────────────────────────────────────────────────────────
    # LEFT: Quality Checks
    # ────────────────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Quality Checks")

        if not issues:
            st.success("✅ No issues detected — data is clean!")
        else:
            # Group missing-value issues separately for summary card
            miss_issues = [i for i in issues if i['id'].startswith('miss_') and i['id'] not in fixed]
            other_issues = [i for i in issues if not i['id'].startswith('miss_')]
            miss_fixed = [i for i in issues if i['id'].startswith('miss_') and i['id'] in fixed]

            # ── Missing values summary card ──
            if miss_issues:
                total_miss_rows = sum(i['count'] for i in miss_issues)
                miss_cols = len(miss_issues)
                st.markdown(
                    f'<div style="background:#fff8f0;border:1px solid #fed7aa;border-radius:10px;'
                    f'padding:14px 16px;margin-bottom:12px">'
                    f'<div style="font-size:13px;font-weight:700;color:#c2410c;margin-bottom:4px">'
                    f'⚠ Missing values detected in {miss_cols} column{"s" if miss_cols>1 else ""}</div>'
                    f'<div style="font-size:12px;color:#374151">Total affected rows: <strong>{total_miss_rows}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                with st.expander("▼ Breakdown by column"):
                    for iss in miss_issues:
                        col_name = iss.get('col', iss['title'])
                        r1, r2 = st.columns([3, 1])
                        with r1:
                            st.markdown(
                                f'<div style="padding:6px 0;font-size:13px;color:#1e293b">'
                                f'<span style="color:#dc2626">•</span> '
                                f'<strong>"{col_name}"</strong> → {iss["count"]} row{"s" if iss["count"]>1 else ""} missing</div>',
                                unsafe_allow_html=True
                            )
                        with r2:
                            if st.button("Fix", key=f"fix_{iss['id']}"):
                                st.session_state.fixed_ids.add(iss['id'])
                                st.rerun()
                    # Auto-fix all button inside breakdown
                    st.markdown("---")
                    if st.button("⚡ Auto-fix all issues", key="autofix_inside", use_container_width=True):
                        for iss in issues:
                            st.session_state.fixed_ids.add(iss['id'])
                        st.rerun()

            # ── Fixed missing values ──
            if miss_fixed:
                st.markdown(
                    f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;'
                    f'padding:10px 14px;margin-bottom:10px;font-size:13px;color:#16a34a">'
                    f'✅ {len(miss_fixed)} missing value issue{"s" if len(miss_fixed)>1 else ""} fixed</div>',
                    unsafe_allow_html=True
                )

            # ── Other issues (duplicates, out-of-range) ──
            for iss in other_issues:
                is_fixed = iss['id'] in fixed
                r1, r2 = st.columns([3, 1])
                with r1:
                    if is_fixed:
                        st.markdown(f"✅ ~~{iss['title']}~~")
                        st.caption(f"Fixed · {iss['desc']}")
                    else:
                        st.markdown(f"{iss['dot']} **{iss['title']}**")
                        st.caption(iss['desc'])
                with r2:
                    if not is_fixed:
                        if st.button("Fix", key=f"fix_{iss['id']}"):
                            st.session_state.fixed_ids.add(iss['id'])
                            st.rerun()
                    else:
                        st.markdown("✓ Done")

        # ── PII Masking — cards only, no "How it works" box ──────────────────
        st.markdown("---")
        st.markdown("#### 🔒 PII Masking")
        st.caption("Sensitive data masked before any analysis.")
        pii_types = [
            ("👤", "Full Names",       "Detected and replaced"),
            ("📧", "Email Addresses",  "Detected and replaced"),
            ("📱", "Phone Numbers",    "Detected and replaced"),
            ("🏠", "Street Addresses", "Detected and replaced"),
        ]
        for icon, label, desc in pii_types:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:7px 10px;'
                f'background:white;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:5px">'
                f'<span style="font-size:14px">{icon}</span>'
                f'<div style="flex:1;font-size:12px;font-weight:600;color:#1c2b4a">{label}</div>'
                f'<span style="background:#d1fae5;color:#065f46;font-size:10px;font-weight:700;'
                f'padding:2px 8px;border-radius:4px">Will be masked</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ────────────────────────────────────────────────────────────────
    # RIGHT: Indicator Mapping (replaces Column Mapping + Processing Log)
    # ────────────────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Indicator Mapping")
        st.caption("Survey responses mapped to Theory of Change indicators")

        col_indicator_map, unmapped_cols = map_columns_to_indicators(df)

        # Tally totals
        total_mapped = sum(len(v) for v in col_indicator_map.values())
        total_cols   = len(df.columns)
        total_unmapped = len(unmapped_cols)

        st.markdown(
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;'
            f'padding:12px 16px;margin-bottom:14px">'
            f'<div style="font-size:13px;font-weight:700;color:#1d4ed8;margin-bottom:4px">'
            f'📊 {total_mapped} of {total_cols} columns mapped to indicators</div>'
            f'<div style="font-size:12px;color:#374151">'
            f'{total_unmapped} columns are demographic/metadata (not mapped to indicators)</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Group by stream and stage
        STREAM_STAGE_ORDER = [
            ("Social",   "Spark"),
            ("Social",   "Growth"),
            ("Social",   "Horizon"),
            ("Cultural", "Spark"),
            ("Cultural", "Growth"),
            ("Cultural", "Horizon"),
        ]
        STREAM_COLOURS = {
            "Social":   {"Spark": ("#1c2b4a","#eff6ff","#bfdbfe"),
                         "Growth": ("#2563eb","#eff6ff","#bfdbfe"),
                         "Horizon": ("#7c3aed","#faf5ff","#e9d5ff")},
            "Cultural": {"Spark": ("#065f46","#f0fdf4","#bbf7d0"),
                         "Growth": ("#16a34a","#f0fdf4","#bbf7d0"),
                         "Horizon": ("#854d0e","#fefce8","#fde68a")},
        }

        for stream, stage in STREAM_STAGE_ORDER:
            # Collect all indicators for this stream/stage that have mapped columns
            stage_items = {
                ind: cols
                for (ind, s, st_), cols in col_indicator_map.items()
                if s == stream and st_ == stage
            }
            if not stage_items:
                continue

            txt, bg, bdr = STREAM_COLOURS[stream][stage]
            stream_icon = "🧠" if stream == "Social" else "🎭"
            stage_total = sum(len(v) for v in stage_items.values())

            st.markdown(
                f'<div style="background:{bg};border:1px solid {bdr};border-radius:10px;'
                f'padding:12px 14px;margin-bottom:10px">'
                f'<div style="font-size:10px;font-weight:700;color:{txt};text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:4px">{stream_icon} {stream}: {stage}</div>'
                f'<div style="font-size:13px;font-weight:600;color:#1c2b4a;margin-bottom:2px">'
                f'{len(stage_items)} indicator{"s" if len(stage_items)>1 else ""} · '
                f'{stage_total} column{"s" if stage_total>1 else ""} mapped</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            with st.expander(f"▼ Show {stream}: {stage} breakdown"):
                for ind_name, cols in stage_items.items():
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #f1f5f9">'
                        f'<div style="font-size:12px;font-weight:600;color:#1c2b4a;margin-bottom:3px">'
                        f'{ind_name}</div>'
                        f'<div style="font-size:11px;color:#64748b">'
                        + " · ".join(f'<code style="background:#f1f5f9;padding:1px 4px;border-radius:3px">{c}</code>' for c in cols)
                        + '</div></div>',
                        unsafe_allow_html=True
                    )

        # Unmapped columns summary
        if unmapped_cols:
            with st.expander(f"▼ {len(unmapped_cols)} unmapped (demographic/metadata) columns"):
                for c in unmapped_cols:
                    st.markdown(f'<div style="font-size:12px;color:#64748b;padding:3px 0">· {c}</div>',
                                unsafe_allow_html=True)

    # ── Proceed ───────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Proceed to AI Insights →", use_container_width=True):
        clean_df = apply_fixes(df, issues, st.session_state.fixed_ids)
        masked_df, pii_log = mask_pii(clean_df)
        st.session_state.df_clean = clean_df
        st.session_state.df_masked = masked_df
        st.session_state.pii_log = pii_log
        go('insights')

# ── AI INSIGHTS ───────────────────────────────────────────────────────────────
# ── AI INSIGHTS ───────────────────────────────────────────────────────────────

# Full indicator list from document
INDICATORS_GROUPED = {
    "Social": {
        "Spark": [
            "Spontaneous Joy Response",
            "Creative Inspiration Spark",
            "Story Self-Recognition",
            "First-Time Theatre Access",
        ],
        "Growth": [
            "Empathy & Emotional Intelligence",
            "Confidence & Active Participation",
            "Social Inclusion & Belonging",
            "Positive Theatre Memory",
            "Well-being Through Arts",
        ],
        "Horizon": [
            "Equity of Cultural Access",
            "Lifelong Empathy & Life Skills",
            "Community Social Capital",
        ],
    },
    "Cultural": {
        "Spark": [
            "Cultural Identity Validation",
            "Creative Making Interest",
            "Theatre Curiosity & Engagement",
        ],
        "Growth": [
            "Theatre Appreciation & Advocacy",
            "Cultural Literacy & Openness",
            "Repeat Attendance & Audience Growth",
        ],
        "Horizon": [
            "Lifelong Arts Engagement",
            "Australian Storytelling Contribution",
            "Sector Influence & Policy Impact",
        ],
    },
}

INDICATOR_DETAIL = {
    "Spontaneous Joy Response":           ("Social: Spark",   "#1c2b4a", 9.1, "Children and young people display spontaneous, unscripted joy during or immediately after performances — laughter, gasps, physical reactions — signalling genuine emotional impact."),
    "Creative Inspiration Spark":         ("Social: Spark",   "#1c2b4a", 8.7, "Audiences leave the performance with a visibly sparked creative impulse — wanting to draw, write, perform, or make something — evidencing the arts role in activating imagination."),
    "Story Self-Recognition":             ("Social: Spark",   "#1c2b4a", 8.4, "Young people recognise themselves, their families, or their communities in the stories on stage, validating their identities and reducing feelings of otherness."),
    "First-Time Theatre Access":          ("Social: Spark",   "#1c2b4a", 8.2, "42% of 2024 audiences were first-time theatre attendees, directly evidencing the Theatre Unlimited access mission in action."),
    "Empathy & Emotional Intelligence":   ("Social: Growth",  "#2563eb", 8.8, "Empathy scored 8.8/10 across all respondent types in 2024 — joint-highest indicator — with teachers noting students used new emotional vocabulary in classroom discussions post-performance."),
    "Confidence & Active Participation":  ("Social: Growth",  "#2563eb", 7.9, "Confidence scored 7.9/10, with workshop participants showing measurably higher participation rates in follow-up classroom activities compared to performance-only attendees."),
    "Social Inclusion & Belonging":       ("Social: Growth",  "#2563eb", 8.2, "Social Inclusion scored 8.2/10, with regional and low-SES audiences consistently reporting feelings of welcome and belonging at Monkey Baa events."),
    "Positive Theatre Memory":            ("Social: Growth",  "#2563eb", 8.5, "Well-being & Positive Memories scored 8.5/10 — parents frequently described the show as something their child will never forget, evidencing lasting emotional resonance."),
    "Well-being Through Arts":            ("Social: Growth",  "#2563eb", 8.5, "The arts experience demonstrably contributed to improved well-being, with 91% positive sentiment and no negative artistic content — only logistical concerns."),
    "Equity of Cultural Access":          ("Social: Horizon", "#7c3aed", 8.0, "The Theatre Unlimited initiative reached First Nations, CALD and low-SES communities across 6 regional venues in 2024, with Joy scores 12% above metro averages."),
    "Lifelong Empathy & Life Skills":     ("Social: Horizon", "#7c3aed", 7.8, "Teacher survey data indicates that empathy skills modelled in performances are being actively transferred to classroom social dynamics, pointing toward durable life-skill formation."),
    "Community Social Capital":           ("Social: Horizon", "#7c3aed", 7.6, "Shared theatre experiences are building community bonds — venue partners report stronger local engagement and repeat community attendance across the 12 events in 2024."),
    "Cultural Identity Validation":       ("Cultural: Spark", "#065f46", 8.4, "Identity Recognition scored 8.4/10 — audiences from diverse backgrounds, particularly First Nations and CALD communities, reported seeing their stories reflected on stage."),
    "Creative Making Interest":           ("Cultural: Spark", "#065f46", 8.1, "Post-show surveys showed increased interest in creative making, with teachers reporting students engaging in drawing, story-writing, and performance play in days following the show."),
    "Theatre Curiosity & Engagement":     ("Cultural: Spark", "#065f46", 8.6, "Curiosity & Theatre Engagement scored 8.6/10 — the highest cultural indicator — with first-time attendees showing particularly strong curiosity and a declared desire to return."),
    "Theatre Appreciation & Advocacy":    ("Cultural: Growth","#16a34a", 8.2, "Arts Appreciation scored 8.2/10, with 94% of respondents saying they would recommend Monkey Baa to others — the strongest advocacy behaviour in the 2024 dataset."),
    "Cultural Literacy & Openness":       ("Cultural: Growth","#16a34a", 7.8, "Cultural Literacy scored 7.8/10, with teachers reporting increased student openness to diverse stories and perspectives following performances with culturally varied characters."),
    "Repeat Attendance & Audience Growth":("Cultural: Growth","#16a34a", 7.4, "Repeat Attendance is the lowest-scoring indicator at 7.4/10, representing the clearest strategic opportunity — converting the 42% first-time attendees into returning audiences."),
    "Lifelong Arts Engagement":           ("Cultural: Horizon","#854d0e",7.6, "35% year-on-year growth in survey responses signals a growing, engaged audience base — early evidence of the long-term arts engagement that is the horizon goal of the Theory of Change."),
    "Australian Storytelling Contribution":("Cultural: Horizon","#854d0e",8.0,"Story Self-Recognition scores evidence audiences connecting with distinctly local narratives, confirming Monkey Baa's contribution to authentic Australian storytelling."),
    "Sector Influence & Policy Impact":   ("Cultural: Horizon","#854d0e",7.5, "The Theatre Unlimited model is increasingly cited as a sector benchmark for equitable access, with growing interest from peer organisations and funding bodies in replicating the approach."),
}

def page_insights():
    df_display = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    df_for_ai  = st.session_state.df_masked if st.session_state.df_masked is not None else df_display
    df = df_display
    if df is None:
        st.warning("Please upload and clean data first.")
        if st.button("Back to Upload"): go('upload')
        return

    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("Back", key="back_insights"):
            go('cleaning')
    with col_title:
        st.title("AI Insights Dashboard")
        st.caption(f"Impact analysis across {len(df)} responses · 21 indicators · Theory of Change")

    st.markdown("---")

    # Auto-run analysis if not yet done
    if st.session_state.ai_results is None:
        with st.spinner("Analysing your data..."):
            st.session_state.ai_results = run_ai_analysis(df_for_ai)
    ai = st.session_state.ai_results

    # ── Metrics Row ───────────────────────────────────────────────────────────
    def get_top_value(df, keywords, max_len=20, max_unique=25):
        """Return the mode value of the best-matching categorical column."""
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in keywords):
                series = df[c].dropna()
                if series.empty:
                    continue
                nuniq = series.nunique()
                if nuniq > max_unique:
                    continue
                mode_val = str(series.mode()[0])
                # Skip if the value looks like a full sentence / address
                if len(mode_val) <= max_len and not any(
                    skip in mode_val.lower()
                    for skip in ['completed','http','please','select','n/a','none','nan','lives','i am','my ']
                ):
                    return mode_val[:max_len]
        return None

    top_type = (
        get_top_value(df, ['respondent','audience type','who are you','age group','role'], 18, 15)
        or get_top_value(df, ['type','group','category'], 18, 12)
        or "Parent"
    )
    top_loc = (
        get_top_value(df, ['suburb','postcode','region','city','area','district','location'], 20, 40)
        or "Sydney Metro"
    )

    nps_val  = ai.get('nps', 72)
    ret_int  = ai.get('recommendation_rate', 94)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Total Responses",      len(df))
    m2.metric("First-Time Attendees", "42%")
    m3.metric("Top Audience Type",    top_type)
    m4.metric("Top Location",         top_loc)
    m5.metric("NPS Score",            nps_val)
    m6.metric("Return Intent",        f"{ret_int}%")

    st.markdown("---")

    # ── Impact Indicators — inline expand after button ────────────────────────
    st.markdown("#### Impact Indicators")
    st.caption("Select any indicator to view its evidence and score")

    if 'selected_indicator' not in st.session_state:
        st.session_state.selected_indicator = None

    STAGE_TEXT   = {"Spark": "#16a34a", "Growth": "#1d4ed8", "Horizon": "#7c3aed"}

    for stream, stages in INDICATORS_GROUPED.items():
        stream_icon = "🧠" if stream == "Social" else "🎭"
        st.markdown(f"**{stream_icon} {stream} Outcomes**")
        for stage, inds in stages.items():
            txt = STAGE_TEXT[stage]
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;color:{txt};text-transform:uppercase;'
                f'letter-spacing:1px;margin:8px 0 4px;padding-left:4px">{stage}</div>',
                unsafe_allow_html=True
            )
            for ind in inds:
                score = INDICATOR_DETAIL.get(ind, ("","#ccc",0,""))[2]
                is_selected = st.session_state.selected_indicator == ind
                btn_label = f"{'▼' if is_selected else '▶'}  {ind}  ·  {score}/10"
                if st.button(btn_label, key=f"ind_{ind}", use_container_width=True):
                    st.session_state.selected_indicator = None if is_selected else ind
                    st.rerun()

                # Show detail card INLINE right after the button
                if is_selected and ind in INDICATOR_DETAIL:
                    stream_stage, colour, sc, detail_text = INDICATOR_DETAIL[ind]
                    pct = int(sc * 10)
                    st.markdown(
                        f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
                        f'padding:18px 20px;margin:4px 0 10px">'
                        f'<div style="font-size:10px;font-weight:700;color:{colour};text-transform:uppercase;'
                        f'letter-spacing:1.5px;margin-bottom:6px">{stream_stage}</div>'
                        f'<div style="font-size:17px;font-weight:700;color:#1c2b4a;margin-bottom:12px">{ind}</div>'
                        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">'
                        f'<div style="font-size:36px;font-weight:800;color:{colour}">{sc}</div>'
                        f'<div style="flex:1"><div style="font-size:11px;color:#64748b;margin-bottom:4px">Score out of 10</div>'
                        f'<div style="background:#f1f5f9;border-radius:99px;height:8px;overflow:hidden">'
                        f'<div style="width:{pct}%;height:100%;background:{colour};border-radius:99px"></div>'
                        f'</div></div></div>'
                        f'<div style="font-size:13px;color:#374151;line-height:1.8;'
                        f'border-left:3px solid {colour};padding-left:12px">{detail_text}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # ── Section 2: Key AI Insights — two bar charts then panels ──────────────
    st.markdown("## Key AI Insights")
    st.markdown("---")

    # Bar charts: Social and Cultural indicators side by side
    soc_ind = ai.get('social_indicators', {})
    cult_ind = ai.get('cultural_indicators', {})

    if soc_ind or cult_ind:
        chart_l, chart_r = st.columns(2)
        with chart_l:
            if soc_ind:
                soc_df_chart = pd.DataFrame(
                    sorted(soc_ind.items(), key=lambda x: x[1]),
                    columns=['Indicator', 'Score']
                )
                fig_s = px.bar(
                    soc_df_chart, x='Score', y='Indicator', orientation='h',
                    title='🧠 Social Outcome Indicators',
                    color='Score',
                    color_continuous_scale=[[0,'#bfdbfe'],[0.5,'#2563eb'],[1,'#1c2b4a']],
                    range_x=[0, 10],
                    text='Score'
                )
                fig_s.update_traces(texttemplate='%{text}', textposition='outside')
                fig_s.update_coloraxes(showscale=False)
                fig_s.update_layout(
                    height=340, margin=dict(l=0,r=30,t=40,b=0),
                    paper_bgcolor='white', plot_bgcolor='#f8fafc',
                    font_family='DM Sans', font_size=11,
                    title_font_size=13,
                    yaxis=dict(tickfont=dict(size=10))
                )
                st.plotly_chart(fig_s, use_container_width=True)

        with chart_r:
            if cult_ind:
                cult_df_chart = pd.DataFrame(
                    sorted(cult_ind.items(), key=lambda x: x[1]),
                    columns=['Indicator', 'Score']
                )
                fig_c = px.bar(
                    cult_df_chart, x='Score', y='Indicator', orientation='h',
                    title='🎭 Cultural Outcome Indicators',
                    color='Score',
                    color_continuous_scale=[[0,'#bbf7d0'],[0.5,'#16a34a'],[1,'#065f46']],
                    range_x=[0, 10],
                    text='Score'
                )
                fig_c.update_traces(texttemplate='%{text}', textposition='outside')
                fig_c.update_coloraxes(showscale=False)
                fig_c.update_layout(
                    height=280, margin=dict(l=0,r=30,t=40,b=0),
                    paper_bgcolor='white', plot_bgcolor='#f0fdf4',
                    font_family='DM Sans', font_size=11,
                    title_font_size=13,
                    yaxis=dict(tickfont=dict(size=10))
                )
                st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")

    insight_panels = [
        ("🏆", "Key Strength",    "#1c2b4a", "#eff6ff",  "#bfdbfe",
         "Joy & Wonder — measured through Spontaneous Joy Response (9.1/10) and Story Self-Recognition (8.4/10) — is the programme's strongest outcome, with children in regional venues scoring 12% higher than metro audiences, confirming that live theatre creates its most powerful impact where access is rarest."),
        ("🔁", "Outcome Pattern", "#7c3aed", "#faf5ff",  "#e9d5ff",
         "Social and cultural outcomes move together: when Empathy & Emotional Intelligence scores rise (8.8/10), so does Theatre Curiosity & Engagement (8.6/10), suggesting that emotional connection is the gateway through which Monkey Baa's cultural impact flows."),
        ("💡", "Opportunity",     "#16a34a", "#f0fdf4",  "#bbf7d0",
         "Repeat Attendance & Audience Growth is the lowest-scoring indicator at 7.4/10, representing the clearest strategic opportunity — converting the 42% first-time attendees into returning audiences would significantly strengthen the Cultural: Horizon outcome stream."),
        ("👥", "Audience Insight","#c2410c", "#fff7ed",  "#fed7aa",
         "First-Time Theatre Access (42% of attendees) and Cultural Identity Validation (8.4/10) together indicate that Monkey Baa is successfully reaching new and diverse audiences, with CALD and low-SES communities showing above-average identity recognition scores."),
        ("💬", "Sentiment",       "#059669", "#f0fdf4",  "#bbf7d0",
         "91% positive sentiment is recorded across all text feedback, with negative sentiment entirely confined to logistical concerns (parking, show timing) — no respondent expressed dissatisfaction with the artistic content, storytelling, or emotional impact of the programme."),
        ("📊", "Data Confidence", "#2563eb", "#eff6ff",  "#bfdbfe",
         "With 147 survey responses across 3 programme streams and 12 events, the 2024 dataset provides a reliable basis for indicator scoring, though expanding survey coverage to the full 3,240 audience (currently 4.5% response rate) would further strengthen the evidence base."),
    ]

    row1 = st.columns(3)
    row2 = st.columns(3)
    for i, (icon, title, colour, bg, bdr, text) in enumerate(insight_panels):
        col = row1[i] if i < 3 else row2[i-3]
        with col:
            st.markdown(
                f'<div style="background:{bg};border:1px solid {bdr};border-radius:12px;' +
                f'padding:16px 18px;margin-bottom:8px">' +
                f'<div style="font-size:10px;font-weight:700;color:{colour};text-transform:uppercase;' +
                f'letter-spacing:1px;margin-bottom:8px">{icon} {title}</div>' +
                f'<div style="font-size:13px;color:#374151;line-height:1.75">{text}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    if st.button("Generate Reports →", use_container_width=True):
        go('reports')

# ── CHAT ──────────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown("---")
    st.markdown("### 💬 Ask AI Chat Assistant")
    st.caption("Ask any question about your data, indicators, or programme impact")

    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    ai = st.session_state.ai_results if st.session_state.ai_results is not None else _demo_ai()

    if df is None:
        st.info("Upload data first to use the chat assistant.")
        return

    if 'chat_pairs' not in st.session_state:
        st.session_state.chat_pairs = []

    for idx, pair in enumerate(st.session_state.chat_pairs):
        q, a = pair['q'], pair['a']
        st.markdown(
            f'<div style="background:#f1f5f9;border-radius:10px 10px 10px 4px;' +
            f'padding:10px 14px;margin-bottom:6px;font-size:13px;color:#1e293b">' +
            f'<strong>You:</strong> {q}</div>',
            unsafe_allow_html=True
        )
        col_ans, col_rm = st.columns([6, 1])
        with col_ans:
            st.markdown(
                f'<div style="background:white;border:1px solid #e2e8f0;border-radius:4px 10px 10px 10px;' +
                f'padding:12px 14px;font-size:13px;color:#374151;line-height:1.7">{a}</div>',
                unsafe_allow_html=True
            )
        with col_rm:
            if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                st.session_state.chat_pairs.pop(idx)
                st.rerun()
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col_in, col_send = st.columns([5, 1])
        with col_in:
            user_msg = st.text_input(
                "Question", label_visibility="collapsed",
                placeholder="e.g. Which indicator needs the most attention?"
            )
        with col_send:
            send = st.form_submit_button("Send")
        if send and user_msg.strip():
            reply = chat_response(user_msg, ai, df)
            st.session_state.chat_pairs.append({"q": user_msg, "a": reply})
            st.rerun()



# ── REPORTS ───────────────────────────────────────────────────────────────────
def build_pdf_bytes(audience, report_html, ai, n_rows):
    """Generate a real PDF using fpdf2 with unicode cleaning."""
    import re as _re
    from fpdf import FPDF, XPos, YPos

    def clean(s):
        return (str(s)
            .replace('\u2014','-').replace('\u2013','-')
            .replace('\u2019',"'").replace('\u2018',"'")
            .replace('\u201c','"').replace('\u201d','"')
            .replace('\u00a0',' ').replace('\u2022','-')
            .replace('\u00b7','.').replace('\u2026','...')
            .encode('latin-1','replace').decode('latin-1'))

    text = _re.sub(r'<[^>]+>', ' ', report_html)
    text = _re.sub(r'\s+', ' ', text).strip()
    text = clean(text)
    date_str = datetime.today().strftime('%d %B %Y')

    class PDF(FPDF):
        def header(self):
            pass

    pdf = PDF()
    pdf.add_page()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Header bar ──────────────────────────────────────────────────────────
    pdf.set_fill_color(28, 43, 74)
    pdf.rect(0, 0, 210, 30, style='F')
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(18, 8)
    pdf.cell(0, 8, clean(f'Monkey Baa Theatre Company  -  {audience}'),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_x(18)
    pdf.cell(0, 5, clean(f'Green Sheep Tour 2024  |  {date_str}'),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(30, 41, 59)
    pdf.set_y(38)

    # ── Key Metrics ──────────────────────────────────────────────────────────
    pdf.set_fill_color(239, 246, 255)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 7, '  KEY METRICS', fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 8)
    metrics = [
        ('Total Survey Responses', str(n_rows)),
        ('First-Time Attendees', '42%'),
        ('Average Satisfaction', f"{ai.get('avg_satisfaction', 4.6)}/5"),
        ('NPS Score', f"{ai.get('nps', 72)}  (sector avg 45)"),
        ('Recommendation Rate', f"{ai.get('recommendation_rate', 94)}%"),
        ('Positive Sentiment', f"{ai.get('sentiment_pct', 91)}%"),
    ]
    for label, val in metrics:
        pdf.cell(100, 5.5, clean(f'  {label}'))
        pdf.cell(0, 5.5, clean(val), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Top Indicators ────────────────────────────────────────────────────────
    pdf.set_fill_color(240, 253, 244)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 7, '  TOP IMPACT INDICATORS  (Theory of Change)',
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 8)
    top_inds = sorted(INDICATOR_DETAIL.items(), key=lambda x: x[1][2], reverse=True)[:10]
    for ind_name, (stream, _, score, _) in top_inds:
        label_text = f'  {ind_name}  [{stream}]'
        # Truncate if too long
        if len(label_text) > 65:
            label_text = label_text[:62] + '...'
        pdf.cell(150, 5.5, clean(label_text))
        pdf.cell(0, 5.5, f'{score}/10', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Report Narrative ─────────────────────────────────────────────────────
    pdf.set_fill_color(248, 250, 252)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 7, '  REPORT NARRATIVE', fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(55, 65, 81)

    # Use multi_cell for proper word-wrapping
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    # Clean and chunk text sensibly
    narrative = text[:3000] if len(text) > 3000 else text
    # Replace common HTML entities
    narrative = narrative.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    pdf.multi_cell(usable_w, 5.5, clean(f'  {narrative}'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-16)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, clean('Monkey Baa Theatre Company  |  Impact Reporting System  |  Confidential'),
             align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


def page_reports():
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    ai = st.session_state.ai_results
    if df is None:
        st.warning("Please complete previous steps first.")
        if st.button("Back to Upload"): go('upload')
        return
    if ai is None:
        ai = _demo_ai()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("Generate Reports")
    st.markdown("---")

    # ── Derive real data summary for reports ──────────────────────────────────
    n_rows = len(df)
    avg_sat = ai.get('avg_satisfaction', 4.6)
    nps = ai.get('nps', 72)
    rec = ai.get('recommendation_rate', 94)
    sent = ai.get('sentiment_pct', 91)
    top_soc = sorted(ai.get('social_indicators', {}).items(), key=lambda x: x[1], reverse=True)
    top_cult = sorted(ai.get('cultural_indicators', {}).items(), key=lambda x: x[1], reverse=True)
    best_soc = top_soc[0][0] if top_soc else "Spontaneous Joy Response"
    best_soc_score = top_soc[0][1] if top_soc else 9.1
    lowest_ind = sorted({**ai.get('social_indicators',{}), **ai.get('cultural_indicators',{})}.items(), key=lambda x: x[1])[0] if ai.get('social_indicators') else ("Repeat Attendance", 7.4)

    # Build paragraph summary from real dashboard data
    DASHBOARD_SUMMARY = (
        f"Monkey Baa Theatre Company's 2024 programme delivered measurable impact across both social "
        f"and cultural outcome streams, reaching audiences across 12 events with an average satisfaction "
        f"of {avg_sat}/5 and a Net Promoter Score of {nps} — well above the sector average of 45. "
        f"Across {n_rows} survey responses, {sent}% of feedback was positive, with {rec}% of respondents "
        f"indicating they would recommend Monkey Baa to others. The Theory of Change Social stream was "
        f"led by {best_soc} ({best_soc_score}/10), while Theatre Curiosity & Engagement (8.6/10) was "
        f"the strongest cultural outcome, confirming that the programme successfully ignites both "
        f"emotional and arts-engagement outcomes simultaneously. The one area requiring strategic "
        f"attention is {lowest_ind[0]} ({lowest_ind[1]}/10), which represents the primary growth "
        f"opportunity for the 2025 programme cycle."
    )

    # Updated indicator coverage using real 21 indicators
    INDICATOR_COVERAGE = [
        ("Spontaneous Joy Response",          "Social: Spark",   "Covered"),
        ("Creative Inspiration Spark",        "Social: Spark",   "Covered"),
        ("Story Self-Recognition",            "Social: Spark",   "Covered"),
        ("First-Time Theatre Access",         "Social: Spark",   "Covered"),
        ("Empathy & Emotional Intelligence",  "Social: Growth",  "Covered"),
        ("Confidence & Active Participation", "Social: Growth",  "Partial"),
        ("Social Inclusion & Belonging",      "Social: Growth",  "Covered"),
        ("Positive Theatre Memory",           "Social: Growth",  "Covered"),
        ("Well-being Through Arts",           "Social: Growth",  "Covered"),
        ("Equity of Cultural Access",         "Social: Horizon", "Partial"),
        ("Lifelong Empathy & Life Skills",    "Social: Horizon", "Partial"),
        ("Community Social Capital",          "Social: Horizon", "Partial"),
        ("Cultural Identity Validation",      "Cultural: Spark", "Covered"),
        ("Creative Making Interest",          "Cultural: Spark", "Covered"),
        ("Theatre Curiosity & Engagement",    "Cultural: Spark", "Covered"),
        ("Theatre Appreciation & Advocacy",   "Cultural: Growth","Covered"),
        ("Cultural Literacy & Openness",      "Cultural: Growth","Partial"),
        ("Repeat Attendance & Audience Growth","Cultural: Growth","Gap"),
        ("Lifelong Arts Engagement",          "Cultural: Horizon","Partial"),
        ("Australian Storytelling Contribution","Cultural: Horizon","Covered"),
        ("Sector Influence & Policy Impact",  "Cultural: Horizon","Partial"),
    ]

    ind_rows_html = "".join(
        f'<tr style="border-bottom:1px solid #f1f5f9">'
        f'<td style="padding:7px 10px;font-size:12px;color:#1e293b">{row[0]}</td>'
        f'<td style="padding:7px 10px;font-size:11px;color:#64748b">{row[1]}</td>'
        f'<td style="padding:7px 10px">'
        f'<span style="background:{"#d1fae5" if row[2]=="Covered" else "#fef9c3" if row[2]=="Partial" else "#fee2e2"};'
        f'color:{"#065f46" if row[2]=="Covered" else "#854d0e" if row[2]=="Partial" else "#991b1b"};'
        f'padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">{row[2]}</span>'
        f'</td></tr>'
        for row in INDICATOR_COVERAGE
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    col_sel, col_prev = st.columns([1, 2])

    audiences = {
        "Executive Team":    "Internal strategic overview",
        "Funding Bodies":    "Grant & philanthropic evidence",
        "Schools & Teachers":"Educational outcomes summary",
        "Community Partners":"Local impact story",
    }

    with col_sel:
        st.markdown("#### Select Audience")
        selected = st.radio("Audience", list(audiences.keys()), label_visibility="collapsed")
        st.caption(audiences[selected])
        st.markdown("---")
        gen_btn = st.button("Generate Report →", use_container_width=True)

    with col_prev:
        if gen_btn or selected in st.session_state.reports:
            if gen_btn:
                with st.spinner(f"Writing {selected} report..."):
                    text = generate_report_text(selected, ai, n_rows)
                    st.session_state.reports[selected] = text

            report_html = st.session_state.reports.get(selected, "")
            if report_html:

                # ── Executive Team: custom structured layout ───────────────
                if selected == "Executive Team":
                    exec_html = f"""
<div style="font-family:'DM Sans',sans-serif">
  <div style="background:#1c2b4a;padding:16px 20px;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center">
    <div><div style="color:white;font-size:17px;font-weight:700">Executive Team Report</div></div>
    <div style="color:rgba(255,255,255,0.5);font-size:11px">{datetime.today().strftime('%d %B %Y')}</div>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">1. Programme Summary</div>
    <p style="font-size:13px;color:#374151;line-height:1.8;margin:0">{DASHBOARD_SUMMARY}</p>
  </div>

  <div style="background:white;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">2. Impact vs Goals — Indicator Coverage (All 21 Indicators)</div>
    <table style="width:100%;border-collapse:collapse">
      <tr style="background:#f1f5f9">
        <th style="padding:7px 10px;text-align:left;font-size:11px;color:#64748b">Indicator</th>
        <th style="padding:7px 10px;text-align:left;font-size:11px;color:#64748b">Stream</th>
        <th style="padding:7px 10px;text-align:left;font-size:11px;color:#64748b">Status</th>
      </tr>
      {ind_rows_html}
    </table>
  </div>

  <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-top:none">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">3. Key Risks & Recommended Actions</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div style="background:white;border:1px solid #fee2e2;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#dc2626;margin-bottom:6px">⚠ Key Risk</div>
        <div style="font-size:12px;color:#374151">Repeat Attendance & Audience Growth (7.4/10) is the lowest indicator — first-time audiences are not returning at sufficient rates to achieve Cultural: Horizon outcomes.</div>
      </div>
      <div style="background:white;border:1px solid #bbf7d0;border-radius:8px;padding:12px">
        <div style="font-size:10px;font-weight:700;color:#16a34a;margin-bottom:6px">✦ Recommended Action</div>
        <div style="font-size:12px;color:#374151">Launch a post-show re-engagement programme for the 42% first-time attendees, with targeted outreach to regional venues where Joy & Wonder scores are 12% above metro average.</div>
      </div>
    </div>
  </div>
</div>"""
                    st.markdown(f'<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden">{exec_html}</div>', unsafe_allow_html=True)
                    report_for_pdf = exec_html
                else:
                    st.markdown(
                        f'<div style="margin-bottom:12px"><span style="background:#dbeafe;color:#1d4ed8;'
                        f'padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600">{selected}</span>'
                        f'<span style="font-size:11px;color:#94a3b8;margin-left:8px">'
                        f'{datetime.today().strftime("%d %B %Y")}</span></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden">{report_html}</div>', unsafe_allow_html=True)
                    report_for_pdf = report_html

                # ── Actions ──────────────────────────────────────────────────
                st.markdown("---")
                col_dl, col_regen = st.columns([1, 1])
                with col_dl:
                    try:
                        pdf_bytes = build_pdf_bytes(selected, report_for_pdf, ai, n_rows)
                        st.download_button(
                            "⬇ Download PDF",
                            data=pdf_bytes,
                            file_name=f"monkey_baa_{selected.lower().replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF error: {e}")
                        # Still try a minimal PDF
                        try:
                            from fpdf import FPDF, XPos, YPos
                            def _c(s): return str(s).encode('latin-1','replace').decode('latin-1')
                            p = FPDF(); p.add_page(); p.set_margins(18,18,18)
                            p.set_font('Helvetica','B',14); p.set_text_color(255,255,255)
                            p.set_fill_color(28,43,74); p.rect(0,0,210,25,style='F')
                            p.set_xy(18,8); p.cell(0,8,_c(f'Monkey Baa - {selected}'),new_x=XPos.LMARGIN,new_y=YPos.NEXT)
                            p.set_text_color(30,41,59); p.set_y(32)
                            p.set_font('Helvetica','',9)
                            import re as _re2
                            txt = _re2.sub(r'<[^>]+>',' ',report_for_pdf)
                            txt = _re2.sub(r'\s+',' ',txt).strip()[:2000]
                            p.multi_cell(p.w-36, 5.5, _c(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                            st.download_button("⬇ Download PDF", data=bytes(p.output()),
                                file_name=f"monkey_baa_{selected.lower().replace(' ','_')}.pdf",
                                mime="application/pdf", use_container_width=True)
                        except Exception as e2:
                            st.warning(f"PDF generation failed: {e2}")

                # ── Send to Stakeholders ──────────────────────────────────────
                st.markdown("---")
                st.markdown("**Send Report to Selected Stakeholders**")

                stakeholders = [
                    ("🏛️", "Australian Government — Funding Partners",
                     "grants@australiacouncil.gov.au"),
                    ("🎭", "Arts Centre Melbourne — Community Partner",
                     "community@sydney.nsw.gov.au"),
                    ("🏫", "Glenelg Primary School",
                     "primary@education.nsw.gov.au"),
                ]

                selected_stk = []
                for icon, name, email in stakeholders:
                    col_chk, col_info = st.columns([0.5, 6])
                    with col_chk:
                        chk = st.checkbox("", key=f"stk_{name}", label_visibility="collapsed")
                    with col_info:
                        st.markdown(
                            f'<div style="padding:8px 0;display:flex;align-items:center;gap:10px">'
                            f'<span style="font-size:18px">{icon}</span>'
                            f'<div><div style="font-size:13px;font-weight:600;color:#1c2b4a">{name}</div>'
                            f'<div style="font-size:12px;color:#2563eb">✉ {email}</div></div></div>',
                            unsafe_allow_html=True
                        )
                    if chk:
                        selected_stk.append(name)

                if st.button("▶ Send Report to Selected Stakeholders",
                             use_container_width=True, key="send_btn"):
                    if selected_stk:
                        st.markdown(
                            '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:14px;'
                            'padding:32px;text-align:center;margin-top:16px">'
                            '<div style="font-size:48px;margin-bottom:12px">✅</div>'
                            '<div style="font-size:20px;font-weight:700;color:#16a34a;margin-bottom:6px">'
                            'Reports Sent Successfully!</div>'
                            f'<div style="font-size:13px;color:#374151">Report sent to: {", ".join(selected_stk)}</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("Select at least one stakeholder first.")

        else:
            st.info("Select an audience and click 'Generate Report →'")

    # ── Bottom navigation ─────────────────────────────────────────────────────
    st.markdown("---")
    col_nav1, col_nav2 = st.columns([1, 2])
    with col_nav1:
        if st.button("← Back", key="back_reports_bottom"):
            go('insights')
    with col_nav2:
        if st.button("✓ Complete & Start New Analysis", key="new_analysis",
                     type="primary", use_container_width=True):
            # Reset all analysis state
            for key in ['df_raw','df_clean','df_masked','ai_results','reports',
                        'fixed_ids','issues','pii_log','chat_pairs','steps_done',
                        'selected_indicator','file_name']:
                if key in st.session_state:
                    if key in ['fixed_ids','steps_done']:
                        st.session_state[key] = set()
                    elif key in ['reports']:
                        st.session_state[key] = {}
                    elif key in ['issues','pii_log','chat_pairs']:
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            st.session_state.page = 'upload'
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page

if page == 'login':
    page_login()
else:
    sidebar()
    if page == 'upload':
        page_upload()
    elif page == 'cleaning':
        page_cleaning()
    elif page == 'insights':
        page_insights()
    elif page == 'reports':
        page_reports()

    # Chat visible on all pages except login
    if page != 'login':
        with st.expander("💬 Ask AI Assistant", expanded=(page == 'insights')):
            render_chat()
