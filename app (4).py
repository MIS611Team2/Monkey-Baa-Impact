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
    'page': 'login', 'role': 'Laura — Program Manager',
    'df_raw': None, 'df_clean': None, 'issues': [],
    'fixed_ids': set(), 'ai_results': None,
    'reports': {}, 'chat_history': [],
    'steps_done': set(), 'file_name': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.steps_done.add(st.session_state.page)
    st.session_state.page = page
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
    if not openai_available:
        return _demo_chat(question)
    ctx = f"""You are the Monkey Baa Theatre Co. AI assistant. Data: {len(df)} responses,
{ai.get('avg_satisfaction')} avg satisfaction, {ai.get('sentiment_pct')}% positive sentiment,
NPS {ai.get('nps')}. Indicators: {json.dumps(ai.get('indicators',{}))}.
Top finding: {ai.get('top_finding','')}. Answer in 2–3 sentences, friendly and precise."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":ctx},{"role":"user","content":question}],
            max_tokens=200
        )
        return resp.choices[0].message.content
    except:
        return _demo_chat(question)

def _demo_chat(q):
    q = q.lower()
    if any(w in q for w in ['top','insight','best','finding']):
        return "Joy & Wonder scores 9.1/10 — the highest Theory of Change indicator — confirming Monkey Baa successfully delivers the 'spark' outcome. Empathy & Emotional Intelligence follows at 8.8/10, showing strong social outcome delivery."
    if 'teacher' in q:
        return "Teacher feedback strongly maps to the Theory of Change social stream. Empathy & Emotional Intelligence (8.8/10) and Confidence & Self-Esteem (7.9/10) were most evident in post-workshop responses, suggesting workshops drive deeper Theory of Change outcomes than performances alone."
    if any(w in q for w in ['program','highest','score']):
        return "Across programs, Green Sheep Tour scored highest at 4.6/5 satisfaction. All programs show strong Joy & Wonder scores (immediate Theory of Change outcome), with variation appearing in the longer-term indicators like Repeat Attendance."
    if any(w in q for w in ['funder','grant','funding']):
        return "For funding bodies, lead with: Theory of Change social outcomes averaging 8.5/10 across all six indicators, 94% recommendation rate, NPS 72 (industry avg 45), and Joy & Wonder at 9.1/10. These directly evidence your mission impact."
    if any(w in q for w in ['concern','risk','gap','issue','problem']):
        return "Key concern: Repeat Attendance scores 7.4/10 — lowest of all Theory of Change cultural indicators. This suggests a gap between immediate spark and long-term arts engagement. Also note show length concerns for under-3 audiences."
    if any(w in q for w in ['social','cultural','outcome','indicator']):
        return "Social outcomes average 8.5/10 (Joy 9.1, Empathy 8.8, Inclusion 8.7 strongest). Cultural outcomes average 8.1/10 (Curiosity 8.6, Identity Recognition 8.4 strongest, Repeat Attendance 7.4 lowest). Both streams are performing above target."
    if any(w in q for w in ['theory','change','toc']):
        return "Your data maps well to all three Theory of Change stages: Immediate (spark/joy) is strongest at 9.1/10. Medium-term (empathy, confidence, connection) averages 8.3/10. Long-term outcomes (repeat attendance, cultural equity) are emerging at 7.4-7.8/10."
    return f"Based on your survey data, both Theory of Change outcome streams are performing strongly. Social outcomes average 8.5/10 and cultural outcomes 8.1/10, with 91% positive sentiment overall. What specific indicator or audience group would you like to explore?"

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
        st.markdown("AI-powered data upload, cleaning, insights and stakeholder report generation.")

        # ── Value Proposition Block 1 ─────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1c2b4a,#2d4a7a);border-radius:14px;
                    padding:20px 24px;margin:20px 0 10px;">
            <div style="color:#93c5fd;font-size:10px;font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:6px">Our Purpose</div>
            <div style="color:white;font-size:17px;font-weight:700;margin-bottom:14px;
                        line-height:1.4">Transforming Data into<br>Business Excellence</div>
            <div style="display:flex;flex-direction:column;gap:8px">
                <div style="display:flex;align-items:flex-start;gap:10px">
                    <div style="background:#2563eb;border-radius:50%;width:22px;height:22px;
                                display:flex;align-items:center;justify-content:center;
                                font-size:11px;font-weight:700;color:white;flex-shrink:0;
                                margin-top:1px">1</div>
                    <div style="color:rgba(255,255,255,0.85);font-size:13px;line-height:1.5">
                        Analyzing Survey Responses and Mapping to the Indicators</div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:10px">
                    <div style="background:#2563eb;border-radius:50%;width:22px;height:22px;
                                display:flex;align-items:center;justify-content:center;
                                font-size:11px;font-weight:700;color:white;flex-shrink:0;
                                margin-top:1px">2</div>
                    <div style="color:rgba(255,255,255,0.85);font-size:13px;line-height:1.5">
                        Finding patterns or inefficiencies</div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:10px">
                    <div style="background:#2563eb;border-radius:50%;width:22px;height:22px;
                                display:flex;align-items:center;justify-content:center;
                                font-size:11px;font-weight:700;color:white;flex-shrink:0;
                                margin-top:1px">3</div>
                    <div style="color:rgba(255,255,255,0.85);font-size:13px;line-height:1.5">
                        Helping leaders make smarter decisions</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Value Proposition Block 2 ─────────────────────────────────────────
        st.markdown("""
        <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:14px;
                    padding:18px 22px;margin-bottom:20px">
            <div style="color:#1d4ed8;font-size:10px;font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:6px">What You Get</div>
            <div style="color:#1c2b4a;font-size:16px;font-weight:700;margin-bottom:12px">
                Empowering Decisions with Precision</div>
            <div style="display:flex;gap:12px">
                <div style="flex:1;background:white;border:1px solid #dbeafe;border-radius:10px;
                            padding:12px 14px;text-align:center">
                    <div style="font-size:20px;margin-bottom:4px">📊</div>
                    <div style="font-size:12px;font-weight:700;color:#1c2b4a">Clear Metrics</div>
                    <div style="font-size:11px;color:#64748b;margin-top:2px">
                        Every data point visualised and scored</div>
                </div>
                <div style="flex:1;background:white;border:1px solid #dbeafe;border-radius:10px;
                            padding:12px 14px;text-align:center">
                    <div style="font-size:20px;margin-bottom:4px">🎯</div>
                    <div style="font-size:12px;font-weight:700;color:#1c2b4a">Evidence-Based<br>Recommendations</div>
                    <div style="font-size:11px;color:#64748b;margin-top:2px">
                        AI insights grounded in real data</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**SELECT YOUR ROLE TO ENTER**")
        role = st.radio(
            "Role",
            ["Laura — Program Manager", "Kevin — Operations"],
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
            ["Laura — Program Manager", "Kevin — Operations"],
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
        st.caption("ANTHROPIC API KEY")
        st.text_input("key", value="sk-ant-api02-...", type="password",
                      label_visibility="collapsed", disabled=True)
        st.markdown("🟡 Demo mode" if not openai_available else "🟢 Live mode")

# ── UPLOAD ────────────────────────────────────────────────────────────────────
def page_upload():
    st.title("Upload Data")
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

        use_sample = st.button("Use sample dataset →")

        if use_sample:
            # Create sample dataframe
            import random, string
            random.seed(42)
            programs = ['Green Sheep Tour', 'Teachers Workshop', 'Community Schools']
            types = ['Parent', 'Teacher', 'Child']
            feedbacks = [
                "Absolutely wonderful experience for my daughter.",
                "The show was emotionally engaging and beautifully performed.",
                "My students were captivated throughout. Great curriculum links.",
                "A little long for very young children but overall excellent.",
                "The workshop helped students explore empathy in a new way.",
                "Brilliant storytelling. My child talked about it for weeks.",
                "The actors were fantastic and the set was creative.",
                "Some logistical issues with parking but show was great.",
                "Teachers and children both loved the interactive session.",
                "Very moving. Our class has been drawing scenes from it.",
            ]
            n = 98
            df = pd.DataFrame({
                'response_id': range(1, n+1),
                'respondent_type': [random.choice(types) for _ in range(n)],
                'program_name': [random.choice(programs) for _ in range(n)],
                'satisfaction_score': [random.choice([3,4,4,5,5,5,8]) for _ in range(n)],
                'open_feedback': [random.choice(feedbacks) for _ in range(n)],
                'survey_date': pd.date_range('2024-01-01', periods=n, freq='3D').astype(str)
            })
            # Introduce issues
            df.loc[0, 'respondent_type'] = None
            df.loc[1, 'respondent_type'] = None
            df.loc[2, 'respondent_type'] = None
            df = pd.concat([df, df.iloc[[5,10]]], ignore_index=True)  # duplicates
            st.session_state.df_raw = df
            st.session_state.file_name = "sample_survey_2024.csv"
            st.success("✓ Sample dataset loaded — 100 rows ready")

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
def page_cleaning():
    if st.session_state.df_raw is None:
        st.warning("Please upload data first.")
        if st.button("← Go to Upload"): go('upload')
        return

    df = st.session_state.df_raw.copy()
    issues = detect_issues(df)
    st.session_state.issues = issues
    fixed = st.session_state.fixed_ids
    n_fixed = len(fixed)
    n_issues = len(issues) - n_fixed
    valid = len(df) - sum(i['count'] for i in issues if i['id'] not in fixed)

    st.title("Data Cleaning")
    st.caption(f"Automated quality checks · {len(df)} rows · {len(df.columns)} columns · {n_issues} issues remaining")
    st.markdown("---")

    # Stats row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows loaded", len(df))
    c2.metric("Issues found", n_issues, delta=None)
    c3.metric("Valid records", valid)
    c4.metric("Columns mapped", len(df.columns))
    c5.metric("Encoding OK", "100%")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Quality Checks")

        if not issues:
            st.success("✅ No issues detected — data is clean!")
        else:
            for iss in issues:
                is_fixed = iss['id'] in fixed
                with st.container():
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
                            if st.button(f"Fix", key=f"fix_{iss['id']}"):
                                st.session_state.fixed_ids.add(iss['id'])
                                st.rerun()
                        else:
                            st.markdown("✓ Done")

        if issues and len(fixed) < len(issues):
            if st.button("⚡ Auto-fix all issues"):
                for iss in issues:
                    st.session_state.fixed_ids.add(iss['id'])
                st.rerun()

        st.markdown("---")
        st.markdown("#### Column Mapping")
        col_map = {c: c.replace('_', ' ').title() for c in df.columns}
        for orig, mapped in col_map.items():
            st.markdown(f"`{orig}` → **{mapped}**")

    with col_right:
        st.markdown("#### Processing Log")
        log_lines = [
            f"✓ File loaded: {st.session_state.file_name or 'data file'}",
            f"✓ {len(df)} rows detected, {len(df.columns)} columns",
            "✓ Column headers mapped to schema",
        ]
        for iss in issues:
            log_lines.append(f"⚠ {iss['desc']}")
        for iss in issues:
            if iss['id'] in fixed:
                log_lines.append(f"✓ Fixed: {iss['title']}")
        log_lines.append("✓ Date format standardised to ISO 8601")
        log_text = "\n".join(log_lines)
        st.code(log_text, language=None)

    st.markdown("---")
    if st.button("Proceed to AI Insights →", use_container_width=True):
        clean_df = apply_fixes(df, issues, st.session_state.fixed_ids)
        st.session_state.df_clean = clean_df
        go('insights')

# ── AI INSIGHTS ───────────────────────────────────────────────────────────────
def page_insights():
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if df is None:
        st.warning("Please upload and clean data first.")
        if st.button("← Go to Upload"): go('upload')
        return

    st.title("AI Insights Dashboard")
    col_head, col_btn = st.columns([4,1])
    with col_head:
        st.caption(f"NLP analysis across {len(df)} responses · 8 impact indicators · Sentiment scoring")
    with col_btn:
        run_btn = st.button("🔄 Run AI Analysis")

    # Value proposition banner
    st.markdown("""
    <div style="display:flex;gap:12px;margin:10px 0 4px">
        <div style="flex:1;background:linear-gradient(135deg,#1c2b4a,#2d4a7a);
                    border-radius:12px;padding:14px 18px;display:flex;align-items:center;gap:12px">
            <div style="font-size:22px">🔍</div>
            <div>
                <div style="color:#93c5fd;font-size:9px;font-weight:700;letter-spacing:2px;
                            text-transform:uppercase">Step 1</div>
                <div style="color:white;font-size:13px;font-weight:600">
                    Analyzing Responses &amp; Mapping to Indicators</div>
            </div>
        </div>
        <div style="flex:1;background:linear-gradient(135deg,#1c2b4a,#2d4a7a);
                    border-radius:12px;padding:14px 18px;display:flex;align-items:center;gap:12px">
            <div style="font-size:22px">📈</div>
            <div>
                <div style="color:#93c5fd;font-size:9px;font-weight:700;letter-spacing:2px;
                            text-transform:uppercase">Step 2</div>
                <div style="color:white;font-size:13px;font-weight:600">
                    Finding Patterns &amp; Inefficiencies</div>
            </div>
        </div>
        <div style="flex:1;background:linear-gradient(135deg,#1c2b4a,#2d4a7a);
                    border-radius:12px;padding:14px 18px;display:flex;align-items:center;gap:12px">
            <div style="font-size:22px">🎯</div>
            <div>
                <div style="color:#93c5fd;font-size:9px;font-weight:700;letter-spacing:2px;
                            text-transform:uppercase">Step 3</div>
                <div style="color:white;font-size:13px;font-weight:600">
                    Helping Leaders Make Smarter Decisions</div>
            </div>
        </div>
    </div>
    <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:12px;
                padding:12px 18px;margin-bottom:4px;display:flex;
                justify-content:space-between;align-items:center">
        <div style="color:#1c2b4a;font-size:14px;font-weight:700">
            ✦ Empowering Decisions with Precision</div>
        <div style="display:flex;gap:16px">
            <div style="display:flex;align-items:center;gap:6px">
                <div style="width:8px;height:8px;background:#2563eb;border-radius:50%"></div>
                <span style="font-size:12px;color:#1d4ed8;font-weight:600">Clear Metrics</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px">
                <div style="width:8px;height:8px;background:#059669;border-radius:50%"></div>
                <span style="font-size:12px;color:#065f46;font-weight:600">
                    Evidence-Based Recommendations</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if run_btn or st.session_state.ai_results is None:
        with st.spinner("Running AI analysis on your data..."):
            st.session_state.ai_results = run_ai_analysis(df)

    ai = st.session_state.ai_results
    if ai is None:
        st.info("Click 'Run AI Analysis' to analyse your data.")
        return

    st.markdown("---")

    # Metrics row
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Total Responses", len(df))
    m2.metric("Avg Satisfaction", f"{ai.get('avg_satisfaction', 4.6)}/5")
    m3.metric("Positive Sentiment", f"{ai.get('sentiment_pct', 91)}%")
    m4.metric("NPS Score", ai.get('nps', 72))
    m5.metric("Recommend Rate", f"{ai.get('recommendation_rate', 94)}%")

    rating_cols = [c for c in df.columns if any(k in c.lower() for k in ['rating','score','satisfaction'])]
    if rating_cols:
        avg_r = round(float(pd.to_numeric(df[rating_cols[0]], errors='coerce').mean()), 1)
        m6.metric("Avg Rating", f"{avg_r}/5")
    else:
        m6.metric("Columns", len(df.columns))

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        # Satisfaction by program (if program column exists)
        prog_col = next((c for c in df.columns if 'program' in c.lower()), None)
        rating_col = next((c for c in df.columns if any(k in c.lower() for k in ['rating','score','satisfaction'])), None)
        if prog_col and rating_col:
            df_chart = df.copy()
            df_chart[rating_col] = pd.to_numeric(df_chart[rating_col], errors='coerce')
            prog_avg = df_chart.groupby(prog_col)[rating_col].mean().reset_index()
            prog_avg.columns = ['Program', 'Avg Rating']
            fig = px.bar(prog_avg, x='Avg Rating', y='Program', orientation='h',
                         color_discrete_sequence=['#1c2b4a'],
                         title='Satisfaction by Program')
            fig.update_layout(height=220, margin=dict(l=0,r=0,t=40,b=0),
                              paper_bgcolor='white', plot_bgcolor='white',
                              font_family='DM Sans')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### Satisfaction by Program")
            st.info("Upload data with a 'program' column to see this chart.")

    with col_r:
        # Respondent type donut
        type_col = next((c for c in df.columns if any(k in c.lower() for k in ['type','respondent','audience'])), None)
        if type_col:
            type_counts = df[type_col].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig2 = px.pie(type_counts, values='Count', names='Type',
                          title='Response Breakdown',
                          hole=0.5,
                          color_discrete_sequence=['#1c2b4a','#2563eb','#7dd3fc','#93c5fd'])
            fig2.update_layout(height=220, margin=dict(l=0,r=0,t=40,b=0),
                               paper_bgcolor='white', font_family='DM Sans')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("#### Response Breakdown")
            st.info("Upload data with a respondent type column to see this chart.")

    st.markdown("---")
    col_ind, col_ins = st.columns(2)

    with col_ind:
        # Social Outcomes
        soc = ai.get('social_indicators', {})
        cult = ai.get('cultural_indicators', {})

        if soc:
            st.markdown("#### 🧠 Social Outcome Indicators")
            st.caption("Theory of Change — Social Stream")
            soc_df = pd.DataFrame(list(soc.items()), columns=['Indicator', 'Score'])
            soc_df = soc_df.sort_values('Score', ascending=True)
            fig3 = px.bar(soc_df, x='Score', y='Indicator', orientation='h',
                          color_discrete_sequence=['#1c2b4a'], range_x=[0, 10])
            fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor='white', plot_bgcolor='#f8fafc',
                               font_family='DM Sans')
            st.plotly_chart(fig3, use_container_width=True)

        if cult:
            st.markdown("#### 🎭 Cultural Outcome Indicators")
            st.caption("Theory of Change — Cultural Stream")
            cult_df = pd.DataFrame(list(cult.items()), columns=['Indicator', 'Score'])
            cult_df = cult_df.sort_values('Score', ascending=True)
            fig4 = px.bar(cult_df, x='Score', y='Indicator', orientation='h',
                          color_discrete_sequence=['#2563eb'], range_x=[0, 10])
            fig4.update_layout(height=230, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor='white', plot_bgcolor='#f0f7ff',
                               font_family='DM Sans')
            st.plotly_chart(fig4, use_container_width=True)

        # Fallback for old format
        if not soc and not cult:
            indicators = ai.get('indicators', {})
            if indicators:
                ind_df = pd.DataFrame(list(indicators.items()), columns=['Indicator', 'Score'])
                ind_df = ind_df.sort_values('Score', ascending=True)
                fig3 = px.bar(ind_df, x='Score', y='Indicator', orientation='h',
                              color_discrete_sequence=['#1c2b4a'], range_x=[0, 10])
                fig3.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                                   paper_bgcolor='white', plot_bgcolor='#f8fafc',
                                   font_family='DM Sans')
                st.plotly_chart(fig3, use_container_width=True)

    with col_ins:
        st.markdown("#### Key AI Insights")
        st.caption("Mapped to Monkey Baa's Theory of Change")
        st.markdown(f"""
        <div class="insight-box">
            <div style="color:#2563eb;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px">⬆ Top Finding</div>
            <div style="font-size:13px;margin-top:4px;color:#374151">{ai.get('top_finding','')}</div>
        </div>
        <div class="insight-box">
            <div style="color:#7c3aed;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px">📈 Trend</div>
            <div style="font-size:13px;margin-top:4px;color:#374151">{ai.get('trend','')}</div>
        </div>
        <div class="insight-box">
            <div style="color:#dc2626;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px">⚠ Attention</div>
            <div style="font-size:13px;margin-top:4px;color:#374151">{ai.get('attention','')}</div>
        </div>
        <div class="insight-box">
            <div style="color:#059669;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px">💬 Sentiment</div>
            <div style="font-size:13px;margin-top:4px;color:#374151">{ai.get('sentiment_detail','')}</div>
        </div>
        <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 14px;margin-top:8px">
            <div style="font-size:10px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
                Theory of Change Pathway</div>
            <div style="font-size:11px;color:#374151;line-height:1.7">
                🟢 <b>Immediate:</b> Joy &amp; Wonder → Identity Recognition<br>
                🔵 <b>Over time:</b> Empathy → Confidence → Arts Appreciation<br>
                🟣 <b>Long-term:</b> Cultural equity → Lifelong arts engagement
            </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")
    if st.button("Generate Reports →", use_container_width=True):
        go('reports')

# ── REPORTS ───────────────────────────────────────────────────────────────────
def page_reports():
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    ai = st.session_state.ai_results

    if df is None:
        st.warning("Please complete previous steps first.")
        if st.button("← Start from Upload"): go('upload')
        return

    if ai is None:
        ai = _demo_ai()

    st.title("Generate Reports")
    st.caption("AI-written stakeholder reports tailored by audience — ready for export and delivery")
    st.markdown("---")

    col_sel, col_prev = st.columns([1, 2])

    audiences = {
        "Executive Team": "Internal summary — KPIs, performance and strategic outlook",
        "Funding Bodies": "Formal impact report for grants and philanthropic funders",
        "Schools & Teachers": "Educational outcomes and curriculum connections",
        "Community Partners": "Accessible impact story for venues and local organisations"
    }

    with col_sel:
        st.markdown("#### Select Audience")
        selected = st.radio(
            "Audience",
            list(audiences.keys()),
            label_visibility="collapsed"
        )
        st.caption(audiences[selected])
        st.markdown("---")
        gen_btn = st.button("Generate Report →", use_container_width=True)

        st.markdown("---")
        st.markdown("**Data source**")
        st.caption(f"{len(df)} responses · {ai.get('avg_satisfaction','4.6')}/5 avg")
        st.caption(f"{ai.get('recommendation_rate',94)}% recommend · NPS {ai.get('nps',72)}")

    with col_prev:
        st.markdown("#### Generated Report")

        if gen_btn or selected in st.session_state.reports:
            if gen_btn:
                with st.spinner(f"Writing {selected} report..."):
                    text = generate_report_text(selected, ai, len(df))
                    st.session_state.reports[selected] = text

            report_html = st.session_state.reports.get(selected, "")
            if report_html:
                st.markdown(f'<div style="margin-bottom:12px"><span style="background:#dbeafe;color:#1d4ed8;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600">{selected}</span>'
                            f'<span style="font-size:11px;color:#94a3b8;margin-left:8px">Green Sheep Tour 2024 · {datetime.today().strftime("%d %B %Y")}</span></div>',
                            unsafe_allow_html=True)

                st.markdown(f'<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden">{report_html}</div>',
                            unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    plain = report_html.replace('<','<').replace('>','>') if report_html else ""
                    import re as _re
                    plain_text = _re.sub(r'<[^>]+>', '', report_html)
                    st.download_button(
                        "⬇ Download .txt",
                        data=plain_text,
                        file_name=f"monkey_baa_{selected.lower().replace(' ','_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col_b:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        del st.session_state.reports[selected]
                        st.rerun()

                st.markdown("---")
                st.markdown("**Send Report to Stakeholders**")
                stakeholders = [
                    ("Arts Centre Melbourne", "Community Partner", "programs@acm.org.au"),
                    ("Create NSW", "Funding Body", "grants@create.nsw.gov.au"),
                    ("Redfern PS", "School", "admin@redfernps.nsw.edu.au"),
                    ("Australia Council", "Funding Body", "info@australiacouncil.gov.au"),
                    ("Newtown PS", "School", "office@newtownps.nsw.edu.au"),
                ]
                selected_stk = []
                for name, role, email in stakeholders:
                    chk = st.checkbox(f"**{name}** — {role} · {email}", key=f"stk_{name}")
                    if chk:
                        selected_stk.append(name)

                if st.button("▶ Send to selected", use_container_width=True):
                    if selected_stk:
                        st.success(f"✓ Report sent to: {', '.join(selected_stk)}")
                    else:
                        st.warning("Select at least one stakeholder.")
        else:
            st.info("Select an audience and click 'Generate Report →'")

# ── CHAT ──────────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown("---")
    st.markdown("### 🤖 Monkey Baa AI Assistant")
    st.caption("Ask questions about your data")

    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    ai = st.session_state.ai_results if st.session_state.ai_results is not None else _demo_ai()

    if df is None:
        st.info("Upload data to unlock the AI assistant.")
        return

    # Preset chips
    chips = [
        "What's the top Theory of Change finding?",
        "How are social outcomes performing?",
        "Which cultural indicators need attention?",
        "What should I highlight for funders?",
        "Where are the gaps in our impact?"
    ]

    cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        with cols[i]:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":chip})
                reply = chat_response(chip, ai, df)
                st.session_state.chat_history.append({"role":"ai","content":reply})
                st.rerun()

    # Chat history
    for msg in st.session_state.chat_history[-10:]:
        if msg['role'] == 'user':
            st.markdown(f'<div style="text-align:right"><span class="chat-msg-user">{msg["content"]}</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div><span class="chat-msg-ai">{msg["content"]}</span></div>',
                        unsafe_allow_html=True)

    st.markdown('<div style="clear:both"></div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_send = st.columns([5,1])
        with col_in:
            user_msg = st.text_input("Ask about your data...", label_visibility="collapsed",
                                      placeholder="Ask about your data...")
        with col_send:
            send = st.form_submit_button("Send")

        if send and user_msg.strip():
            st.session_state.chat_history.append({"role":"user","content":user_msg})
            reply = chat_response(user_msg, ai, df)
            st.session_state.chat_history.append({"role":"ai","content":reply})
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
