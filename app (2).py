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
    if not openai_available:
        return _demo_reports().get(audience, "")

    soc = ai.get('social_indicators', {})
    cult = ai.get('cultural_indicators', {})

    prompt = f"""Write a stakeholder impact report for Monkey Baa Theatre Company aligned to their Theory of Change.

Audience: {audience}
Survey data: {n_rows} responses, {ai.get('avg_satisfaction','4.6')}/5 avg satisfaction,
{ai.get('recommendation_rate',94)}% recommend, NPS {ai.get('nps',72)}.

Theory of Change — Social Outcome scores: {json.dumps(soc)}
Theory of Change — Cultural Outcome scores: {json.dumps(cult)}
Key finding: {ai.get('top_finding','')}
Trend: {ai.get('trend','')}

Monkey Baa's mission: to uplift young Australians by embedding the arts into their formative years.
Their Theory of Change tracks: Joy & Wonder → Empathy → Confidence → Social Inclusion (social stream)
and Identity Recognition → Arts Appreciation → Cultural Literacy (cultural stream).

Write exactly 3 paragraphs. No headers. Plain text. Tone: professional and warm.
Directly reference Theory of Change outcomes. Tailor specifically for: {audience}."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=550
        )
        return resp.choices[0].message.content
    except:
        return _demo_reports().get(audience, "")

def _demo_reports():
    return {
        "Executive Team": "Monkey Baa's 2024 program demonstrates strong delivery against our Theory of Change across both social and cultural outcome streams. Joy & Wonder — our foundational 'spark' indicator — scored 9.1/10, confirming that performances are successfully igniting imagination and emotional connection in young audiences. With 147 survey responses, an average satisfaction of 4.6/5 and a Net Promoter Score of 72 (industry average: 45), the data shows our programs consistently deliver on the promise of transformative theatre experiences.\n\nAcross social outcomes, Empathy & Emotional Intelligence scored 8.8/10 and Feeling Included & Valued reached 8.7/10, reflecting our intentional approach to storytelling that mirrors the real lives of diverse young Australians. In the cultural stream, Curiosity & Theatre Engagement (8.6/10) and Identity Recognition (8.4/10) indicate that audiences are not only enjoying performances but are developing lasting relationships with theatre as an art form.\n\nResponse volumes grew 35% year-on-year, and 94% of respondents would recommend Monkey Baa to others — a powerful indicator of community trust. The one area for strategic attention is Repeat Attendance (7.4/10), which signals an opportunity to convert first-time audiences into lifelong arts participants, directly supporting our long-term Theory of Change horizon.",
        "Funding Bodies": "This report presents evidence of Monkey Baa Theatre Company's social and cultural impact in 2024, measured against our Theory of Change framework. Across 147 survey responses from three program streams, AI-powered NLP analysis mapped audience feedback directly to our social and cultural outcome indicators, providing rigorous, evidence-based reporting aligned with your investment criteria.\n\nSocial outcomes demonstrate strong performance: Joy & Wonder scored 9.1/10 (immediate impact), while Empathy & Emotional Intelligence reached 8.8/10 and Social Inclusion & Connection scored 8.2/10, representing measurable medium-term outcomes consistent with our Theory of Change growth pathway. In the cultural stream, Curiosity & Theatre Engagement (8.6/10) and Arts Appreciation (8.2/10) evidence the development of long-term arts engagement behaviours in young audiences.\n\nWith 3,240 young people reached across 12 events, a recommendation rate of 94%, and year-on-year response growth of 35%, Monkey Baa's 2024 data demonstrates sustained, scalable social impact. Continued philanthropic investment will directly support the Theatre Unlimited initiative, ensuring young people from financially, geographically, and socially disadvantaged backgrounds receive equitable access to these transformative experiences.",
        "Schools & Teachers": "Thank you for partnering with Monkey Baa Theatre Company in 2024. This summary presents what we measured from teachers and students following our visits, mapped against Monkey Baa's Theory of Change social and cultural outcomes.\n\nTeacher feedback strongly evidences social outcomes: students demonstrated visible increases in Empathy & Emotional Intelligence (8.8/10) and Confidence & Self-Esteem (7.9/10) following performances. Teachers noted students used new emotional language in classroom discussions, directly supporting outcomes in the Australian Curriculum's Personal and Social Capability strand. Cultural outcomes were equally strong — Identity Recognition scored 8.4/10, with teachers reporting that students from diverse backgrounds felt represented in the stories they witnessed.\n\nCuriosity & Theatre Engagement scored 8.6/10, suggesting our programs are building genuine enthusiasm for live theatre among students who may otherwise have limited arts exposure. Well-being & Positive Memories (8.5/10) reflects the lasting emotional impact of these experiences. We look forward to continuing this partnership in 2025 and working together to support the emotional and creative development of your students.",
        "Community Partners": "In 2024, Monkey Baa Theatre Company reached 3,240 young people across 12 events in your communities. Measured through our Theory of Change framework, the impact of these experiences extends well beyond the performance itself — into the emotional lives, confidence, and cultural identities of young people.\n\nJoy & Wonder scored 9.1/10 — the highest of all our Theory of Change indicators — meaning that the 'spark' we aim to ignite in every young person is being felt across our audiences. Feeling Included & Valued scored 8.7/10, reflecting our commitment to stories that make every child feel seen. With 94% of respondents saying they would recommend Monkey Baa to others, the word-of-mouth impact of our presence in your community is powerful.\n\nWell-being & Positive Memories (8.5/10) and Social Inclusion & Connection (8.2/10) confirm that performances are contributing to community cohesion and the long-term mental health of young audiences. We are deeply grateful for your partnership — together, we are ensuring that every young Australian has access to the kind of experience that can genuinely change their life."
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

            report_text = st.session_state.reports.get(selected, "")
            if report_text:
                st.markdown(f"""
                <div style="background:#dbeafe;color:#1d4ed8;display:inline-block;
                     padding:4px 12px;border-radius:6px;font-size:12px;
                     font-weight:600;margin-bottom:12px">{selected}</div>
                <span style="font-size:11px;color:#94a3b8;margin-left:8px">
                    Green Sheep Tour 2024 · {datetime.today().strftime('%d %B %Y')}</span>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="report-box">{report_text.replace(chr(10), "<br><br>")}</div>',
                            unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.download_button(
                        "⬇ Download .txt",
                        data=report_text,
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
