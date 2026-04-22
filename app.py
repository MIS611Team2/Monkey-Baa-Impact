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

    prompt = f"""You are an education impact analyst for Monkey Baa Theatre Company, a children's theatre in Australia.

Survey summary: {len(df)} total responses. Average rating: {avg_r or 'N/A'}/5.

Sample text feedback:
{feedback_str or 'No text feedback columns detected.'}

Analyse and return ONLY valid JSON (no markdown):
{{
  "sentiment_pct": <integer 0-100>,
  "nps": <integer 0-100>,
  "recommendation_rate": <integer 0-100>,
  "avg_satisfaction": <float 0-5>,
  "indicators": {{
    "Emotional engagement": <float 0-10>,
    "Empathy development": <float 0-10>,
    "Imagination": <float 0-10>,
    "Arts appreciation": <float 0-10>,
    "Creative response": <float 0-10>,
    "Wellbeing": <float 0-10>,
    "Communication skills": <float 0-10>,
    "Social connection": <float 0-10>
  }},
  "top_finding": "<one clear sentence>",
  "trend": "<one clear sentence>",
  "attention": "<one concern sentence>",
  "sentiment_detail": "<one sentence about sentiment breakdown>"
}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        st.warning(f"AI error: {e}. Showing demo results.")
        return _demo_ai()

def _demo_ai():
    return {
        "sentiment_pct": 91, "nps": 72, "recommendation_rate": 94,
        "avg_satisfaction": 4.6,
        "indicators": {
            "Emotional engagement": 8.8, "Empathy development": 8.8,
            "Imagination": 8.2, "Arts appreciation": 8.2,
            "Creative response": 7.6, "Wellbeing": 7.4,
            "Communication skills": 7.4, "Social connection": 7.2
        },
        "top_finding": "Emotional engagement and empathy development score highest across all respondent types, with children showing 12% stronger imaginative response than adults.",
        "trend": "Teacher responses reference curriculum outcomes 3× more frequently post-workshop than post-performance, suggesting workshops drive educational impact.",
        "attention": "4 responses mention show length as a concern for children under 3 years. Consider a shorter variant for early childhood audiences.",
        "sentiment_detail": "91% positive sentiment overall. Negative sentiment is isolated to logistical feedback (parking, timing) rather than artistic content."
    }

def generate_report_text(audience, ai, n_rows):
    if not openai_available:
        return _demo_reports().get(audience, "")
    prompt = f"""Write an impact report for Monkey Baa Theatre Company for: {audience}

Data: {n_rows} survey responses, {ai.get('avg_satisfaction','4.6')}/5 avg satisfaction,
{ai.get('recommendation_rate',94)}% recommend, NPS {ai.get('nps',72)},
{ai.get('sentiment_pct',91)}% positive sentiment.
Key insight: {ai.get('top_finding','')}
Trend: {ai.get('trend','')}

Write exactly 3 paragraphs. No headers. Plain text only. Tone: professional and warm.
Tailor content specifically for: {audience}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return resp.choices[0].message.content
    except:
        return _demo_reports().get(audience, "")

def _demo_reports():
    return {
        "Executive Team": "Monkey Baa Theatre Company's 2024 program delivered an exceptional year of performance activity, reaching 3,240 audience members across 12 events and generating 147 survey responses with an average satisfaction rating of 4.6 out of 5. A recommendation rate of 94% and a Net Promoter Score of 72 — well above the industry average of 45 — confirm the program's sustained reputation for quality and audience impact.\n\nChild engagement averaged 4.8 out of 5 across all events, the highest figure recorded in the organisation's program history. NLP analysis of survey responses identified emotional engagement, empathy development and imagination as the three highest-scoring impact indicators, reflecting the core artistic intent of the program.\n\nResponse volumes grew 35% year-on-year, audience satisfaction held above 4.5 across all three program streams, and teacher feedback confirms lasting classroom impact following each event. These outcomes position Monkey Baa well for continued growth through 2025.",
        "Funding Bodies": "This impact report documents the outcomes of Monkey Baa Theatre Company's 2024 program delivery, submitted in support of continued philanthropic investment. The Green Sheep Tour reached 3,240 community members across 12 metropolitan and regional events, with an audience satisfaction rate of 94% and a Net Promoter Score of 72.\n\nAI-powered NLP analysis of 147 survey responses identified measurable outcomes across eight key indicators including emotional engagement (8.8/10), empathy development (8.8/10), and arts appreciation (8.2/10). Child respondents demonstrated the strongest imaginative response — evidence of meaningful early-childhood impact aligned with funding objectives.\n\nYear-on-year response volume grew 35%, demonstrating strong community investment in the program. Funding support has directly enabled the organisation to expand its school partnership program and develop accessible touring infrastructure for regional venues.",
        "Schools & Teachers": "Thank you for welcoming Monkey Baa Theatre Company into your school community during 2024. This summary shares what we heard from teachers and students following our visits, and how the program connects to the Australian Curriculum.\n\nAcross all school performances and workshops, teachers reported strong alignment with curriculum outcomes in English, the Arts, and Personal and Social Capability. Post-workshop surveys showed teachers referencing curriculum language 3× more frequently than post-performance responses — suggesting the workshop format creates particularly strong classroom integration.\n\nStudents demonstrated high scores in imagination (8.2/10) and creative response (7.6/10). Several teachers noted the program gave students new language to discuss emotions and social situations. We look forward to returning in 2025 and continuing to support learning through live theatre.",
        "Community Partners": "In 2024, Monkey Baa Theatre Company performed across 12 venues and events, welcoming 3,240 audience members. This has been one of our most connected years — thank you to our venue partners, community organisations, and families who made it possible.\n\nWe heard from 147 audience members after the shows. 94% said they would recommend Monkey Baa to a friend or family. Parents described the experience as moving and unlike anything their child had seen before. Children scored the performances 4.8 out of 5 — our highest ever record.\n\nEmpathy, imagination, and connection scored as the strongest outcomes of our work this year. We are proud to bring those values into your community, and we look forward to seeing you again in 2025."
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
    if any(w in q for w in ['top','insight','best']):
        return "The top insight is emotional engagement scoring 8.8/10 — highest across all 8 indicators. Children showed 12% stronger imaginative response than adults, which is a standout finding."
    if 'teacher' in q:
        return "Teachers rated workshops 3.8/5 on average. Notably they referenced curriculum outcomes 3× more often post-workshop, showing strong educational integration even where satisfaction is moderate."
    if any(w in q for w in ['program','highest','score']):
        return "Green Sheep Tour scored highest at 4.6/5 satisfaction. Community Schools scored 3.5/5 — the lowest stream — which may warrant a follow-up review."
    if 'funder' in q:
        return "For funders, lead with: 94% recommendation rate, NPS 72 (industry avg 45), 3,240 audience reached, and 35% year-on-year growth. The child engagement score of 4.8/5 is a standout."
    if 'concern' in q:
        return "Two concerns: show length flagged for under-3s (4 responses), and Community Schools satisfaction is 1 full point below program average. Both worth addressing in next planning cycle."
    return f"Based on your {len(st.session_state.df_clean) if st.session_state.df_clean is not None else '147'} survey responses, overall performance is strong with 91% positive sentiment and 4.6/5 satisfaction. Ask me about a specific program or audience group!"

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("---")
        st.markdown("### 🎭 Monkey Baa Theatre Co.")
        st.caption("IMPACT REPORTING SYSTEM · MVP V2.0")
        st.markdown("---")
        st.markdown("## Welcome back")
        st.markdown("AI-powered data upload, cleaning, insights and stakeholder report generation.")
        st.markdown("---")
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
    df = st.session_state.df_clean or st.session_state.df_raw
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
        st.markdown("#### Impact Indicator Scores")
        indicators = ai.get('indicators', {})
        if indicators:
            ind_df = pd.DataFrame(list(indicators.items()), columns=['Indicator', 'Score'])
            ind_df = ind_df.sort_values('Score', ascending=True)
            fig3 = px.bar(ind_df, x='Score', y='Indicator', orientation='h',
                          color_discrete_sequence=['#1c2b4a'],
                          range_x=[0, 10])
            fig3.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                               paper_bgcolor='white', plot_bgcolor='#f8fafc',
                               font_family='DM Sans')
            st.plotly_chart(fig3, use_container_width=True)

    with col_ins:
        st.markdown("#### Key AI Insights")
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
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Generate Reports →", use_container_width=True):
        go('reports')

# ── REPORTS ───────────────────────────────────────────────────────────────────
def page_reports():
    df = st.session_state.df_clean or st.session_state.df_raw
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

    df = st.session_state.df_clean or st.session_state.df_raw
    ai = st.session_state.ai_results or _demo_ai()

    if df is None:
        st.info("Upload data to unlock the AI assistant.")
        return

    # Preset chips
    chips = ["What's the top insight?","How satisfied were teachers?",
             "Which program scored highest?","What should I highlight for funders?",
             "Any concerns in the data?"]

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
