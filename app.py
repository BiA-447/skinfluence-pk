import os

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

import core

st.set_page_config(page_title="Skinfluence PK", page_icon="🧴", layout="centered")

DISCLAIMER = (
    "Disclaimer: Skinfluence PK provides general skincare information and product recommendations "
    "for informational purposes only. It does not provide medical advice, diagnose skin conditions, "
    "prescribe treatments, or replace a qualified dermatologist or healthcare professional. "
    "Recommendations are based on the information you provide and available product data. "
    "If you have severe, persistent, or worsening skin concerns, consult a qualified healthcare professional."
)


# ---------- loaded ONCE per server, not per query ----------
@st.cache_resource
def load_resources():
    index, chunks, model_name = core.load_index()
    embedder = SentenceTransformer(model_name)   # same model that built the index
    return index, chunks, embedder


@st.cache_resource
def get_client():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


index, chunks, embedder = load_resources()
client = get_client()

# ---------- header ----------
st.title("🧴 Skinfluence PK")
st.write("Personalised skincare recommendations built around what you can actually buy in Pakistan.")
st.info(DISCLAIMER)

if client is None:
    st.error("Groq API key not found. Add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# ---------- questionnaire ----------
with st.form("profile"):
    st.subheader("1. Your skin")
    age = st.number_input("Age", 10, 90, 22)
    skin_type = st.selectbox("Skin type", ["Dry", "Oily", "Combination", "Normal", "Sensitive / Unsure"])
    c1, c2, c3 = st.columns(3)
    acne = c1.selectbox("Acne", ["None", "Occasional", "Frequent", "Severe / Persistent Concern"])
    redness = c2.selectbox("Redness", ["None", "Mild", "Frequent", "Significant"])
    irritation = c3.selectbox("Irritation", ["None", "Sometimes", "Frequently", "Very Sensitive"])
    goals = st.multiselect("Skin goals", [
        "Reduce breakouts", "Improve hydration", "Improve skin texture",
        "Reduce appearance of dark spots/acne marks", "Control excess oil", "Support skin barrier",
        "Brighten dull-looking skin", "Maintain healthy-looking skin", "Simplify skincare routine",
        "Early anti-aging / prevention"])
    concerns = st.multiselect("Extra concerns (optional)", [
        "Blackheads", "Whiteheads / closed comedones", "Post-acne marks", "Uneven texture", "Dryness",
        "Flakiness", "Excess oil", "Dehydrated skin", "Dark circles", "Uneven skin tone",
        "Roughness", "Enlarged-looking pores"])
    extra_note = st.text_input("Anything else we should know? (optional)")

    st.subheader("2. Your current routine")
    no_routine = st.checkbox("I don't currently have a skincare routine")
    current = st.text_area("Products you use now (one per line, e.g. 'CeraVe Foaming Cleanser - cleanser')",
                           height=110)

    st.subheader("3. Preferences")
    budget = st.number_input("Skincare budget for NEW purchases (Rs.)", 0, 200000, 4000, step=500)
    where = st.radio("Where should we look?", ["Pakistan only", "Include imported products"])
    go = st.form_submit_button("Build my routine", type="primary")

# ---------- run ----------
if go:
    profile = dict(age=age, skin_type=skin_type, acne=acne, redness=redness, irritation=irritation,
                   goals=goals, concerns=concerns, extra_note=extra_note,
                   current_routine="" if no_routine else current,
                   budget=budget, allow_imported=(where != "Pakistan only"))

    flags, must_escalate = core.safety_flags(profile)
    for f in flags:
        st.warning(f)

    with st.spinner("Searching products and building your routine..."):
        try:
            routine, audit, info = core.generate_routine(profile, index, chunks, embedder, client)
        except Exception as e:
            st.error(f"The AI service is busy or returned an error. Please try again in a minute. ({e})")
            st.stop()

    if routine is None:
        st.error(info["error"])
        st.stop()

    def show_step(s):
        m = s["product"] or s["existing"]
        owned = s["product"] is None
        with st.container(border=True):
            tag = "✅ You already own this" if owned else "🛒 New purchase"
            st.markdown(f"**{s['step']}: {m['name']}**  \n{m['brand']} · {tag}")
            seller = m.get("seller") or "not listed"
            st.caption(f"Category: {m['category']} · Origin: {m['origin']} · "
                       f"Price: {core.price_text(m)} · Seller: {seller} · Last checked: {m['last_verified']}")
            st.write(f"**Why this product?** {s['why']}")
            if not owned and m.get("purchase_url"):
                st.link_button("Buy Product", m["purchase_url"])
            elif not owned:
                st.caption("Direct purchase link not available yet.")

    st.header("Your personalised routine")
    st.subheader("☀️ Morning")
    for s in routine["am"]:
        show_step(s)
    st.subheader("🌙 Evening")
    for s in routine["pm"]:
        show_step(s)

    st.subheader("Budget")
    st.write(f"New purchases: **Rs. {info['cost']:,.0f}** of your Rs. {budget:,.0f} budget.")
    if info["over_budget"]:
        st.warning("This routine is slightly over your budget. You can start with the cleanser, "
                   "moisturizer and sunscreen first and add the treatment later.")
    if info["strong_count"] > 1:
        st.warning("This routine contains more than one strong active. Introduce them one at a time, "
                   "and stop if your skin becomes irritated.")

    if audit:
        st.subheader("Your existing routine")
        icon = {"Keep": "✅", "Review": "🟡", "Consider Replacing": "🔁", "Unknown": "❔"}
        for a in audit:
            st.write(f"{icon[a['verdict']]} **{a['product']}**: {a['verdict']}. {a['reason']}")

    if routine["notes"]:
        st.info(routine["notes"])
    st.caption("New products: patch test first and add only one new active at a time.")
    st.caption(DISCLAIMER)
