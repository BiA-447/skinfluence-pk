"""
core.py  -  all the logic (no Streamlit code here).
It only LOADS the saved FAISS index. It never re-embeds products or knowledge.
The only thing embedded at runtime is the user's short profile text.
"""
import json
import re

import faiss
from rapidfuzz import fuzz

GROQ_MODEL = "llama-3.3-70b-versatile"      # fallback: "llama-3.1-8b-instant"

# product category (from CSV) -> routine "bucket"
BUCKETS = {
    "cleanser": ["Cleanser"],
    "moisturizer": ["Moisturizer"],
    "sunscreen": ["Sunscreen"],
    "treatment": ["Serum", "Essence", "Focused treatment", "Pharmacy"],
}
PER_BUCKET = {"cleanser": 4, "moisturizer": 4, "sunscreen": 4, "treatment": 6}

# Medicines: still in the index (so we can recognise them), but NEVER recommended.
BLOCKED_RE = re.compile(
    r"tretinoin|clindamycin|isotretinoin|hydroquinone|clobetasol|betamethasone|mometasone",
    re.I)
# Strong actives: removed for very sensitive / severe cases, and counted for warnings.
STRONG_RE = re.compile(
    r"\b(retinol|retinal\w*|adapalene|benzoyl peroxide|glycolic|salicylic|aha|bha|lactic|mandelic)\b",
    re.I)

ESCALATION_WORDS = ["swelling", "swollen", "pus", "allergic", "allergy", "open wound", "bleeding",
                    "rapidly", "getting worse", "spreading", "eye", "eyelid", "blister", "fever"]

VERDICTS = ["Keep", "Review", "Consider Replacing", "Unknown"]


# ------------------------------------------------------------------ loading
def load_index(index_path="index/faiss.index", chunks_path="index/chunks.json"):
    index = faiss.read_index(index_path)
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return index, data["chunks"], data["model"]


# ------------------------------------------------------------------ helpers
def ingredients_text(meta):
    return " ".join(meta.get("key_ingredients", [])) + " " + meta.get("description", "")


def is_blocked(meta):
    return bool(BLOCKED_RE.search(ingredients_text(meta)))


def is_strong(meta):
    return bool(STRONG_RE.search(ingredients_text(meta)))


def bucket_of(category):
    for b, cats in BUCKETS.items():
        if category in cats:
            return b
    return None


def price_text(meta):
    return f"Rs. {meta['price']:,.0f}" if meta.get("price") else "Price not available"


# ------------------------------------------------------------------ safety (plain Python, no LLM)
def safety_flags(profile):
    flags, must_escalate = [], False
    if profile["acne"] == "Severe / Persistent Concern":
        flags.append("Severe or persistent acne should be evaluated by a dermatologist. "
                     "We will keep suggestions gentle and avoid strong actives.")
        must_escalate = True
    if profile["irritation"] in ("Frequently", "Very Sensitive") or profile["redness"] == "Significant":
        flags.append("Your skin looks very sensitive or reactive. We will keep the routine minimal "
                     "and avoid strong actives. Consider professional advice.")
    note = (profile.get("extra_note") or "").lower()
    hit = [w for w in ESCALATION_WORDS if w in note]
    if hit:
        flags.append("What you described (" + ", ".join(hit) + ") may need professional care. "
                     "Please see a doctor or dermatologist rather than relying on an app.")
        must_escalate = True
    return flags, must_escalate


def gentle_only(profile):
    return (profile["acne"] == "Severe / Persistent Concern"
            or profile["irritation"] in ("Frequently", "Very Sensitive")
            or profile["redness"] == "Significant"
            or (profile["skin_type"] == "Sensitive / Unsure" and profile["irritation"] != "None"))


# ------------------------------------------------------------------ retrieval
def profile_query(p):
    return (f"{p['skin_type']} skin. Acne: {p['acne']}. Redness: {p['redness']}. "
            f"Sensitivity: {p['irritation']}. Goals: {', '.join(p['goals']) or 'general care'}. "
            f"Concerns: {', '.join(p['concerns']) or 'none listed'}.")


def retrieve(profile, index, chunks, embedder):
    """Embeds ONLY the query, searches the saved FAISS index, then filters."""
    q = embedder.encode([profile_query(profile)], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, index.ntotal)        # tiny index -> rank everything

    gentle = gentle_only(profile)
    grouped = {b: [] for b in BUCKETS}
    knowledge = []
    for i in ids[0]:
        c = chunks[i]
        if c["type"] == "knowledge":
            if len(knowledge) < 5:
                knowledge.append(c)
            continue
        m = c["meta"]
        b = bucket_of(m["category"])
        if b is None or len(grouped[b]) >= PER_BUCKET[b]:
            continue
        if m["origin"] != "Pakistan" and not profile["allow_imported"]:
            continue
        if is_blocked(m):                               # medicines are never recommended
            continue
        if gentle and is_strong(m):
            continue
        if m.get("price") and m["price"] > profile["budget"] and profile["budget"] > 0:
            continue
        grouped[b].append(m)
    return grouped, knowledge


GENERIC_WORDS = {"cleanser", "cleansing", "cream", "moisturizer", "moisturiser", "moisturizing",
                 "sunscreen", "sunblock", "serum", "face", "wash", "gel", "lotion", "toner",
                 "spf", "daily", "skin", "essence", "treatment"}
SKIP_WORDS = {"the", "facial", "and", "by", "for"}


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in SKIP_WORDS]


def match_existing(text, chunks):
    """Match the user's own products to our database.
    Every distinctive word (brand / product name) must be found, and most words must match."""
    products = {c["meta"]["product_id"]: c["meta"] for c in chunks if c["type"] == "product"}
    cand_tokens = {pid: _tokens(f"{m['name']} {m['brand']}") for pid, m in products.items()}
    matched, unmatched = [], []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name = re.split(r"\s+[-–|]\s+", line)[0]
        user = _tokens(name)
        distinct = [w for w in user if w not in GENERIC_WORDS]
        best, best_key = None, None
        if distinct:
            for pid, ctoks in cand_tokens.items():
                found = {w: max((fuzz.ratio(w, c) for c in ctoks), default=0) >= 85 for w in user}
                coverage = sum(found.values()) / len(user)
                if coverage >= 0.67 and all(found[w] for w in distinct):
                    key = (coverage, -len(ctoks))          # best coverage, then closest length
                    if best_key is None or key > best_key:
                        best, best_key = pid, key
        if best:
            if products[best] not in matched:
                matched.append(products[best])
        else:
            unmatched.append(line)
    return matched, unmatched


# ------------------------------------------------------------------ prompt + LLM
def product_line(m):
    return (f"[{m['product_id']}] {m['name']} ({m['brand']}) | {m['origin']} | {price_text(m)} | "
            f"Ingredients: {', '.join(m['key_ingredients']) or 'n/a'} | "
            f"Skin types: {', '.join(m['skin_types']) or 'n/a'} | "
            f"Concerns: {', '.join(m['concerns']) or 'n/a'} | "
            f"Strong active: {'yes' if is_strong(m) else 'no'}")


def build_messages(profile, grouped, knowledge, matched, unmatched, extra=""):
    options = ""
    for b, items in grouped.items():
        options += f"\n## {b.upper()} OPTIONS\n"
        options += "\n".join(product_line(m) for m in items) or "(none available)"
        options += "\n"
    owned = "\n".join(product_line(m) for m in matched) or "(none matched)"
    kb = "\n\n".join(c["text"][:600] for c in knowledge)
    shown = {k: v for k, v in profile.items() if k != "extra_note"}

    system = (
        "You are a careful skincare assistant for users in Pakistan.\n"
        "RULES:\n"
        "1. Recommend ONLY products listed under OPTIONS or EXISTING PRODUCTS, using their product_id. "
        "Never invent products, prices, sellers or links.\n"
        "2. Do not diagnose or prescribe. Do not claim to cure anything.\n"
        "3. If an existing product is suitable, keep it (use_existing) instead of buying a new one.\n"
        "4. Total price of NEW purchases must stay within the budget.\n"
        "5. Use at most ONE strong-active product in the whole routine (zero if the skin is sensitive). "
        "Sunscreen only in the morning.\n"
        "6. Base 'why' only on the product details and knowledge given. Keep each 'why' under 25 words.\n"
        "Reply with JSON ONLY, in exactly this shape:\n"
        '{"am":[{"step":"Cleanser","product_id":"P001 or null","use_existing":"P002 or null","why":"..."}],'
        '"pm":[...],'
        '"audit":[{"product_id":"P002","verdict":"Keep|Review|Consider Replacing|Unknown","reason":"..."}],'
        '"notes":"one or two friendly sentences"}'
        "\nAM steps: Cleanser, Treatment (optional), Moisturizer, Sunscreen. "
        "PM steps: Cleanser, Treatment (optional), Moisturizer. "
        "'audit' lists ONLY the EXISTING PRODUCTS shown."
    )
    user = (f"USER PROFILE:\n{json.dumps(shown)}\n\n"
            f"EXISTING PRODUCTS (already owned, cost Rs. 0):\n{owned}\n\n"
            f"OPTIONS TO BUY:\n{options}\n\nSKINCARE KNOWLEDGE:\n{kb}\n{extra}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_llm(client, messages):
    resp = client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.2,
        response_format={"type": "json_object"}, messages=messages)
    return json.loads(resp.choices[0].message.content)


# ------------------------------------------------------------------ checking the LLM answer
def clean_result(result, lookup, owned_ids):
    """Drops hallucinated IDs, fixes verdicts. Returns a safe structure."""
    def clean_steps(steps, allow_sunscreen):
        out = []
        for s in steps or []:
            step = str(s.get("step", "")).strip()
            if step.lower() == "sunscreen" and not allow_sunscreen:
                continue
            pid, ex = s.get("product_id"), s.get("use_existing")
            m = lookup.get(pid) if pid else None
            e = lookup.get(ex) if ex and ex in owned_ids else None
            if m or e:
                out.append({"step": step, "product": m, "existing": e, "why": s.get("why", "")})
        return out
    return {
        "am": clean_steps(result.get("am"), True),
        "pm": clean_steps(result.get("pm"), False),
        "audit_raw": result.get("audit") or [],
        "notes": result.get("notes", ""),
    }


def new_purchases(routine, owned_ids):
    seen = {}
    for part in ("am", "pm"):
        for s in routine[part]:
            m = s["product"]
            if m and m["product_id"] not in owned_ids:
                seen[m["product_id"]] = m
    return list(seen.values())


def total_cost(routine, owned_ids):
    return sum(m["price"] or 0 for m in new_purchases(routine, owned_ids))


def strong_count(routine):
    ids = {}
    for part in ("am", "pm"):
        for s in routine[part]:
            for m in (s["product"], s["existing"]):
                if m and is_strong(m):
                    ids[m["product_id"]] = m
    return len(ids)


def build_audit(audit_raw, matched, unmatched):
    by_id = {a.get("product_id"): a for a in audit_raw}
    rows = []
    for m in matched:
        a = by_id.get(m["product_id"], {})
        v = a.get("verdict") if a.get("verdict") in VERDICTS else "Unknown"
        rows.append({"product": f"{m['name']} ({m['brand']})", "verdict": v,
                     "reason": a.get("reason") or "Not enough information to judge."})
    for line in unmatched:                  # products not in our database are always Unknown
        rows.append({"product": line, "verdict": "Unknown",
                     "reason": "This product is not in our database, so we cannot judge it."})
    return rows


def generate_routine(profile, index, chunks, embedder, client):
    """The whole pipeline. Returns (routine, audit, meta_info) or raises."""
    grouped, knowledge = retrieve(profile, index, chunks, embedder)
    matched, unmatched = match_existing(profile.get("current_routine", ""), chunks)
    owned_ids = {m["product_id"] for m in matched}
    lookup = {m["product_id"]: m for items in grouped.values() for m in items}
    lookup.update({m["product_id"]: m for m in matched})

    if not any(grouped.values()) and not matched:
        return None, [], {"error": "No matching products found. Try a higher budget or include imported products."}

    messages = build_messages(profile, grouped, knowledge, matched, unmatched)
    routine = clean_result(call_llm(client, messages), lookup, owned_ids)

    # budget check done in Python (LLMs are bad at arithmetic) - one retry if over
    cost = total_cost(routine, owned_ids)
    if profile["budget"] > 0 and cost > profile["budget"]:
        extra = (f"\nYour last routine cost Rs. {cost:,.0f} in new purchases, which is over the budget of "
                 f"Rs. {profile['budget']:,.0f}. Choose cheaper options or keep more existing products.")
        routine = clean_result(call_llm(client, build_messages(
            profile, grouped, knowledge, matched, unmatched, extra)), lookup, owned_ids)
        cost = total_cost(routine, owned_ids)

    audit = build_audit(routine["audit_raw"], matched, unmatched)
    info = {
        "cost": cost,
        "over_budget": profile["budget"] > 0 and cost > profile["budget"],
        "strong_count": strong_count(routine),
        "new": new_purchases(routine, owned_ids),
        "knowledge": knowledge,
    }
    return routine, audit, info
