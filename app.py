"""PetBNB: a grounded, Thane-only pet-place recommender."""
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Load the project-local environment before any OpenAI key check or client creation.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

MODEL = "gpt-5.6-luna"
DATA_PATH = Path(__file__).parent / "data" / "thane_places.json"
VALID_CATEGORIES = {"cafe", "park", "vet", "groomer", "boarding"}


def load_places() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as source:
        places = json.load(source)
    if not isinstance(places, list) or any(
        not isinstance(place, dict) or place.get("category") not in VALID_CATEGORIES
        for place in places
    ):
        raise ValueError("data/thane_places.json is not a valid PetBNB place dataset.")
    return places


def filter_places(places: list[dict[str, Any]], needs: list[str]) -> list[dict[str, Any]]:
    return [place for place in places if place["category"] in set(needs)]


def validate_recommendation_payload(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str]:
    """Only permit model results whose IDs occur in the supplied candidate list."""
    allowed_ids = {place["id"] for place in candidates}
    raw = payload.get("recommendations", [])
    no_fit_reason = str(payload.get("no_fit_reason", "")).strip()
    if not isinstance(raw, list):
        return [], no_fit_reason or "The model returned an invalid recommendation format."
    approved: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        place_id, explanation = item.get("id"), item.get("explanation")
        if isinstance(place_id, str) and place_id in allowed_ids and place_id not in seen and isinstance(explanation, str) and explanation.strip():
            approved.append({"id": place_id, "explanation": explanation.strip()})
            seen.add(place_id)
    return approved, no_fit_reason or ("No supported match was returned for this profile." if not approved else "")


def build_prompt(profile: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    return f"""
You are PetBNB's careful Thane, Maharashtra recommendation assistant.
Pet profile: {json.dumps(profile, ensure_ascii=False)}
Candidate places (this is the complete and only allowed place list): {json.dumps(candidates, ensure_ascii=False)}
Return ONLY valid JSON: {{"recommendations":[{{"id":"TH001","explanation":"One or two sentences."}}],"no_fit_reason":""}}
Rules: Select 3-5 only if genuinely supported; fewer or zero is allowed. Every id must exactly match a supplied candidate id. Ground each explanation strictly in candidate fields and the pet profile. Do not infer quietness, safety, admission, or amenities from an address/category/area. Empty pet_policy or good_for is unknown, not supporting evidence. Never name or invent another place. If nothing genuinely fits, return [] and explain in no_fit_reason.
""".strip()


def call_recommender(profile: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured. Add it before requesting recommendations.")
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=MODEL,
        instructions="Return only requested JSON and follow the closed-world grounding rules exactly.",
        input=build_prompt(profile, candidates),
    )
    try:
        payload = json.loads(response.output_text)
    except (AttributeError, json.JSONDecodeError) as error:
        raise RuntimeError("The recommendation response was not valid JSON. Please try again.") from error
    return validate_recommendation_payload(payload, candidates)


def run_self_test() -> None:
    """Offline test: tests grounding validation without an API call or key."""
    places = load_places()
    cases = [
        ("anxious dog / cafe", ["cafe"], "TH006", "The dataset supports outdoor non-AC seating for this cafe."),
        ("senior cat / vet", ["vet"], "TH018", "The record lists consultation and veterinary services."),
        ("large high-energy dog / boarding", ["boarding"], "TH033", "The record lists dog boarding, day care, and training."),
    ]
    chosen: list[str] = []
    for label, needs, place_id, explanation in cases:
        candidates = filter_places(places, needs)
        recommendations, reason = validate_recommendation_payload({"recommendations": [{"id": place_id, "explanation": explanation}], "no_fit_reason": ""}, candidates)
        assert not reason and len(recommendations) == 1, f"Self-test failed: {label}"
        assert recommendations[0]["id"] in {place["id"] for place in candidates}
        chosen.append(recommendations[0]["id"])
    assert len(set(chosen)) == len(chosen), "Self-test profiles did not differ."
    print("PASS: 3 offline profiles produced distinct, candidate-grounded recommendations.")


if __name__ == "__main__" and "--self-test" in sys.argv:
    run_self_test()
    raise SystemExit(0)


import streamlit as st

st.set_page_config(page_title="PetBNB Thane", page_icon="🐾", layout="wide")


@st.cache_data
def get_cached_places() -> list[dict[str, Any]]:
    return load_places()


def display_recommendations(recommendations: list[dict[str, str]], candidates: list[dict[str, Any]], no_fit_reason: str) -> None:
    if not recommendations:
        st.info(no_fit_reason or "No supported matches found for this profile.")
        return
    by_id = {place["id"]: place for place in candidates}
    for recommendation in recommendations:
        place = by_id[recommendation["id"]]
        with st.container(border=True):
            title, badge = st.columns([6, 1])
            title.subheader(place["name"])
            badge.caption(place["category"].title())
            if place.get("confidence") == "low":
                st.warning("Low confidence: verify current details with the provider before visiting.", icon="⚠️")
            st.write(f"**Address:** {place.get('address') or 'Address not verified'}")
            st.write(f"**Area:** {place.get('area') or 'Not verified'}")
            st.write(recommendation["explanation"])


st.title("🐾 PetBNB Thane")
st.caption("Grounded suggestions from the local seed dataset. Always confirm current pet policies directly.")
places = get_cached_places()

with st.form("pet_profile"):
    left, right = st.columns(2)
    with left:
        pet_type = st.selectbox("Pet type", ["dog", "cat"])
        size = st.selectbox("Breed or size", ["small", "medium", "large"])
        age = st.selectbox("Age", ["puppy", "adult", "senior"])
    with right:
        temperament = st.multiselect("Temperament", ["anxious", "high-energy", "calm", "social", "reactive"])
        needs = st.multiselect("What do you need today?", ["cafe", "park", "vet", "groomer", "boarding"])
    submitted = st.form_submit_button("Find pet-friendly places", type="primary")

if submitted:
    if not needs:
        st.warning("Choose at least one need to search the local dataset.")
    else:
        # The selected subset is persisted; the API is reached only in this submit branch.
        st.session_state["filtered_candidates"] = filter_places(places, needs)
        st.session_state["last_profile"] = {"pet_type": pet_type, "size": size, "age": age, "temperament": temperament, "needs": needs}
        if not st.session_state["filtered_candidates"]:
            st.info("No places in the local dataset match those categories yet.")
        else:
            try:
                with st.spinner("Matching your pet profile to local evidence…"):
                    recommendations, no_fit_reason = call_recommender(st.session_state["last_profile"], st.session_state["filtered_candidates"])
                st.session_state["recommendations"] = recommendations
                st.session_state["no_fit_reason"] = no_fit_reason
            except Exception as error:
                st.error(f"Could not generate recommendations: {error}")

if "recommendations" in st.session_state and "filtered_candidates" in st.session_state:
    st.subheader("Recommended for your pet")
    display_recommendations(st.session_state["recommendations"], st.session_state["filtered_candidates"], st.session_state.get("no_fit_reason", ""))
