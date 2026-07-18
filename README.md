# 🐾 PetBNB Thane

A grounded, Thane-only pet-place recommender built for OpenAI Build Week (Track: Apps for Your Life).

## What it does

Pet parents in Thane struggle to find reliable, pet-friendly cafes, parks, vets, groomers, 
and boarding options — information is scattered across word-of-mouth and social media, with 
no single trustworthy source. PetBNB takes a pet's profile (type, breed, age, temperament) 
and today's need, and returns relevant places from a curated local dataset — with GPT-5.6 
explaining *why* each place fits, grounded strictly in verified data, not invented claims.

## How it works

1. User fills out a pet profile and selects a need (cafe, park, vet, groomer, boarding).
2. The app filters a local seed dataset (`data/thane_places.json`) to relevant candidates.
3. GPT-5.6 (`gpt-5.6-luna`) reasons over the candidates and the pet profile to select the 
   top 3–5 genuine matches, with a closed-world constraint: it can only recommend places 
   that exist in the dataset, and can only justify a match using fields actually present 
   on that place's record — no inferred amenities, no invented places.
4. If nothing genuinely fits, the app says so rather than forcing a low-quality match.

## Key engineering decision: grounding against hallucination

The biggest risk with an LLM-powered recommender is confident-sounding but false 
information — a real risk for a local business directory. `validate_recommendation_payload()` 
cross-checks every model-returned place ID against the actual candidate list before display; 
any ID not in the dataset is silently dropped. The prompt itself (`build_prompt()`) explicitly 
forbids inferring unstated attributes (e.g., quietness, safety, breed restrictions) from 
category or address alone. This was tested via an offline self-test (`app.py --self-test`) 
covering three distinct pet profiles, confirming grounded, non-overlapping recommendations 
without needing a live API call.

## How Codex accelerated the build

Codex was used to:
- Compile the initial Thane places dataset from publicly available sources (blogs, directory 
  listings), structured to a consistent schema with a confidence flag for uncertain entries.
- Scaffold the recommendation pipeline, including the grounding/validation logic that 
  prevents the model from recommending places outside the verified dataset.
- Debug environment configuration (`.env` loading, dependency resolution) during local setup.
- Built a one-time build script (`scripts/fetch_breeds.py`) to pull standard dog/cat breed 
  lists from TheDogAPI/TheCatAPI, stored as static local data — avoids inventing an arbitrary 
  breed list and keeps the app itself free of runtime network dependencies.

Primary build thread `/feedback` Session ID: `[ADD YOUR SESSION ID HERE]`

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root: