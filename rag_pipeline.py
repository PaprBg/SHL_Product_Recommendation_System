import os
import json
import google.generativeai as genai
from backend.search_assessments import find_assessments

from dotenv import load_dotenv
load_dotenv()



# -------------------- GEMINI SETUP --------------------

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-pro")


# -------------------- STEP 1: QUERY UNDERSTANDING --------------------

def extract_structured_query(user_input: str) -> dict:
    """
    Uses Gemini to extract structured fields from unstructured input
    """

    prompt = f"""
You are an NLP system that extracts structured information.

From the following input, extract:
- job_role
- job_level
- skills
- remote_testing_required (Yes/No)
- assessment_preferences (if any)

Return ONLY valid JSON.

Input:
{user_input}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback if model adds text
        return {}


# -------------------- STEP 2: BUILD REFINED QUERY --------------------

def build_refined_query(structured_data: dict) -> str:
    """
    Convert structured fields into a clean semantic query
    """

    parts = []

    if structured_data.get("job_role"):
        parts.append(structured_data["job_role"])

    if structured_data.get("skills"):
        parts.append(", ".join(structured_data["skills"]))

    if structured_data.get("job_level"):
        parts.append(structured_data["job_level"])

    if structured_data.get("remote_testing_required") == "Yes":
        parts.append("remote testing")

    return " ".join(parts)


# -------------------- STEP 3: RETRIEVE CANDIDATES --------------------

def retrieve_candidates(refined_query: str, k=5):
    return find_assessments(refined_query, k=k)


# -------------------- STEP 4: GEMINI RE-RANK + EXPLAIN --------------------

def explain_and_filter(user_input, structured_data, retrieved_results):

    prompt = f"""
User requirement:
{user_input}

Structured understanding:
{json.dumps(structured_data, indent=2)}

Retrieved assessments:
{json.dumps(retrieved_results, indent=2)}

Task:
1. Filter out irrelevant assessments
2. Rank the remaining ones by relevance
3. Explain why each assessment matches the requirement

Return response in clear natural language.
"""

    response = model.generate_content(prompt)
    return response.text


# -------------------- MAIN PIPELINE --------------------

def run_rag_pipeline(user_input: str, k=5):

    print("\n🔹 Extracting structured features...")
    structured_data = extract_structured_query(user_input)
    print(structured_data)

    print("\n🔹 Building refined query...")
    refined_query = build_refined_query(structured_data)
    print("Refined Query:", refined_query)

    print("\n🔹 Retrieving assessments...")
    retrieved_results = retrieve_candidates(refined_query, k)

    print("\n🔹 Generating explanation...")
    final_response = explain_and_filter(
        user_input,
        structured_data,
        retrieved_results
    )

    return final_response


# -------------------- CLI TEST --------------------

if __name__ == "__main__":

    user_input = """
We are hiring entry-level finance graduates.
Candidates should have accounting knowledge and should be able to take assessments remotely.
"""

    output = run_rag_pipeline(user_input, k=5)
    print("\n✅ FINAL RESPONSE\n")
    print(output)
