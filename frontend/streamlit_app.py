import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🔍 SHL Assessment Recommendation System")
st.write("Semantic search over SHL assessments using NLP")

query = st.text_area(
    "Enter job description or assessment requirement:",
    height=120
)

k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching assessments..."):
            response = requests.post(
                "http://127.0.0.1:5000/recommend",
                json={"query": query, "k": k}
            )

            if response.status_code == 200:
                data = response.json()
                results = data["results"]

                st.success(f"Found {len(results)} assessments")

                for i, r in enumerate(results, 1):
                    with st.container():
                        st.subheader(f"{i}. {r['product_name']}")
                        st.write(f"**Test Type:** {r['test_type']}")
                        st.write(f"**Job Levels:** {r['Job_levels']}")
                        st.write(f"**Remote Testing:** {r['remote_testing_available']}")
                        st.write(f"**Score:** {r['Score']}")
                        st.markdown(f"[View Assessment]({r['product_url']})")
                        st.divider()
            else:
                st.error("Backend error")
