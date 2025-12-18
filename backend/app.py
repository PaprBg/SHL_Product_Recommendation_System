from flask import Flask, request, jsonify
from search_assessments import find_assessments

app = Flask(__name__)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    query = data.get("query", "")
    k = data.get("k", 5)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = find_assessments(query, k)

    return jsonify({
        "query": query,
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=True)
