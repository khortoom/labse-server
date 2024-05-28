from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# Load the model
laBSE_model = SentenceTransformer("setu4993/LaBSE")

# Initialize the Flask app
app = Flask(__name__)


@app.route("/encode", methods=["POST"])
def encode_query():
    data = request.json

    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    encoded_query = laBSE_model.encode(query).tolist()

    return jsonify({"vector": encoded_query})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
    # Run the Flask app
    # app.run(host="0.0.0.0", port=5000)
