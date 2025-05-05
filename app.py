from flask import Flask, render_template, request, jsonify
from model_utils import DiseaseMedicineRecommender

app = Flask(__name__)
recommender = DiseaseMedicineRecommender("final.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["message"]
    medicines = recommender.recommend(user_input)
    return jsonify({"response": ", ".join(medicines)})

if __name__ == "__main__":
    app.run(debug=True)
