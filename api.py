from flask import Flask, request, jsonify, send_file, url_for
from flask import render_template
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils_model import load_model_checkpoint, predict_sentiment
import torchtext.vocab as torch_vocab
from dataset import Vocabulary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===== Load vocab =====
word_embedding = torch_vocab.Vectors(
    name="./downloads/vi_word2vec.txt",
    unk_init=torch.Tensor.normal_
)
vocab = Vocabulary()
words_list = list(word_embedding.stoi.keys())
for word in words_list:
    vocab.add(word)

# ===== Load model =====
models = load_model_checkpoint(word_embedding, vocab, DEVICE)

# ===== Init API =====
static_dir = "./static/"
os.makedirs(static_dir, exist_ok=True)

app = Flask(__name__)


@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result, probability = predict_sentiment(
        models[model_name], text, vocab, DEVICE)
    print(text, result, probability)
    return jsonify({
        "prediction": int(result),
        "confidence": float(probability)
    })


# ====== ROUTE: TRAINING CURVES ======
@app.route("/train_plot/<metric>")
def train_plot(metric):
    # Lấy giá trị model từ query string, ví dụ ?model=lstm
    model = request.args.get("model", "lstm")
    """
    Tạo và lưu biểu đồ loss hoặc acc vào thư mục static
    metric: "loss" | "acc"
    """
    log_path = f"training_history/train_log_{model}.csv"
    if not os.path.exists(log_path):
        return jsonify({"error": f"File train_log_{model}.csv chưa tồn tại"}), 404

    df = pd.read_csv(log_path)
    if df.empty:
        return jsonify({"error": f"File train_log_{model}.csv rỗng"}), 400

    if metric not in ["loss", "acc"]:
        return jsonify({"error": "metric must be 'loss' or 'acc'"}), 400

    plt.figure(figsize=(6, 4))
    if metric == "loss":
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker='x')
        plt.ylabel("Loss")
    else:
        plt.plot(df["epoch"], df["train_acc"], label="Train Acc", marker='o')
        plt.plot(df["epoch"], df["val_acc"], label="Val Acc", marker='x')
        plt.ylabel("Accuracy")

    plt.xlabel("Epoch")
    plt.title(f"Training {metric.capitalize()}")
    plt.legend()
    plt.grid(True)

    # Lưu file vào static/
    file_path = os.path.join(static_dir, f"{metric}.png")

    plt.savefig(file_path, format="png", bbox_inches="tight")
    plt.close()

    # Trả về đường dẫn tĩnh (static)
    img_url = url_for("static", filename=f"{metric}.png", _external=False)
    return jsonify({"image_url": img_url})


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=False)
