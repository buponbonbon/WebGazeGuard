import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nlp.classifier import PhoBERTClassifier

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = PhoBERTClassifier.load(
        "nlp/checkpoints/best_model_PhoBert.pt",
        device=device
    )


    print("Loading test dataset...")
    df = pd.read_csv("nlp/test.csv")

    texts = df["description"].astype(str).tolist()

    # change label from 1-4 to 0-3
    y_true = df["label"].astype(int) - 1

    preds = []

    print("Running predictions...")
    for text in texts:
        pred, _ = model.predict(text, device=device)
        preds.append(pred[0])

    df["pred"] = preds

    print("\n=== NLP Evaluation ===")

    acc = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="macro")
    recall = recall_score(y_true, preds, average="macro")
    f1 = f1_score(y_true, preds, average="macro")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print(classification_report(y_true, preds, digits=4))

    df.to_csv("test_predictions.csv", index=False)

    print("Saved predictions → test_predictions.csv")

if __name__ == "__main__":
    main()
