import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from src.features import select_features
from src.preprocessing import build_preprocessor
from src.train import train_models


def save_comparison_results(results_dict):
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================
    # 1️⃣ CSV COMPARATIVO
    # =========================
    metrics_list = []

    for model_name, result in results_dict.items():
        metrics_list.append({
            "Model": model_name,
            "Accuracy": round(result["accuracy"], 4),
            "Precision": round(result["precision"], 4),
            "Recall": round(result["recall"], 4),
            "F1": round(result["f1"], 4),
            "ROC_AUC": round(result["roc_auc"], 4) if result["roc_auc"] is not None else None
        })

    df_metrics = pd.DataFrame(metrics_list)
    csv_path = f"reports/model_comparison_{timestamp}.csv"
    df_metrics.to_csv(csv_path, index=False)

    print(f"\n📁 CSV comparativo salvo em: {csv_path}")

    # =========================
    # 2️⃣ GRÁFICO ROC
    # =========================
    plt.figure()

    for model_name, result in results_dict.items():
        if result["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(result["y_test"], result["y_proba"])
            roc_auc_value = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    roc_path = f"reports/roc_curve_comparison_{timestamp}.png"
    plt.savefig(roc_path)
    plt.close()

    print(f"📊 ROC curve salva em: {roc_path}")


def main():

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("data/1000_Sales_Records.csv")

    X, y, num_feat, cat_feat = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(num_feat, cat_feat)

    models = train_models(X_train, y_train, preprocessor)

    print("\n===== MODEL EVALUATION =====")

    results_dict = {}

    for name, model in models.items():

        print(f"\n{name}")

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = None
            roc_auc = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")

        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Guardar resultados para CSV e ROC
        results_dict[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "y_test": y_test,
            "y_proba": y_proba
        }

    # =========================
    # SALVAR RESULTADOS FINAIS
    # =========================
    save_comparison_results(results_dict)


if __name__ == "__main__":
    main()