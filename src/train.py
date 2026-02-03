import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # IMPORTANT: ensures saving works in CLI scripts on Windows

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def main(test_size: float, random_state: int) -> None:
    # Resolve project root (iris-classifier/) from this file location
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix + save
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    disp.plot()
    plt.tight_layout()

    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(test_size=args.test_size, random_state=args.random_state)
