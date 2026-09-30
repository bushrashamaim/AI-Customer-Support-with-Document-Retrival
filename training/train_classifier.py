"""
ML Document Classifier Training
TF-IDF + SVM with GridSearchCV for hyperparameter tuning.

Usage:
    python training/train_classifier.py --data-dir ./labeled_docs --output models/classifier.pkl
    python training/train_classifier.py --csv dataset.csv --output models/classifier.pkl --evaluate
"""

import argparse
import json
import pickle
import sys
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

# Document type labels
LABELS = [
    "invoice", "contract", "id_card", "passport",
    "receipt", "bank_statement", "resume",
    "medical_report", "legal_document", "form", "letter",
]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_from_directory(data_dir: str) -> tuple[list[str], list[str]]:
    """
    Load text files from a labeled directory structure:
    data_dir/
      invoice/   doc1.txt  doc2.txt ...
      contract/  doc3.txt ...
      ...
    Returns (texts, labels)
    """
    texts, labels = [], []
    data_path = Path(data_dir)

    for label_dir in sorted(data_path.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for txt_file in label_dir.glob("*.txt"):
            texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)

    logger.info(f"Loaded {len(texts)} documents from {data_dir}")
    _log_class_dist(labels)
    return texts, labels


def load_from_csv(csv_path: str, text_col: str = "text", label_col: str = "label"):
    """Load from CSV with text and label columns."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").tolist()
    labels = df[label_col].tolist()
    logger.info(f"Loaded {len(texts)} rows from {csv_path}")
    _log_class_dist(labels)
    return texts, labels


def _log_class_dist(labels: list[str]):
    from collections import Counter
    counts = Counter(labels)
    logger.info("Class distribution:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {label:20}: {count}")


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_feature_pipeline():
    """
    TF-IDF with sublinear scaling + character n-grams
    for robust short-text classification.
    """
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
    )

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    return Pipeline([
        ("features", features),
        ("norm", Normalizer()),
    ])


# ─── Trainer ─────────────────────────────────────────────────────────────────

class ClassifierTrainer:
    def __init__(self, output_path: str = "models/classifier.pkl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_pipeline = None
        self.classifier = None
        self.label_encoder = None

    def train(
        self,
        texts: list[str],
        labels: list[str],
        algorithm: str = "svm",
        tune_hyperparams: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Train classifier with optional hyperparameter tuning.

        Args:
            texts:           Raw OCR text for each document
            labels:          Document type labels
            algorithm:       svm | logistic | random_forest | gradient_boost
            tune_hyperparams: Run GridSearchCV (slower but better accuracy)
        """
        from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import classification_report, confusion_matrix
        import joblib

        logger.info(f"Training {algorithm} classifier on {len(texts)} documents...")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Build features
        self.feature_pipeline = build_feature_pipeline()
        X_train_feat = self.feature_pipeline.fit_transform(X_train)
        X_test_feat = self.feature_pipeline.transform(X_test)

        # Choose base classifier
        if algorithm == "svm":
            base_clf = LinearSVC(max_iter=2000, class_weight="balanced")
            # Wrap for probability support
            clf = CalibratedClassifierCV(base_clf, cv=3)
            param_grid = {"base_estimator__C": [0.1, 1.0, 5.0, 10.0]}
        elif algorithm == "logistic":
            clf = LogisticRegression(
                max_iter=1000, class_weight="balanced",
                multi_class="multinomial", solver="lbfgs",
            )
            param_grid = {"C": [0.1, 1.0, 5.0, 10.0]}
        elif algorithm == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                n_jobs=-1, random_state=random_state,
            )
            param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 20, 40]}
        elif algorithm == "gradient_boost":
            clf = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, random_state=random_state
            )
            param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2]}
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Optionally tune
        if tune_hyperparams:
            logger.info("Running GridSearchCV (this may take a few minutes)...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            search = GridSearchCV(clf, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
            search.fit(X_train_feat, y_train)
            clf = search.best_estimator_
            logger.info(f"Best params: {search.best_params_}")
            logger.info(f"Best CV F1: {search.best_score_:.4f}")
        else:
            clf.fit(X_train_feat, y_train)

        self.classifier = clf

        # Evaluate
        y_pred = clf.predict(X_test_feat)
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        logger.info(f"\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._save_confusion_matrix(cm, class_names)

        # Save model
        self.save()
        return report

    def save(self):
        import joblib
        bundle = {
            "model": self.classifier,
            "vectorizer": self.feature_pipeline,
            "label_encoder": self.label_encoder,
            "version": "1.0",
        }
        joblib.dump(bundle, self.output_path)
        size_kb = self.output_path.stat().st_size / 1024
        logger.success(f"Model saved: {self.output_path} ({size_kb:.1f} KB)")

    def _save_confusion_matrix(self, cm: np.ndarray, labels: list):
        """Save confusion matrix as JSON for visualization."""
        cm_path = self.output_path.parent / "confusion_matrix.json"
        data = {
            "labels": list(labels),
            "matrix": cm.tolist(),
        }
        cm_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Confusion matrix saved: {cm_path}")

    @staticmethod
    def evaluate_saved(model_path: str, texts: list[str], labels: list[str]):
        """Evaluate a saved model on new data."""
        import joblib
        from sklearn.metrics import classification_report
        bundle = joblib.load(model_path)
        model = bundle["model"]
        vectorizer = bundle["vectorizer"]
        le = bundle["label_encoder"]

        X = vectorizer.transform(texts)
        y_true = le.transform(labels)
        y_pred = model.predict(X)
        report = classification_report(y_true, y_pred, target_names=le.classes_)
        print(report)
        return report


# ─── Synthetic data generator for testing ────────────────────────────────────

def generate_synthetic_dataset(n_per_class: int = 50) -> tuple[list[str], list[str]]:
    """
    Generate synthetic training data using keyword templates.
    Useful for bootstrapping when real labeled data isn't available.
    """
    import random

    templates = {
        "invoice": [
            "Invoice #{n}\nBill To: {name}\nSubtotal: ${amount}\nTax (10%): ${tax}\nAmount Due: ${total}\nPayment Due: {date}",
            "INVOICE\nInvoice No: INV-{n}\nDate: {date}\nQty  Description  Unit Price  Total\n1  Service  ${amount}  ${amount}\nVAT: ${tax}\nTotal: ${total}",
        ],
        "contract": [
            "SERVICE AGREEMENT\nThis Agreement is entered into as of {date} between {name} (hereinafter 'Client') and ABC Corp.\nThe parties agree to the following terms and conditions.\nLiability shall be limited to the contract value.\nGoverning law: State of California.\nSigned by both parties.",
            "CONSULTING CONTRACT\nWhereas the Client desires to engage Consultant, and Consultant agrees to provide services.\nObligations: Consultant shall deliver work by {date}.\nIndemnification clause applies.\nJurisdiction: New York.",
        ],
        "resume": [
            "CURRICULUM VITAE\n{name}\nEmail: {name}@email.com\nWork Experience\n2020-Present: Software Engineer at TechCorp\n2018-2020: Junior Developer\nEducation\nB.Sc Computer Science, 2018\nSkills: Python, Java, SQL\nReferences available upon request.",
        ],
        "receipt": [
            "RECEIPT\nThank you for your purchase!\nDate: {date}\nItem: Product A  Qty: 2  ${amount}\nItem: Product B  Qty: 1  $15.00\nSubtotal: ${amount}\nCash: ${total}\nChange: $0.00\nTransaction ID: {n}",
        ],
        "bank_statement": [
            "BANK STATEMENT\nAccount Number: 1234-5678-{n}\nStatement Period: {date}\nOpening Balance: ${amount}\nDebit  Credit  Date  Description\n${amount}  -  {date}  Purchase\n-  ${amount}  {date}  Deposit\nClosing Balance: ${total}\nIBAN: GB{n}BARC",
        ],
        "letter": [
            "Dear {name},\n\nI am writing to inform you regarding the matter discussed on {date}.\nPlease find enclosed the relevant documentation.\nKindly review and respond at your earliest convenience.\n\nYours sincerely,\nJohn Smith",
        ],
        "form": [
            "APPLICATION FORM\nPlease fill in all sections.\nFirst Name: ___________\nLast Name: ___________\nDate of Birth: ___________\nAddress: ___________\nPhone Number: ___________\nCheck all that apply: [ ] Option A  [ ] Option B\nSignature: ___________  Date: ___________",
        ],
        "medical_report": [
            "MEDICAL REPORT\nPatient: {name}  DOB: {date}\nPhysician: Dr. Smith\nDiagnosis: Routine checkup\nBlood Pressure: 120/80\nSymptoms: None reported\nTreatment: Continue current medication\nDosage: 10mg daily\nLab Results: Within normal range\nNext Appointment: {date}",
        ],
    }

    import random
    names = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Williams"]
    texts, labels = [], []

    for label, tmpl_list in templates.items():
        for _ in range(n_per_class):
            tmpl = random.choice(tmpl_list)
            text = tmpl.format(
                n=random.randint(1000, 9999),
                name=random.choice(names),
                amount=f"{random.uniform(50, 5000):.2f}",
                tax=f"{random.uniform(5, 500):.2f}",
                total=f"{random.uniform(100, 6000):.2f}",
                date=f"0{random.randint(1,9)}/{random.randint(10,28)}/202{random.randint(0,4)}",
            )
            texts.append(text)
            labels.append(label)

    logger.info(f"Generated {len(texts)} synthetic samples across {len(templates)} classes")
    return texts, labels


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train document type classifier")
    p.add_argument("--data-dir", default=None, help="Labeled directory with subfolders per class")
    p.add_argument("--csv", default=None, help="CSV file with text and label columns")
    p.add_argument("--output", default="models/classifier.pkl")
    p.add_argument("--algorithm", default="svm", choices=["svm", "logistic", "random_forest", "gradient_boost"])
    p.add_argument("--tune", action="store_true", help="Run GridSearchCV hyperparameter tuning")
    p.add_argument("--evaluate", action="store_true", help="Evaluate saved model only")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (for testing)")
    p.add_argument("--n-per-class", type=int, default=100, help="Synthetic samples per class")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = ClassifierTrainer(output_path=args.output)

    if args.synthetic or (not args.data_dir and not args.csv):
        logger.info("Using synthetic training data...")
        texts, labels = generate_synthetic_dataset(n_per_class=args.n_per_class)
    elif args.data_dir:
        texts, labels = load_from_directory(args.data_dir)
    elif args.csv:
        texts, labels = load_from_csv(args.csv)
    else:
        logger.error("Provide --data-dir, --csv, or --synthetic")
        sys.exit(1)

    if args.evaluate and Path(args.output).exists():
        ClassifierTrainer.evaluate_saved(args.output, texts, labels)
    else:
        trainer.train(texts, labels, algorithm=args.algorithm, tune_hyperparams=args.tune)
