import os
import re
import pandas as pd
import numpy as np

# Force matplotlib to use writable local config and a non-GUI backend.
MPL_CONFIG_DIR = os.path.join(os.getcwd(), ".matplotlib")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_CONFIG_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Keep all generated figures in one folder so the project is easy to submit.
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# =========================================
# Section 1: Data Loading
# =========================================

# 1. load data
df = pd.read_csv("Resume/Resume.csv")

print("=" * 70)
print("Section 1: Data Loading")
print("=" * 70)
print(f"Dataset shape: {df.shape}")
print("\nFirst 3 rows:")
print(df.head(3))


# =========================================
# Section 2: Preprocessing
# =========================================

# 2. function to clean text
def clean_resume_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text


def print_section_title(title):
    """Print a clean notebook-style section title."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def shorten_text(text, max_length=180):
    """Shorten long text and keep it safe to print in this console."""
    text = str(text).replace("\n", " ").strip()
    text = text.encode("cp1255", errors="replace").decode("cp1255")
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_dataframe_cleanly(df_to_print, max_colwidth=120):
    """Print tables with cleaner formatting for the report."""
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", max_colwidth
    ):
        print(df_to_print.to_string(index=False))


# 3. clean the text
df["Clean_Resume"] = df["Resume_str"].apply(clean_resume_text)

print("\n" + "=" * 70)
print("Section 2: Preprocessing")
print("=" * 70)
preprocessing_preview = pd.DataFrame({
    "Original Text": df["Resume_str"].head(3).apply(lambda text: shorten_text(text, 160)),
    "Cleaned Text": df["Clean_Resume"].head(3).apply(lambda text: shorten_text(text, 160))
})
print_dataframe_cleanly(preprocessing_preview, max_colwidth=90)


# 4. split into input and output
X = df["Clean_Resume"]
y = df["Category"]

# 5. split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print_section_title("Train / Test Split")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\n--- Train Set (X_train) ---")
print(X_train.head())

print("\n--- Test Set (X_test) ---")
print(X_test.head())


# =========================================
# Section 3: Feature Engineering
# =========================================

# 7. transform words into numbers
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n" + "=" * 70)
print("Section 3: Feature Engineering")
print("=" * 70)
print("\n--- First 20 learned features (words) ---")
print(vectorizer.get_feature_names_out()[:20])

# 8. feature names
feature_names = vectorizer.get_feature_names_out()

# 9. show 2 examples from train
print("\n--- 2 examples from Train after TF-IDF ---")
train_features_df = pd.DataFrame(X_train_tfidf[:2].toarray(), columns=feature_names)
print(train_features_df.loc[:, (train_features_df != 0).any(axis=0)].head())

# 10. show 2 examples from test
print("\n--- 2 examples from Test after TF-IDF ---")
test_features_df = pd.DataFrame(X_test_tfidf[:2].toarray(), columns=feature_names)
print(test_features_df.loc[:, (test_features_df != 0).any(axis=0)].head())


# ---------------------------------
# 9. Naive Bayes From Scratch
# ---------------------------------
class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha   # hyperparameter for Laplace smoothing

    def train(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}
        self.word_probs = {}

        n_samples = X.shape[0]  # count the resumes in the data

        for c in self.classes:
            X_c = X[y == c]

            # prior probability P(class)
            self.class_probs[c] = X_c.shape[0] / n_samples

            # word counts in this class + Laplace smoothing
            word_count = np.sum(X_c, axis=0) + self.alpha

            # total word count in this class
            total_words = np.sum(word_count)

            # conditional probability P(word | class)
            self.word_probs[c] = word_count / total_words

    def predict_one(self, x):  # input resume and output category
        class_scores = {}

        for c in self.classes:
            # start with log P(class)
            log_prob = np.log(self.class_probs[c])

            # add sum of log P(word | class)
            log_prob += np.sum(x * np.log(self.word_probs[c]))

            class_scores[c] = log_prob

        return max(class_scores, key=class_scores.get)  # choose the max category

    def predict(self, X):  # move on the resumes and return all predictions
        predictions = []
        for x in X:
            predictions.append(self.predict_one(x))
        return predictions


# ---------------------------------
# 10. Convert sparse matrix to arrays
# ---------------------------------
X_train_array = X_train_tfidf.toarray()
X_test_array = X_test_tfidf.toarray()
y_train_array = y_train.values
y_test_array = y_test.values


# ---------------------------------
# 11. Train the model
# ---------------------------------
nb_model = NaiveBayes(0.1)
nb_model.train(X_train_array, y_train_array)


# ---------------------------------
# 12. Predict
# ---------------------------------
y_pred = nb_model.predict(X_test_array)


# ---------------------------------
# 13. Evaluate
# ---------------------------------
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test_array, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_array, y_pred, zero_division=0))


def get_top_tfidf_features(vectorizer_obj, vector_row, top_n=8):
    """Return the highest-weight non-zero TF-IDF features for one sample."""
    feature_names_local = vectorizer_obj.get_feature_names_out()
    row_array = vector_row.toarray().flatten()
    non_zero_indices = np.where(row_array > 0)[0]

    if len(non_zero_indices) == 0:
        return []

    top_indices = non_zero_indices[np.argsort(row_array[non_zero_indices])[::-1][:top_n]]
    return [(feature_names_local[i], float(row_array[i])) for i in top_indices]


def format_tfidf_representation(vectorizer_obj, vector_row, top_n=8):
    """Create a readable TF-IDF description for report tables."""
    top_features = get_top_tfidf_features(vectorizer_obj, vector_row, top_n=top_n)
    if not top_features:
        return "No non-zero TF-IDF features"
    return ", ".join([f"{feature} ({score:.3f})" for feature, score in top_features])


def create_feature_examples_df(original_texts, cleaned_texts, pipeline, num_examples=3):
    """
    Create side-by-side feature engineering examples for presentation:
    Original Text | Cleaned Text | TF-IDF Representation
    """
    vectorizer_obj = pipeline.named_steps["tfidf"]
    transformed = vectorizer_obj.transform(cleaned_texts.iloc[:num_examples])
    rows = []

    for i in range(min(num_examples, len(cleaned_texts))):
        rows.append({
            "Original Text": shorten_text(original_texts.iloc[i], 180),
            "Cleaned Text": shorten_text(cleaned_texts.iloc[i], 180),
            "TF-IDF Representation": format_tfidf_representation(vectorizer_obj, transformed[i], top_n=8)
        })

    return pd.DataFrame(rows)


def show_feature_engineering_examples(original_texts, cleaned_texts, pipeline, title, num_examples=3):
    """Print feature-engineering examples as a side-by-side table."""
    print_section_title(title)
    examples_df = create_feature_examples_df(
        original_texts,
        cleaned_texts,
        pipeline,
        num_examples=num_examples
    )
    print_dataframe_cleanly(examples_df, max_colwidth=95)
    return examples_df


def plot_model_accuracy_bar(results_df, title, filename):
    """Plot a professional bar chart comparing model accuracies."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.figure(figsize=(12, 6))
    sorted_df = results_df.sort_values("Validation Accuracy", ascending=False)
    ax = sns.barplot(
        data=sorted_df,
        x="Validation Accuracy",
        y="Model / Features",
        hue="Model / Features",
        dodge=False,
        palette="Blues_r",
        legend=False
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Validation Accuracy")
    plt.ylabel("Model / Feature Combination")
    plt.xlim(0, 1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot to {output_path}")


def plot_confusion_matrix_heatmap(cm_df, title, filename):
    """Plot confusion matrix as a clean heatmap."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        cm_df,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Number of Samples"}
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")


def plot_top_model_features(pipeline, title, filename, top_n=15):
    """Show one simple explainability view using top important words."""
    vectorizer_obj = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names_local = vectorizer_obj.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        importance = np.mean(np.abs(classifier.coef_), axis=0)
    elif hasattr(classifier, "feature_log_prob_"):
        importance = np.max(classifier.feature_log_prob_, axis=0) - np.min(classifier.feature_log_prob_, axis=0)
    else:
        print("\nTop feature visualization is not available for this classifier.")
        return

    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_features_df = pd.DataFrame({
        "Feature": feature_names_local[top_indices],
        "Importance": importance[top_indices]
    })

    print_section_title("Bonus: Top Influential Features")
    print_dataframe_cleanly(top_features_df)

    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        dodge=False,
        palette="viridis",
        legend=False
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved top features plot to {output_path}")


def evaluate_candidate_models(X_train_part, y_train_part, X_val_part, y_val_part, candidates):
    """Train and compare several model + TF-IDF combinations."""
    results = []

    for config in candidates:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=config["max_features"],
                ngram_range=config["ngram_range"],
                stop_words="english"
            )),
            ("classifier", clone(config["model"]))
        ])

        pipeline.fit(X_train_part, y_train_part)
        val_predictions = pipeline.predict(X_val_part)
        val_accuracy = accuracy_score(y_val_part, val_predictions)

        results.append({
            "name": config["name"],
            "pipeline": pipeline,
            "validation_accuracy": val_accuracy,
            "max_features": config["max_features"],
            "ngram_range": config["ngram_range"]
        })

    return sorted(results, key=lambda item: item["validation_accuracy"], reverse=True)


def format_grid_search_results(cv_results_df):
    """Create a clean table with the required grid search summary columns."""
    formatted_rows = []

    for _, row in cv_results_df.iterrows():
        params = row["params"].copy()
        classifier = params.pop("classifier")
        formatted_rows.append({
            "model": classifier.__class__.__name__,
            "parameters": str(params),
            "mean validation score": round(row["mean_test_score"], 4)
        })

    results_df = pd.DataFrame(formatted_rows)
    return results_df.sort_values("mean validation score", ascending=False).reset_index(drop=True)

# Recreate a dataframe split so we can present original and cleaned text together.
# Using the same random state keeps it aligned with the earlier split.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.copy()
test_df = test_df.copy()


# =========================================
# Section 4: Model Training
# =========================================

print("\n" + "=" * 70)
print("Section 4: Model Training")
print("=" * 70)

# Create a validation split from the training set for model comparison.
train_sub_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["Category"]
)

print_section_title("Training / Validation Split")
print(f"Training samples used for model selection: {len(train_sub_df)}")
print(f"Validation samples: {len(val_df)}")

# Candidate models for the standard model-comparison flow.
candidate_models = [
    {
        "name": "Multinomial Naive Bayes | TF-IDF(1,1), max_features=3000, alpha=1.0",
        "max_features": 3000,
        "ngram_range": (1, 1),
        "model": MultinomialNB(alpha=1.0)
    },
    {
        "name": "Multinomial Naive Bayes | TF-IDF(1,2), max_features=3000, alpha=1.0",
        "max_features": 3000,
        "ngram_range": (1, 2),
        "model": MultinomialNB(alpha=1.0)
    },
    {
        "name": "Linear SVM | TF-IDF(1,1), max_features=3000, C=1",
        "max_features": 3000,
        "ngram_range": (1, 1),
        "model": LinearSVC(C=1.0, random_state=42)
    },
    {
        "name": "Linear SVM | TF-IDF(1,2), max_features=3000, C=1",
        "max_features": 3000,
        "ngram_range": (1, 2),
        "model": LinearSVC(C=1.0, random_state=42)
    }
]

comparison_results = evaluate_candidate_models(
    train_sub_df["Clean_Resume"],
    train_sub_df["Category"],
    val_df["Clean_Resume"],
    val_df["Category"],
    candidate_models
)

print_section_title("Best Model Selection")
comparison_df = pd.DataFrame([
    {
        "Model / Features": result["name"],
        "Validation Accuracy": round(result["validation_accuracy"], 4)
    }
    for result in comparison_results
])
print_dataframe_cleanly(comparison_df)

plot_model_accuracy_bar(
    comparison_df,
    title="Validation Accuracy Comparison Across Models",
    filename="model_accuracy_comparison.png"
)

best_result = comparison_results[0]
print("\nStep 1: Select the best model")
print(f"Selected model: {best_result['name']}")
print(f"Validation accuracy: {best_result['validation_accuracy']:.4f}")
print("Why this model was selected: it achieved the highest validation accuracy in the comparison table.")

print_section_title("Retraining on Full Train Set")
print("Step 2: Build the selected model again using the same TF-IDF settings and classifier.")
best_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=best_result["max_features"],
        ngram_range=best_result["ngram_range"],
        stop_words="english"
    )),
    ("classifier", clone(best_result["pipeline"].named_steps["classifier"]))
])
print("Step 3: Retrain the selected model on the full training set.")
best_pipeline.fit(train_df["Clean_Resume"], train_df["Category"])
print("Retraining complete. The best model is ready for test evaluation.")

show_feature_engineering_examples(
    train_df["Resume_str"],
    train_df["Clean_Resume"],
    best_pipeline,
    "Feature Engineering Examples (Train) - Side-by-Side Table",
    num_examples=3
)

plot_top_model_features(
    best_pipeline,
    title="Top TF-IDF Features of the Best Validation Model",
    filename="top_features_validation_best.png",
    top_n=15
)


# =========================================
# Section 5: Evaluation
# =========================================

print("\n" + "=" * 70)
print("Section 5: Evaluation")
print("=" * 70)

show_feature_engineering_examples(
    test_df["Resume_str"],
    test_df["Clean_Resume"],
    best_pipeline,
    "Feature Engineering Examples (Test) - Side-by-Side Table",
    num_examples=3
)

# Apply the best grid-search model to the test set.
y_test_pred = best_pipeline.predict(test_df["Clean_Resume"])
test_accuracy = accuracy_score(test_df["Category"], y_test_pred)

print_section_title("First 5 Test Predictions - Table")
first_five_predictions = pd.DataFrame({
    "Index": test_df.index[:5],
    "Short Text Snippet": [shorten_text(text, 120) for text in test_df["Resume_str"].iloc[:5]],
    "True Label": test_df["Category"].iloc[:5].values,
    "Predicted Label": y_test_pred[:5]
})
print_dataframe_cleanly(first_five_predictions)

print_section_title("Final Evaluation")
print(f"Best Selected Model: {best_pipeline.named_steps['classifier'].__class__.__name__}")
print(f"Best Model Description: {best_result['name']}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(test_df["Category"], y_test_pred, zero_division=0))

labels = sorted(test_df["Category"].unique())
cm = confusion_matrix(test_df["Category"], y_test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("\nConfusion Matrix:")
print(cm_df)

plot_confusion_matrix_heatmap(
    cm_df,
    title="Confusion Matrix Heatmap on Test Set",
    filename="confusion_matrix_heatmap.png"
)
