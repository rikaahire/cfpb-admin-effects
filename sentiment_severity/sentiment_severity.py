import pandas as pd
import numpy as np
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import csv

# ----------------------------
# 1. Load complaints (with narratives)
# ----------------------------
max_chunks = 40
chunks = []

cols_to_keep = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "State",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Consumer complaint narrative"
]

total_rows = 0

for i, chunk in enumerate(pd.read_csv(
    "complaints.csv",
    usecols=lambda c: c in cols_to_keep,
    chunksize=100000,
    on_bad_lines="skip",
    engine="python",
    quoting=csv.QUOTE_NONE,
    encoding_errors="replace"
)):
    if i >= max_chunks:
        break

    chunks.append(chunk)
    total_rows += len(chunk)

    print(f"Loaded chunk {i + 1}, actual rows so far: {total_rows:,}")

if chunks:
    df = pd.concat(chunks, ignore_index=True)
else:
    df = pd.DataFrame()

# ----------------------------
# 2. Rename columns
# ----------------------------
df = df.rename(columns={
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "State": "state",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "Consumer disputed?": "consumer_disputed",
    "Consumer complaint narrative": "narrative"
})

# ----------------------------
# 3. Filter + clean text
# ----------------------------
df = df[df["narrative"].notna()].copy()
df["text"] = df["narrative"].astype(str)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{5,}\b", " <long_number> ", text)
    text = re.sub(r"[^a-z0-9$%.,!?'\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text"].apply(clean_text)

# Drop empty after cleaning
df = df[df["text_clean"].str.len() > 0].copy()

# ----------------------------
# 4. Feature engineering (severity inputs)
# ----------------------------

# Length features
df["char_count"] = df["text_clean"].str.len()
df["word_count"] = df["text_clean"].str.split().str.len()
df["sentence_count"] = df["text_clean"].str.count(r"[.!?]").clip(lower=1)

# Severity keywords
keywords = [
    "fraud", "scam", "harassment", "harassing", "illegal", "lied",
    "misleading", "stolen", "identity theft", "unauthorized",
    "discrimination", "threat", "threatened", "deceptive",
    "abuse", "abusive", "wrongful", "damages", "foreclosure",
    "eviction", "bankrupt", "bankruptcy", "default", "lawsuit",
    "sue", "ruined my credit", "financial hardship"
]

def keyword_count(text):
    count = 0
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        count += len(re.findall(pattern, text))
    return count

df["keyword_hits"] = df["text_clean"].apply(keyword_count)

# Dollar mentions
df["dollar_mentions"] = df["text_clean"].apply(
    lambda x: len(re.findall(r"\$\s?\d[\d,]*(?:\.\d{2})?", x))
)

# Numeric mentions
df["numeric_mentions"] = df["text_clean"].apply(
    lambda x: len(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", x))
)

# ----------------------------
# 5. Time features (administration)
# ----------------------------
df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

def assign_admin(date):
    if pd.isna(date):
        return np.nan
    elif pd.Timestamp("2021-01-20") <= date < pd.Timestamp("2025-01-20"):
        return "Biden"
    elif date >= pd.Timestamp("2025-01-20"):
        return "Trump"
    else:
        return np.nan

df["administration"] = df["date_received"].apply(assign_admin)

# Drop outside your comparison window
df = df[df["administration"].notna()].copy()

df["regime"] = df["administration"].map({
    "Biden": "active",
    "Trump": "dormant"
})

df["year"] = df["date_received"].dt.year
df["month"] = df["date_received"].dt.month

# ----------------------------
# 6. Outcome variables
# ----------------------------
df["timely_response_binary"] = (
    df["timely_response"]
    .astype(str)
    .str.lower()
    .map({"yes": 1, "no": 0})
)

df["consumer_disputed_binary"] = (
    df["consumer_disputed"]
    .astype(str)
    .str.lower()
    .map({"yes": 1, "no": 0})
)

df["monetary_relief"] = (
    df["company_response"]
    .astype(str)
    .str.lower()
    .str.contains("monetary relief", na=False)
    .astype(int)
)

# ----------------------------
# 7. Prepare severity components
# ----------------------------
def zscore(series):
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

df["keyword_component"] = zscore(df["keyword_hits"])
df["loss_component"] = zscore(df["dollar_mentions"] + df["numeric_mentions"])
df["length_component"] = zscore(df["word_count"])

# ----------------------------
# 8. Sentiment analysis: VADER
# ----------------------------
analyzer = SentimentIntensityAnalyzer()

df["sentiment_vader"] = df["text_clean"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)


# ----------------------------
# 9. Sentiment analysis: Transformer
# ----------------------------
device = 0 if torch.cuda.is_available() else -1

sentiment_model = pipeline(
    "sentiment-analysis",
    model="/scratch/network/.../cfpb/sentiment_model",
    tokenizer="/scratch/network/.../cfpb/sentiment_model",
    device=0
)

texts = df["text_clean"].tolist()

results = sentiment_model(
    texts,
    batch_size=128,
    truncation=True,
    max_length=512
)

df["sentiment_transformer"] = [
    r["score"] if r["label"].upper() == "POSITIVE" else -r["score"]
    for r in results
]


# ----------------------------
# 10. Final severity index
# ----------------------------
df["negativity_component"] = zscore(-1 * df["sentiment_vader"])

df["severity_index"] = df[
    [
        "negativity_component",
        "keyword_component",
        "loss_component",
        "length_component"
    ]
].mean(axis=1)


# ----------------------------
# 11. Basic summary results
# ----------------------------
df = df.drop_duplicates().reset_index(drop=True)

print("Shape:", df.shape)

print("\nAdministration counts:")
print(df["administration"].value_counts())

print("\nMean sentiment by administration:")
print(df.groupby("administration")["sentiment_vader"].mean())

print("\nMean severity by administration:")
print(df.groupby("administration")["severity_index"].mean())

print("\nMean monetary relief by administration:")
print(df.groupby("administration")["monetary_relief"].mean())

print("\nMean timely response by administration:")
print(df.groupby("administration")["timely_response_binary"].mean())


# ----------------------------
# 12. Visualization
# ----------------------------
sns.boxplot(x="administration", y="severity_index", data=df)
plt.title("Complaint Severity by Administration")
plt.xlabel("Administration")
plt.ylabel("Severity Index")
plt.show()


# ----------------------------
# 13. Regression: severity predicting monetary relief
# ----------------------------
reg_df = df[[
    "monetary_relief",
    "severity_index",
    "administration",
    "product",
    "issue"
]].dropna().copy()

# Convert administration to binary: Trump = 1, Biden = 0
reg_df["trump_period"] = (reg_df["administration"] == "Trump").astype(int)

# Interaction term
reg_df["severity_x_trump"] = reg_df["severity_index"] * reg_df["trump_period"]

X = reg_df[[
    "severity_index",
    "trump_period",
    "severity_x_trump"
]]

X = sm.add_constant(X)
y = reg_df["monetary_relief"]

if y.nunique() > 1:
    model = sm.Logit(y, X).fit()
    print("\nLogistic regression: monetary relief")
    print(model.summary())
else:
    print("\nSkipping logistic regression: monetary_relief has only one class in sample.")


# ----------------------------
# 14. Product-level analysis
# ----------------------------

# Keep products with enough observations in both administrations
min_n = 500

product_summary = (
    df.groupby(["product", "administration"])
    .agg(
        n=("complaint_id", "count"),
        mean_sentiment=("sentiment_vader", "mean"),
        mean_severity=("severity_index", "mean"),
        monetary_relief_rate=("monetary_relief", "mean"),
        timely_response_rate=("timely_response_binary", "mean")
    )
    .reset_index()
)

# Pivot so Biden and Trump are side-by-side
product_pivot = product_summary.pivot(
    index="product",
    columns="administration",
    values=[
        "n",
        "mean_sentiment",
        "mean_severity",
        "monetary_relief_rate",
        "timely_response_rate"
    ]
)

# Flatten column names
product_pivot.columns = [
    f"{metric}_{admin}" for metric, admin in product_pivot.columns
]

product_pivot = product_pivot.reset_index()

# Keep only products with enough Biden and Trump complaints
product_pivot = product_pivot[
    (product_pivot["n_Biden"] >= min_n) &
    (product_pivot["n_Trump"] >= min_n)
].copy()

# Calculate Trump - Biden differences
product_pivot["sentiment_change_trump_minus_biden"] = (
    product_pivot["mean_sentiment_Trump"] - product_pivot["mean_sentiment_Biden"]
)

product_pivot["severity_change_trump_minus_biden"] = (
    product_pivot["mean_severity_Trump"] - product_pivot["mean_severity_Biden"]
)

product_pivot["monetary_relief_change_trump_minus_biden"] = (
    product_pivot["monetary_relief_rate_Trump"] - product_pivot["monetary_relief_rate_Biden"]
)

product_pivot["timely_response_change_trump_minus_biden"] = (
    product_pivot["timely_response_rate_Trump"] - product_pivot["timely_response_rate_Biden"]
)

print("\nProduct-level differences: Trump minus Biden")
print(product_pivot[[
    "product",
    "n_Biden",
    "n_Trump",
    "mean_sentiment_Biden",
    "mean_sentiment_Trump",
    "sentiment_change_trump_minus_biden",
    "mean_severity_Biden",
    "mean_severity_Trump",
    "severity_change_trump_minus_biden",
    "monetary_relief_change_trump_minus_biden",
    "timely_response_change_trump_minus_biden"
]].sort_values("severity_change_trump_minus_biden", ascending=False).head(15))

# Products with largest increase in severity under Trump
top_severity_increases = product_pivot.sort_values(
    "severity_change_trump_minus_biden",
    ascending=False
).head(10)

# Products with largest decrease in sentiment under Trump
# More negative = lower sentiment, so sort ascending
top_sentiment_declines = product_pivot.sort_values(
    "sentiment_change_trump_minus_biden",
    ascending=True
).head(10)

print("\nTop products with largest severity increase under Trump:")
print(top_severity_increases[[
    "product",
    "severity_change_trump_minus_biden",
    "mean_severity_Biden",
    "mean_severity_Trump",
    "n_Biden",
    "n_Trump"
]])

print("\nTop products with largest sentiment decline under Trump:")
print(top_sentiment_declines[[
    "product",
    "sentiment_change_trump_minus_biden",
    "mean_sentiment_Biden",
    "mean_sentiment_Trump",
    "n_Biden",
    "n_Trump"
]])

# Save product-level table
product_pivot.to_csv("cfpb_product_level_analysis.csv", index=False)

print("\nSaved file: cfpb_product_level_analysis.csv")


# ----------------------------
# 15. Save final dataset
# ----------------------------
df.to_csv("cfpb_sentiment_severity_prepared.csv", index=False)
print("\nSaved file: cfpb_sentiment_severity_prepared.csv")