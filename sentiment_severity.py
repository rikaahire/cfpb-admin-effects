import pandas as pd
import numpy as np
import re

# ----------------------------
# 1. Load complaints (with narratives)
# ----------------------------
max_chunks = 5
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

for i, chunk in enumerate(pd.read_csv(
    "complaints.csv",
    chunksize=100,
    engine="python",
    on_bad_lines="skip"
)):
    if i >= max_chunks:
        break

    # Keep only needed columns
    available_cols = [col for col in cols_to_keep if col in chunk.columns]
    chunk = chunk[available_cols]

    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

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

# placeholders for next step
df["sentiment_vader"] = np.nan
df["sentiment_transformer"] = np.nan
df["anger_score"] = np.nan

# ----------------------------
# 8. Final dataset
# ----------------------------
df = df.drop_duplicates().reset_index(drop=True)

print("Shape:", df.shape)
print("\nAdministration counts:")
print(df["administration"].value_counts())

print("\nPreview:")
print(df.head())

# optional save
# df.to_csv("cfpb_preprocessed.csv", index=False)