import os
import glob
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# 1. โหลดข้อมูล
raw_dir = "Data/raw"
cleaned_path = os.path.join(raw_dir, "scopus_cleaned.csv")

if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path, low_memory=False)
elif os.path.exists(raw_dir):
    csv_files = [f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if "scopus_cleaned" not in f]
    df_list = []
    for f in csv_files:
        temp_df = pd.read_csv(f, low_memory=False)
        if "Subject" not in temp_df.columns:
            subject_name = os.path.splitext(os.path.basename(f))[0]
            subject_name = re.sub(r'^\d+_', '', subject_name).replace('_', ' ')
            temp_df["Subject"] = subject_name
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
else:
    raise FileNotFoundError("ไม่พบโฟลเดอร์ Data/raw หรือไฟล์ CSV")

# ทำความสะอาดข้อมูล
df["Subject"] = df["Subject"].fillna("Unknown")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(2020).astype(int)
df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)
df["Abstract"] = df["Abstract"].fillna("")
df["Author Keywords"] = df["Author Keywords"].fillna("")

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())

def extract_ngrams(tokens, n=2):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

stopwords = {"the", "and", "for", "with", "that", "this", "from", "using", "which", "are", "were", "been", "have", "has", "based"}
top_subjects = df["Subject"].value_counts().head(10).index.tolist()

# ----------------- มิติที่ 1: NLP -----------------
subject_keywords = {}
for sub in top_subjects:
    kws = df[df["Subject"] == sub]["Author Keywords"].dropna().str.cat(sep=";").split(";")
    subject_keywords[sub] = {k.strip().lower() for k in kws if len(k.strip()) > 2}

jaccard_matrix = np.zeros((len(top_subjects), len(top_subjects)))
for i, s1 in enumerate(top_subjects):
    for j, s2 in enumerate(top_subjects):
        u1, u2 = subject_keywords[s1], subject_keywords[s2]
        union_len = len(u1.union(u2))
        jaccard_matrix[i][j] = len(u1.intersection(u2)) / union_len if union_len > 0 else 0

ttr_results = []
for sub in top_subjects:
    tokens = [t for text in df[df["Subject"] == sub]["Abstract"] for t in tokenize(text) if t not in stopwords]
    ttr = (len(set(tokens)) / len(tokens)) if tokens else 0
    ttr_results.append({"Subject": sub, "TTR": round(ttr, 4)})
df_ttr = pd.DataFrame(ttr_results).sort_values(by="TTR", ascending=True)

top_ngrams_by_sub = {}
for sub in top_subjects[:4]:
    all_tokens = [t for text in df[df["Subject"] == sub]["Abstract"] for t in tokenize(text) if t not in stopwords]
    bigrams = extract_ngrams(all_tokens, 2)
    top_ngrams_by_sub[sub] = Counter(bigrams).most_common(7)

# ----------------- มิติที่ 2: Temporal -----------------
df_temporal = df[df["Subject"].isin(top_subjects)].groupby(["Year", "Subject"]).size().unstack(fill_value=0)

min_year, max_year = df["Year"].min(), df["Year"].max()
mid_year = (min_year + max_year) // 2
early_tokens = [t for text in df[df["Year"] <= mid_year]["Abstract"] for t in tokenize(text) if t not in stopwords]
late_tokens = [t for text in df[df["Year"] > mid_year]["Abstract"] for t in tokenize(text) if t not in stopwords]
early_counts, late_counts = Counter(early_tokens), Counter(late_tokens)
all_kws = list(set(list(early_counts.keys()) + list(late_counts.keys())))
kw_growth = []
for k in all_kws:
    c1, c2 = early_counts.get(k, 0), late_counts.get(k, 0)
    if c1 + c2 >= 50:
        growth_rate = ((c2 - c1) / (c1 + 1)) * 100
        kw_growth.append({"Keyword": k, "Growth": growth_rate})
df_growth = pd.DataFrame(kw_growth).sort_values(by="Growth", ascending=False)
df_slope = pd.concat([df_growth.head(5), df_growth.tail(5)]) if not df_growth.empty else pd.DataFrame()

sample_citations = df[df["Cited by"] > 0].sample(n=min(len(df), 1500), random_state=42)
sample_citations["LogCitations"] = np.log1p(sample_citations["Cited by"])

# ----------------- มิติที่ 3: Publication -----------------
doc_type_counts = df[df["Subject"].isin(top_subjects)].groupby(["Subject", "Document Type"]).size().unstack(fill_value=0)
doc_type_pct = doc_type_counts.div(doc_type_counts.sum(axis=1), axis=0) * 100

top_venues = df["Source title"].value_counts().head(8).reset_index()
top_venues.columns = ["Source", "Count"]

df["OA_Status"] = df["Open Access"].apply(lambda x: "Open Access" if pd.notna(x) and "Open Access" in str(x) else "Subscription")

# ----------------- มิติที่ 4: Collaboration -----------------
def count_authors(author_str):
    if pd.isna(author_str) or not str(author_str).strip(): return 0
    return len(str(author_str).split(";"))
df["Author_Count"] = df["Authors"].apply(count_authors)

def extract_country(affil):
    if pd.isna(affil): return "Unknown"
    tokens = str(affil).split(",")
    return tokens[-1].strip() if tokens else "Unknown"

affil_col = "Affiliations" if "Affiliations" in df.columns else ("Authors with affiliations" if "Authors with affiliations" in df.columns else None)
if affil_col:
    country_counts = df[affil_col].apply(extract_country).value_counts()
    country_counts = country_counts[~country_counts.index.isin(["Unknown", ""])].head(10).reset_index()
    country_counts.columns = ["Country", "Count"]
else:
    country_counts = pd.DataFrame(columns=["Country", "Count"])

# ----------------- มิติที่ 5: ML Readiness -----------------
ml_sample = df[df["Subject"].isin(top_subjects)].sample(n=min(1000, len(df)), random_state=42)
tfidf = TfidfVectorizer(max_features=300, stop_words="english")
X_tfidf = tfidf.fit_transform(ml_sample["Abstract"].replace("", "empty"))
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(X_tfidf.toarray())
ml_sample["tsne_x"] = tsne_results[:, 0]
ml_sample["tsne_y"] = tsne_results[:, 1]

df["Token_Length"] = df["Abstract"].apply(lambda x: len(tokenize(x)))
sorted_lengths = np.sort(df["Token_Length"])
cdf_y = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

missing_cols = ["Author Keywords", "Funding Details", "Affiliations", "Abstract", "Source title"]
missing_df = pd.DataFrame()
for col in missing_cols:
    if col in df.columns:
        missing_df[col] = df.groupby("Subject")[col].apply(lambda x: (x.isna() | (x == "")).mean() * 100)
missing_df = missing_df.loc[top_subjects]

# รวม Chart Specs
charts = {
    "chart1": {"data": [{"z": jaccard_matrix.tolist(), "x": top_subjects, "y": top_subjects, "type": "heatmap", "colorscale": "Viridis"}], "layout": {"title": "1. Cross-Subject Keyword Overlap (Jaccard)", "margin": {"b": 100, "l": 120}}},
    "chart2": {"data": [{"x": df_ttr["TTR"].tolist(), "y": df_ttr["Subject"].tolist(), "type": "bar", "orientation": "h", "marker": {"color": "#3498db"}}], "layout": {"title": "2. Lexical Diversity (Type-Token Ratio)", "xaxis": {"title": "TTR Score"}}},
    "chart3": {"data": [{"x": [item[1] for item in top_ngrams_by_sub[sub]], "y": [item[0] for item in top_ngrams_by_sub[sub]], "name": sub[:18], "type": "bar", "orientation": "h"} for sub in list(top_ngrams_by_sub.keys())[:2]], "layout": {"title": "3. Top Bigrams by Subject Domain", "barmode": "group"}},
    "chart4": {"data": [{"x": df_temporal.index.tolist(), "y": df_temporal[col].tolist(), "name": col, "type": "scatter", "mode": "lines+markers"} for col in df_temporal.columns[:5]], "layout": {"title": "4. Publication Growth by Subject Over Time"}},
    "chart5": {"data": [{"x": df_slope["Growth"].tolist(), "y": df_slope["Keyword"].tolist(), "type": "bar", "orientation": "h", "marker": {"color": ["#27ae60" if g > 0 else "#e74c3c" for g in df_slope["Growth"]]}}] if not df_slope.empty else [], "layout": {"title": "5. Emerging vs Declining Keywords (% Growth Rate)"}},
    "chart6": {"data": [{"x": sample_citations["Year"].tolist(), "y": sample_citations["LogCitations"].tolist(), "mode": "markers", "type": "scatter", "marker": {"size": 6, "opacity": 0.45, "color": "#e67e22"}}], "layout": {"title": "6. Citation Velocity (Year vs Log-Citations)", "yaxis": {"title": "Log(1 + Citations)"}}},
    "chart7": {"data": [{"x": doc_type_pct.index.tolist(), "y": doc_type_pct[col].tolist(), "name": col, "type": "bar"} for col in doc_type_pct.columns], "layout": {"title": "7. Document Type Breakdown (100% Stacked)", "barmode": "stack", "yaxis": {"title": "Percentage (%)"}}},
    "chart8": {"data": [{"labels": top_venues["Source"].tolist(), "values": top_venues["Count"].tolist(), "type": "pie", "hole": 0.45}], "layout": {"title": "8. Top Journals/Conferences Concentration"}},
    "chart9": {"data": [{"type": "violin", "y": df[df["OA_Status"] == st]["Cited by"].tolist(), "name": st, "box": {"visible": True}, "meanline": {"visible": True}} for st in ["Open Access", "Subscription"]], "layout": {"title": "9. Open Access Citation Advantage", "yaxis": {"type": "log", "title": "Citations (Log Scale)"}}},
    "chart10": {"data": [{"y": df[df["Subject"] == sub]["Author_Count"].tolist(), "name": sub[:12], "type": "box"} for sub in top_subjects[:5]], "layout": {"title": "10. Author Count Distribution per Subject"}},
    "chart11": {"data": [{"x": country_counts["Count"].tolist(), "y": country_counts["Country"].tolist(), "type": "bar", "orientation": "h", "marker": {"color": "#1abc9c"}}], "layout": {"title": "11. Top 10 Affiliation Countries"}},
    "chart12": {"data": [{"x": ml_sample[ml_sample["Subject"] == sub]["tsne_x"].tolist(), "y": ml_sample[ml_sample["Subject"] == sub]["tsne_y"].tolist(), "name": sub, "mode": "markers", "type": "scatter", "marker": {"size": 6}} for sub in top_subjects[:6]], "layout": {"title": "12. 2D Semantic Separability (t-SNE TF-IDF)"}},
    "chart13": {"data": [{"x": sorted_lengths.tolist(), "y": cdf_y.tolist(), "type": "scatter", "mode": "lines", "line": {"color": "#8e44ad", "width": 3}}], "layout": {"title": "13. Token Length CDF (Truncation Impact)", "xaxis": {"range": [0, 500], "title": "Word Count"}, "yaxis": {"title": "Cumulative Probability"}}},
    "chart14": {"data": [{"z": missing_df.values.tolist(), "x": missing_df.columns.tolist(), "y": missing_df.index.tolist(), "type": "heatmap", "colorscale": "Reds"}], "layout": {"title": "14. Data Quality / Missingness Rate (%) Matrix", "margin": {"l": 150}}}
}

meta = {
    "total_records": f"{len(df):,}",
    "total_subjects": str(df["Subject"].nunique()),
    "year_range": f"{df['Year'].min()} - {df['Year'].max()}"
}

# 2. ทำการ Render HTML แล้วบันทึกเป็น index.html ในโฟลเดอร์ Root
html_template = """<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scopus Data EDA & Machine Learning Readiness Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { background-color: #f1f5f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1e293b; }
        .dashboard-header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
        .section-header { border-left: 6px solid #3b82f6; padding-left: 12px; margin-top: 35px; margin-bottom: 20px; font-weight: 700; }
        .chart-card { background: #ffffff; border-radius: 10px; padding: 15px; margin-bottom: 24px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        .summary-badge { background: rgba(255, 255, 255, 0.15); padding: 8px 16px; border-radius: 8px; font-size: 0.95rem; }
    </style>
</head>
<body>
    <div class="container-fluid px-4 py-4">
        <div class="dashboard-header d-flex justify-content-between align-items-center">
            <div>
                <h1 class="h3 fw-bold mb-1">Scopus Bibliometric EDA & ML Readiness</h1>
                <p class="mb-0 text-light opacity-75">การวิเคราะห์เชิงลึก 5 มิติ เพื่อเตรียมความพร้อมสำหรับงานวิจัยและโมเดล Machine Learning</p>
            </div>
            <div class="d-flex gap-3">
                <div class="summary-badge">Total Records: <strong>__TOTAL_RECORDS__</strong></div>
                <div class="summary-badge">Subjects: <strong>__TOTAL_SUBJECTS__</strong></div>
                <div class="summary-badge">Years: <strong>__YEAR_RANGE__</strong></div>
            </div>
        </div>

        <h4 class="section-header text-primary">มิติที่ 1: วิเคราะห์ความซับซ้อนของภาษาและข้อความ (NLP & Text Richness)</h4>
        <div class="row">
            <div class="col-lg-6"><div class="chart-card"><div id="chart1"></div></div></div>
            <div class="col-lg-6"><div class="chart-card"><div id="chart2"></div></div></div>
            <div class="col-12"><div class="chart-card"><div id="chart3"></div></div></div>
        </div>

        <h4 class="section-header text-success">มิติที่ 2: วิเคราะห์แนวโน้มเชิงเวลา (Temporal & Evolution Analysis)</h4>
        <div class="row">
            <div class="col-lg-4"><div class="chart-card"><div id="chart4"></div></div></div>
            <div class="col-lg-4"><div class="chart-card"><div id="chart5"></div></div></div>
            <div class="col-lg-4"><div class="chart-card"><div id="chart6"></div></div></div>
        </div>

        <h4 class="section-header text-warning">มิติที่ 3: วิเคราะห์โครงสร้างการตีพิมพ์และแหล่งเผยแพร่ (Publication & Venue Profiles)</h4>
        <div class="row">
            <div class="col-lg-4"><div class="chart-card"><div id="chart7"></div></div></div>
            <div class="col-lg-4"><div class="chart-card"><div id="chart8"></div></div></div>
            <div class="col-lg-4"><div class="chart-card"><div id="chart9"></div></div></div>
        </div>

        <h4 class="section-header text-info">มิติที่ 4: วิเคราะห์เครือข่ายและความร่วมมือ (Collaboration & Network Metrics)</h4>
        <div class="row">
            <div class="col-lg-6"><div class="chart-card"><div id="chart10"></div></div></div>
            <div class="col-lg-6"><div class="chart-card"><div id="chart11"></div></div></div>
        </div>

        <h4 class="section-header text-danger">มิติที่ 5: วิเคราะห์ความพร้อมเชิง Machine Learning (ML Readiness & Separability)</h4>
        <div class="row">
            <div class="col-lg-6"><div class="chart-card"><div id="chart12"></div></div></div>
            <div class="col-lg-6"><div class="chart-card"><div id="chart13"></div></div></div>
            <div class="col-12"><div class="chart-card"><div id="chart14"></div></div></div>
        </div>
    </div>

    <script>
        const chartConfigs = __CHARTS_JSON__;
        const defaultPlotConfig = { responsive: true, displayModeBar: false };

        for (const [chartId, spec] of Object.entries(chartConfigs)) {
            if (spec.data && spec.data.length > 0) {
                Plotly.newPlot(chartId, spec.data, spec.layout, defaultPlotConfig);
            }
        }
    </script>
</body>
</html>
"""

# แทนที่ Placeholder ด้วยข้อมูลจริง
output_html = html_template.replace("__TOTAL_RECORDS__", meta["total_records"])
output_html = output_html.replace("__TOTAL_SUBJECTS__", meta["total_subjects"])
output_html = output_html.replace("__YEAR_RANGE__", meta["year_range"])
output_html = output_html.replace("__CHARTS_JSON__", json.dumps(charts))

# บันทึกไฟล์ index.html ที่ Root Directory
with open("index.html", "w", encoding="utf-8") as f:
    f.write(output_html)

print("Successfully generated index.html for GitHub Pages!")