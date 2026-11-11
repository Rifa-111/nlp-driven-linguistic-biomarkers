"""ASA-CD linguistic biomarker analysis on two public, human-labelled corpora.

Usage: python3 -I biomarkers.py <youtube_csv> <airline_csv> <out_dir>
"""
import sys, re, json, os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

yt_path, air_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

EXCL = {"they", "them", "their", "theirs", "themselves"}
INCL = {"we", "us", "our", "ours", "ourselves"}
# Absolutist dictionary (Al-Mosaiwi & Johnstone, 2018)
ABS = {"absolutely", "all", "always", "complete", "completely", "constant", "constantly",
       "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
       "never", "nothing", "totally", "whole"}
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")
rng = np.random.default_rng(42)


def load(path, text_col, lab_col, name):
    df = pd.read_csv(path)[[text_col, lab_col]].rename(columns={text_col: "text", lab_col: "sentiment"})
    df = df.dropna()
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    if name == "airline":
        df["text"] = df["text"].str.replace(r"@\w+", " ", regex=True)  # drop @handles
    df["corpus"] = name
    return df


def feats(df):
    toks = df["text"].str.lower().apply(TOK.findall)
    df["n_words"] = toks.apply(len)
    df = df[df["n_words"] >= 3].copy()
    toks = df["text"].str.lower().apply(TOK.findall)
    for k, lex in (("excl", EXCL), ("incl", INCL), ("abs", ABS)):
        df[k + "_n"] = toks.apply(lambda t: sum(w in lex for w in t))
        df[k + "_any"] = (df[k + "_n"] > 0).astype(int)
        df[k + "_rate"] = 100 * df[k + "_n"] / df["n_words"]
    df["no_incl"] = 1 - df["incl_any"]
    return df


def boot_ci(x, n=2000):
    x = np.asarray(x)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(df):
    res = {"n_comments": int(len(df)), "sentiment_counts": df["sentiment"].value_counts().to_dict(),
           "median_words": float(df["n_words"].median())}
    order = ["negative", "neutral", "positive"]
    tab = {}
    for s in order:
        d = df[df.sentiment == s]
        row = {}
        for k in ("excl", "abs", "incl"):
            p = d[k + "_any"].mean()
            lo, hi = boot_ci(d[k + "_any"])
            row[k + "_prev"] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)]
            row[k + "_rate_mean"] = round(d[k + "_rate"].mean(), 2)
        row["no_incl_pct"] = round(100 * d["no_incl"].mean(), 1)
        tab[s] = row
    res["by_sentiment"] = tab
    overall = {k + "_prev": round(100 * df[k + "_any"].mean(), 1) for k in ("excl", "abs", "incl")}
    overall.update({k + "_mean_count": round(df[k + "_n"].mean(), 3) for k in ("excl", "abs", "incl")})
    overall["no_incl_pct"] = round(100 * df["no_incl"].mean(), 1)
    res["overall"] = overall
    tests = {}
    for k in ("excl", "abs", "incl"):
        ct = pd.crosstab(df["sentiment"], df[k + "_any"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        kw = stats.kruskal(*[df[df.sentiment == s][k + "_rate"] for s in order])
        tests[k] = {"chi2": round(chi2, 2), "df": int(dof), "p": float(p), "cramers_v": round(float(v), 3),
                    "kruskal_H": round(float(kw.statistic), 2), "kruskal_p": float(kw.pvalue)}
    res["tests"] = tests
    # logistic regression: negative vs positive, controlling for length
    d = df[df.sentiment.isin(["negative", "positive"])].copy()
    d["neg"] = (d.sentiment == "negative").astype(int)
    d["log_words"] = np.log(d["n_words"])
    m = smf.logit("neg ~ excl_any + abs_any + incl_any + log_words", data=d).fit(disp=0)
    ci = m.conf_int()
    res["logit_neg_vs_pos"] = {
        v: {"OR": round(float(np.exp(m.params[v])), 2), "CI": [round(float(np.exp(ci.loc[v, 0])), 2),
            round(float(np.exp(ci.loc[v, 1])), 2)], "p": float(m.pvalues[v])}
        for v in ("excl_any", "abs_any", "incl_any", "log_words")}
    res["logit_n"] = int(len(d))
    res["logit_pseudo_r2"] = round(float(m.prsquared), 3)
    return res


yt = feats(load(yt_path, "Comment", "Sentiment", "youtube"))
air = feats(load(air_path, "text", "sentiment", "airline"))
results = {"youtube": analyse(yt), "airline": analyse(air)}
with open(os.path.join(out, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Figure: prevalence of each biomarker by sentiment, both corpora
labels = {"excl": "Exclusive pronouns\n(they/them/their)", "abs": "Absolutist / generalising\nwords",
          "incl": "Inclusive pronouns\n(we/us/our)"}
cols = {"negative": "#b4452f", "neutral": "#8a8a8a", "positive": "#2f6f8f"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, title) in zip(axes, (("youtube", "A. YouTube comments"), ("airline", "B. Airline tweets"))):
    t = results[name]["by_sentiment"]
    x = np.arange(3)
    for i, s in enumerate(["negative", "neutral", "positive"]):
        vals = [t[s][k + "_prev"][0] for k in labels]
        err = [[t[s][k + "_prev"][0] - t[s][k + "_prev"][1] for k in labels],
               [t[s][k + "_prev"][2] - t[s][k + "_prev"][0] for k in labels]]
        ax.bar(x + (i - 1) * 0.26, vals, 0.26, yerr=err, capsize=2, color=cols[s], label=s.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(labels.values(), fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("% of texts containing marker")
axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, "figure_biomarkers.png"), dpi=300)
print(json.dumps(results, indent=1))
