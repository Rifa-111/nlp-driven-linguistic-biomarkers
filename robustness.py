"""ASA-CD robustness checks.

Usage: python3 -I robustness.py <youtube_csv> <airline_csv> <out_dir>
Reuses the definitions in biomarkers.py (same folder).
"""
import sys, os, json, re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

sys.argv_saved = list(sys.argv)
yt_path, air_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

EXCL = {"they", "them", "their", "theirs", "themselves"}
INCL = {"we", "us", "our", "ours", "ourselves"}
ABS = {"absolutely", "all", "always", "complete", "completely", "constant", "constantly",
       "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
       "never", "nothing", "totally", "whole"}
# words suggesting 'they' may refer to a company, service or its staff
INST = {"airline", "airlines", "flight", "flights", "staff", "crew", "agent", "agents", "gate", "desk",
        "rep", "reps", "service", "pilot", "pilots", "attendant", "attendants", "employee", "employees",
        "company", "companies", "customer", "airport", "baggage", "luggage", "booking", "ticket",
        "apple", "google", "samsung", "microsoft", "tesla", "amazon", "government", "channel", "youtube"}
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")


def load(path, tc, lc, name):
    df = pd.read_csv(path)[[tc, lc]].rename(columns={tc: "text", lc: "sentiment"}).dropna()
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    if name == "airline":
        df["text"] = df["text"].str.replace(r"@\w+", " ", regex=True)
    toks = df["text"].str.lower().apply(TOK.findall)
    df["toks"] = toks
    df["n_words"] = toks.apply(len)
    df = df[df.n_words >= 3].copy()
    for k, lex in (("excl", EXCL), ("incl", INCL), ("abs", ABS)):
        df[k + "_n"] = df.toks.apply(lambda t: sum(w in lex for w in t))
        df[k + "_any"] = (df[k + "_n"] > 0).astype(int)
        df[k + "_rate"] = 100 * df[k + "_n"] / df.n_words
    inst = df.toks.apply(lambda t: any(w in INST for w in t))
    df["excl_inst"] = ((df.excl_any == 1) & inst).astype(int)
    df["excl_other"] = ((df.excl_any == 1) & ~inst).astype(int)
    df["log_words"] = np.log(df.n_words)
    return df


def or_table(m, vars_):
    ci = m.conf_int()
    return {v: {"OR": round(float(np.exp(m.params[v])), 2),
                "CI": [round(float(np.exp(ci.loc[v, 0])), 2), round(float(np.exp(ci.loc[v, 1])), 2)],
                "p": float(m.pvalues[v])} for v in vars_}


def topics(df, k=10, seed=42):
    cv = CountVectorizer(stop_words="english", min_df=5, max_df=0.5, max_features=5000)
    X = cv.fit_transform(df.text.str.lower())
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, learning_method="batch", max_iter=20)
    W = lda.fit_transform(X)
    vocab = np.array(cv.get_feature_names_out())
    top = {i: ", ".join(vocab[lda.components_[i].argsort()[::-1][:6]]) for i in range(k)}
    return W.argmax(1), top


def run(df):
    r = {}
    d = df[df.sentiment.isin(["negative", "positive"])].copy()
    d["neg"] = (d.sentiment == "negative").astype(int)
    base = "excl_any + abs_any + incl_any + log_words"
    r["R0_baseline"] = or_table(smf.logit("neg ~ " + base, d).fit(disp=0), ["excl_any", "abs_any", "incl_any"])
    # R1 rates (per 100 words)
    m = smf.logit("neg ~ excl_rate + abs_rate + incl_rate + log_words", d).fit(disp=0)
    r["R1_rates_per100w"] = or_table(m, ["excl_rate", "abs_rate", "incl_rate"])
    # R2 ordinal across all three classes (higher = more positive); report OR for *more negative*
    o = df.copy()
    o["y"] = pd.Categorical(o.sentiment, categories=["negative", "neutral", "positive"], ordered=True)
    om = OrderedModel(o["y"], o[["excl_any", "abs_any", "incl_any", "log_words"]], distr="logit").fit(method="bfgs", disp=0)
    ci = om.conf_int()
    r["R2_ordinal_OR_more_negative"] = {v: {"OR": round(float(np.exp(-om.params[v])), 2),
        "CI": [round(float(np.exp(-ci.loc[v, 1])), 2), round(float(np.exp(-ci.loc[v, 0])), 2)], "p": float(om.pvalues[v])}
        for v in ["excl_any", "abs_any", "incl_any"]}
    # R3 topic fixed effects
    t, top = topics(df)
    df = df.assign(topic=t)
    d = d.join(df["topic"])
    m = smf.logit("neg ~ " + base + " + C(topic)", d).fit(disp=0, maxiter=200)
    r["R3_topic_fixed_effects"] = or_table(m, ["excl_any", "abs_any", "incl_any"])
    r["R3_topics"] = top
    # R4 referent context split
    m = smf.logit("neg ~ excl_inst + excl_other + abs_any + incl_any + log_words", d).fit(disp=0)
    r["R4_referent_split"] = or_table(m, ["excl_inst", "excl_other"])
    r["R4_share_excl_in_inst_context"] = round(float(d.loc[d.excl_any == 1, "excl_inst"].mean()), 3)
    # R5 length-restricted 5-60 words
    dd = d[(d.n_words >= 5) & (d.n_words <= 60)]
    r["R5_length_5_60"] = or_table(smf.logit("neg ~ " + base, dd).fit(disp=0), ["excl_any", "abs_any", "incl_any"])
    r["R5_n"] = int(len(dd))
    # R6 drop the very common absolutist items 'all' and 'ever'
    abs2 = ABS - {"all", "ever"}
    d["abs2_any"] = d.toks.apply(lambda tk: int(any(w in abs2 for w in tk)))
    r["R6_absolutist_without_all_ever"] = or_table(smf.logit("neg ~ excl_any + abs2_any + incl_any + log_words", d).fit(disp=0), ["abs2_any"])
    return r, df


ytd, air = load(yt_path, "Comment", "Sentiment", "youtube"), load(air_path, "text", "sentiment", "airline")
res = {}
res["youtube"], ytd = run(ytd)
res["airline"], air = run(air)
with open(os.path.join(out, "robustness.json"), "w") as f:
    json.dump(res, f, indent=1)

# Hand-coding sheet: 100 random texts containing an exclusive pronoun per corpus
rng = np.random.default_rng(7)
rows = []
for name, df in (("youtube", ytd), ("airline", air)):
    s = df[df.excl_any == 1].sample(100, random_state=7)
    for i, row in s.iterrows():
        rows.append({"corpus": name, "row_id": i, "sentiment": row.sentiment, "text": row.text,
                     "auto_institutional_context": int(row.excl_inst),
                     "referent_coder1": "", "referent_coder2": ""})
pd.DataFrame(rows).to_csv(os.path.join(out, "referent_coding_sheet.csv"), index=False)
print(json.dumps(res, indent=1))
