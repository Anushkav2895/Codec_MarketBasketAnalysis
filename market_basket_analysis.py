"""Market Basket Analysis on grocery transactions (Apriori vs FP-Growth).

Run from anywhere:  python market_basket_analysis.py
Needs data/Groceries_dataset.csv  (get it with: python data/download_data.py)
Outputs: images/*.png, strong_rules.csv, rules_all.csv, runtime_comparison.csv
"""
import sys
import time
import warnings
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------- Paths (relative to this file, so it works from any folder) ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Groceries_dataset.csv"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ---------- Settings ----------
MIN_SUPPORT = 0.002          # used for the rules below
SUPPORT_LEVELS = [0.005, 0.003, 0.002]   # compared in the runtime test
N_RUNS = 3                   # repeats per algorithm and support level
STRONG_MIN_CONFIDENCE = 0.05
STRONG_MIN_LIFT = 1.2

if not DATA_FILE.exists():
    sys.exit(f"Dataset not found: {DATA_FILE}\nRun first: python data/download_data.py")

# ---------- 1. Load and clean ----------
df = pd.read_csv(DATA_FILE)
print("Rows:", df.shape[0], "| Columns:", list(df.columns))
print("Missing values:", int(df.isna().sum().sum()), "| Duplicate rows:", int(df.duplicated().sum()))

df = df.dropna(subset=["Member_number", "Date", "itemDescription"])
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["itemDescription"] = df["itemDescription"].str.strip().str.lower()

# One transaction = one customer on one date (no invoice ID in this dataset).
# Duplicate items inside a basket are collapsed because we only need presence/absence.
transactions = (
    df.groupby(["Member_number", "Date"])["itemDescription"]
    .apply(lambda s: sorted(set(s)))
    .tolist()
)
n_tx = len(transactions)
print("Transactions:", n_tx, "| Unique items:", df["itemDescription"].nunique(),
      "| Avg basket size:", round(float(np.mean([len(t) for t in transactions])), 2))

# ---------- 2. EDA charts ----------
top_items = df["itemDescription"].value_counts().head(15)
plt.figure(figsize=(9, 6))
sns.barplot(x=top_items.values, y=top_items.index, palette="viridis")
plt.title("Top 15 most purchased items")
plt.xlabel("Number of purchases")
plt.ylabel("")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "01_top_items.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sizes = pd.Series([len(t) for t in transactions])
sns.countplot(x=sizes, color="#4C78A8", ax=axes[0])
axes[0].set_title("Basket size (items per transaction)")
axes[0].set_xlabel("Items in basket")
monthly = df.groupby(df["Date"].dt.to_period("M")).size()
monthly.plot(ax=axes[1], marker="o", color="#D1495B")
axes[1].set_title("Items sold per month")
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "02_basket_size_monthly.png", dpi=150)
plt.close()

# ---------- 3. Encode baskets ----------
te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# ---------- 4. Apriori vs FP-Growth (same data, same settings) ----------
def timed(func, min_support):
    start = time.perf_counter()
    result = func(basket, min_support=min_support, use_colnames=True)
    return time.perf_counter() - start, result

runtime_rows = []
fi_fp = None
for support in SUPPORT_LEVELS:
    ap_times, fp_times = [], []
    for run in range(N_RUNS):
        # alternate which algorithm goes first so neither always runs "cold"
        if run % 2 == 0:
            t_ap, fi_ap = timed(apriori, support)
            t_fp, fi_fp_run = timed(fpgrowth, support)
        else:
            t_fp, fi_fp_run = timed(fpgrowth, support)
            t_ap, fi_ap = timed(apriori, support)
        ap_times.append(t_ap)
        fp_times.append(t_fp)
    same = set(fi_ap["itemsets"]) == set(fi_fp_run["itemsets"])
    runtime_rows.append({
        "min_support": support,
        "n_itemsets": len(fi_fp_run),
        "apriori_median_s": round(median(ap_times), 3),
        "fpgrowth_median_s": round(median(fp_times), 3),
        "same_itemsets": same,
    })
    if support == MIN_SUPPORT:
        fi_fp = fi_fp_run

runtime_df = pd.DataFrame(runtime_rows)
runtime_df.to_csv(BASE_DIR / "runtime_comparison.csv", index=False)
print("\nRuntime comparison (median of", N_RUNS, "runs, same data and settings):")
print(runtime_df.to_string(index=False))
if not runtime_df["same_itemsets"].all():
    print("WARNING: Apriori and FP-Growth returned different itemsets.")

x = np.arange(len(runtime_df))
plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, runtime_df["apriori_median_s"], 0.4, label="Apriori", color="#4C78A8")
plt.bar(x + 0.2, runtime_df["fpgrowth_median_s"], 0.4, label="FP-Growth", color="#F58518")
plt.xticks(x, [f"{s}\n({n} itemsets)" for s, n in zip(runtime_df["min_support"], runtime_df["n_itemsets"])])
plt.xlabel("Minimum support (number of frequent itemsets found)")
plt.ylabel("Median runtime (seconds)")
plt.title(f"Apriori vs FP-Growth runtime (median of {N_RUNS} runs)")
plt.legend()
plt.tight_layout()
plt.savefig(IMAGES_DIR / "07_runtime_comparison.png", dpi=150)
plt.close()

# ---------- 5. Frequent itemsets ----------
frequent = fi_fp.copy()
frequent["length"] = frequent["itemsets"].apply(len)
frequent["itemset_str"] = frequent["itemsets"].apply(lambda s: " + ".join(sorted(s)))
print("\nFrequent itemsets by size (min_support =", MIN_SUPPORT, "):")
print(frequent["length"].value_counts().sort_index().to_string())

top_single = frequent[frequent["length"] == 1].nlargest(15, "support")
top_multi = frequent[frequent["length"] >= 2].nlargest(15, "support")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=top_single, x="support", y="itemset_str", palette="Blues_r", ax=axes[0])
axes[0].set_title("Top single items by support")
axes[0].set_ylabel("")
sns.barplot(data=top_multi, x="support", y="itemset_str", palette="Oranges_r", ax=axes[1])
axes[1].set_title("Top frequent itemsets (2+ items)")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "03_frequent_itemsets.png", dpi=150)
plt.close()

# ---------- 6. Association rules ----------
try:
    rules = association_rules(fi_fp, metric="lift", min_threshold=1.0)
except TypeError:  # newer mlxtend versions need num_itemsets
    rules = association_rules(fi_fp, num_itemsets=n_tx, metric="lift", min_threshold=1.0)

rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
rules["rule"] = rules["antecedents_str"] + "  ->  " + rules["consequents_str"]
rules["baskets"] = (rules["support"] * n_tx).round().astype(int)  # baskets containing both sides
print("\nRules with lift > 1:", len(rules), "| baskets analysed:", n_tx)

cols = ["rule", "support", "confidence", "lift", "baskets"]
rules[["antecedents_str", "consequents_str", "support", "confidence", "lift", "baskets"]] \
    .sort_values("lift", ascending=False).round(4).to_csv(BASE_DIR / "rules_all.csv", index=False)

print("\nTop 15 rules by lift:")
print(rules.sort_values("lift", ascending=False)[cols].head(15).round(4).to_string(index=False))

strong = rules[(rules["confidence"] >= STRONG_MIN_CONFIDENCE) & (rules["lift"] >= STRONG_MIN_LIFT)]
strong = strong.sort_values(["lift", "confidence"], ascending=False)
print(f"\nStrong rules (confidence >= {STRONG_MIN_CONFIDENCE}, lift >= {STRONG_MIN_LIFT}):", len(strong))
print(strong[cols].round(4).to_string(index=False))
strong[cols].round(4).to_csv(BASE_DIR / "strong_rules.csv", index=False)

# Scatter: support vs confidence, coloured by lift
plt.figure(figsize=(9, 6))
sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis",
                 s=60, alpha=0.8, edgecolor="k", linewidth=0.3)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support (share of baskets containing both items)")
plt.ylabel("Confidence")
plt.title("Association rules: support vs confidence")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "04_rules_scatter.png", dpi=150)
plt.close()

# Top rules by lift
top_rules = strong.head(12) if len(strong) >= 5 else rules.nlargest(12, "lift")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_rules, x="lift", y="rule", palette="magma")
plt.title("Top association rules by lift")
plt.xlabel("Lift")
plt.ylabel("")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "05_top_rules_lift.png", dpi=150)
plt.close()

# Lift heatmap (single item -> single item)
single = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
if len(single) > 0:
    pivot = single.pivot_table(index="antecedents_str", columns="consequents_str", values="lift")
    keep = single.groupby("antecedents_str")["lift"].max().nlargest(15).index
    pivot = pivot.loc[pivot.index.isin(keep)]
    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, cbar_kws={"label": "Lift"})
    plt.title("Lift between item pairs")
    plt.xlabel("Consequent (bought next)")
    plt.ylabel("Antecedent (already in basket)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "06_lift_heatmap.png", dpi=150)
    plt.close()

print("\nDone. Charts are in:", IMAGES_DIR)
print("Saved: strong_rules.csv, rules_all.csv, runtime_comparison.csv")
