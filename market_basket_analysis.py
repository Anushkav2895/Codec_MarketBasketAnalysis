# # Market Basket Analysis (Grocery Transactions)
# **Goal:** Find products that are often bought together, so a retailer can improve cross-selling, store layout and promotions.
#
# **Dataset:** [Groceries Dataset (Kaggle)](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset) - columns `Member_number`, `Date`, `itemDescription`.
#
# **Algorithms:** Apriori and FP-Growth (via `mlxtend`)
# **Key measures:** support, confidence, lift

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("images", exist_ok=True)

# ## 1. Load and inspect the data
# Download `Groceries_dataset.csv` from Kaggle and place it in a `data/` folder.

df = pd.read_csv("data/Groceries_dataset.csv")
print(df.shape)
print(df.head())

df.info()
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# ## 2. Build transactions
# There is no invoice number, so one **transaction = one customer on one date** (`Member_number` + `Date`). Duplicate rows are kept as one item per basket because we only care whether an item is present.

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["itemDescription"] = df["itemDescription"].str.strip().str.lower()

transactions = (
    df.groupby(["Member_number", "Date"])["itemDescription"]
    .apply(lambda s: sorted(set(s)))
    .tolist()
)

print("Number of transactions:", len(transactions))
print("Unique items:", df["itemDescription"].nunique())
print("Average basket size:", round(np.mean([len(t) for t in transactions]), 2))
print("Example basket:", transactions[0])

# ## 3. Exploratory analysis
# ### 3.1 Best-selling items

top_items = df["itemDescription"].value_counts().head(15)

plt.figure(figsize=(9, 6))
sns.barplot(x=top_items.values, y=top_items.index, palette="viridis")
plt.title("Top 15 most purchased items")
plt.xlabel("Number of purchases")
plt.ylabel("")
plt.tight_layout()
plt.savefig("images/01_top_items.png", dpi=150)
plt.close()

# ### 3.2 Basket size and monthly trend

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
plt.savefig("images/02_basket_size_monthly.png", dpi=150)
plt.close()

# ## 4. Encode transactions
# Apriori and FP-Growth need a one-hot table: one row per transaction, one True/False column per item.

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
print(basket.shape)
print(basket.head())

# ## 5. Frequent itemsets (Apriori vs FP-Growth)
# Support is the share of transactions that contain an itemset. In grocery data most pairs are rare, so the support threshold has to be low. **If you get too few or too many rules, change `MIN_SUPPORT`.**

MIN_SUPPORT = 0.002

t0 = time.time()
fi_apriori = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
t_apriori = time.time() - t0

t0 = time.time()
fi_fp = fpgrowth(basket, min_support=MIN_SUPPORT, use_colnames=True)
t_fp = time.time() - t0

print(f"Apriori:   {len(fi_apriori)} itemsets in {t_apriori:.2f}s")
print(f"FP-Growth: {len(fi_fp)} itemsets in {t_fp:.2f}s")

frequent_itemsets = fi_fp.copy()
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["itemset_str"] = frequent_itemsets["itemsets"].apply(lambda s: " + ".join(sorted(s)))
print(frequent_itemsets["length"].value_counts().sort_index())

# Both algorithms return the same itemsets; FP-Growth is usually faster on larger data. We continue with the FP-Growth output.
#
# ### 5.1 Visualise frequent itemsets

top_single = frequent_itemsets[frequent_itemsets["length"] == 1].nlargest(15, "support")
top_multi = frequent_itemsets[frequent_itemsets["length"] >= 2].nlargest(15, "support")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=top_single, x="support", y="itemset_str", palette="Blues_r", ax=axes[0])
axes[0].set_title("Top single items by support")
axes[0].set_ylabel("")
sns.barplot(data=top_multi, x="support", y="itemset_str", palette="Oranges_r", ax=axes[1])
axes[1].set_title("Top frequent itemsets (2+ items)")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig("images/03_frequent_itemsets.png", dpi=150)
plt.close()

# ## 6. Association rules
# For a rule *A -> B*:
# - **Support:** how often A and B appear together
# - **Confidence:** of the baskets with A, the share that also contain B
# - **Lift:** how much more often A and B occur together than if they were independent. Lift > 1 means a real positive association.

def build_rules(itemsets, n_transactions):
    try:
        return association_rules(itemsets, metric="lift", min_threshold=1.0)
    except TypeError:  # newer mlxtend versions need num_itemsets
        return association_rules(itemsets, num_itemsets=n_transactions,
                                 metric="lift", min_threshold=1.0)

rules = build_rules(fi_fp, len(basket))
rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
rules["rule"] = rules["antecedents_str"] + "  ->  " + rules["consequents_str"]

print("Number of rules with lift > 1:", len(rules))

cols = ["rule", "support", "confidence", "lift"]
print("\nTop 15 rules by lift:")
print(rules.sort_values("lift", ascending=False)[cols].head(15).round(4).to_string(index=False))

# ### 6.1 Strongest rules
# We keep rules that are both reasonably reliable (confidence) and clearly positive (lift). Adjust the thresholds if the table comes out empty.

strong = rules[(rules["confidence"] >= 0.05) & (rules["lift"] >= 1.2)]
strong = strong.sort_values(["lift", "confidence"], ascending=False)
print("Strong rules:", len(strong))
print("\nStrongest rules:")
print(strong[cols].head(15).round(4).to_string(index=False))
strong[cols].round(4).to_csv("strong_rules.csv", index=False)
print("Saved strong_rules.csv")

# ### 6.2 Support vs confidence, coloured by lift

plt.figure(figsize=(9, 6))
sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                 cmap="viridis", s=60, alpha=0.8, edgecolor="k", linewidth=0.3)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association rules: support vs confidence")
plt.tight_layout()
plt.savefig("images/04_rules_scatter.png", dpi=150)
plt.close()

# ### 6.3 Top rules by lift

top_rules = strong.head(12) if len(strong) >= 5 else rules.nlargest(12, "lift")

plt.figure(figsize=(10, 6))
sns.barplot(data=top_rules, x="lift", y="rule", palette="magma")
plt.title("Top association rules by lift")
plt.xlabel("Lift")
plt.ylabel("")
plt.tight_layout()
plt.savefig("images/05_top_rules_lift.png", dpi=150)
plt.close()

# ### 6.4 Lift heatmap (single item -> single item)

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
    plt.savefig("images/06_lift_heatmap.png", dpi=150)
    plt.close()
else:
    print("No single-item rules found. Try lowering MIN_SUPPORT.")

# ## 7. Business insights
# *(Check these against your own tables and charts before publishing. Replace the examples with the real pairs you see.)*
#
# 1. **Bundle offers:** put the highest-lift pairs in the "Strongest rules" table into combo deals or "frequently bought together" prompts.
# 2. **Store layout:** place items with strong associations near each other, so a customer picking up the first is reminded of the second.
# 3. **Targeted promotions:** discount the high-traffic item (such as whole milk or other vegetables) and promote its associated item at full price.
# 4. **Online recommendations:** use rules with high confidence as a simple "customers also bought" feature on the website or app.
# 5. **Stock planning:** items that appear in many strong rules should never run out, because a missing item can lose the whole basket.
#
# **Limitations:** a transaction here is defined as customer + date, the data covers a limited period, and rules show association, not causation. Test any promotion with an A/B experiment before rolling it out.

# ## 8. Conclusion
# Apriori and FP-Growth found the same frequent itemsets, and association rules (filtered by lift and confidence) reveal which products are bought together. These can feed bundles, shelf placement, and recommendations.
#
# **Possible next steps:** analyse rules by season or customer segment, add profit margins to rank rules by value, and build a small recommendation function that takes a basket and suggests the next item.

print("\nDone. Charts are in the images folder.")
