# Market Basket Analysis

Finding products that are frequently bought together in grocery transactions, to support cross-selling, store layout and promotions.

## Overview
- **Dataset:** Groceries Dataset (Kaggle, heeraldedhia/groceries-dataset)
- **Transactions:** one transaction = one customer on one date
- **Algorithms:** Apriori and FP-Growth (mlxtend)
- **Measures:** support, confidence, lift

## Project structure
    data/                       Groceries_dataset.csv, download_data.py
    images/                     charts saved by the script
    market_basket_analysis.py   full analysis
    strong_rules.csv            filtered association rules
    requirements.txt

## How to run
    pip install -r requirements.txt
    python data/download_data.py
    python market_basket_analysis.py

## Key findings
- Strongest rule: curd -> sausage (lift 1.45)
- Brown bread and canned beer are bought together more often than chance (lift 1.36)
- Sausage is associated with frozen vegetables and bottled beer (lift about 1.22)
- Associations are modest (confidence 5-9%, support 0.2-0.3%), so they should be treated as hypotheses and tested

## Business recommendations
- Test a "frequently bought together" prompt or bundle for curd and sausage
- Place sausage near frozen vegetables and beer to test the layout effect
- Run an A/B test on any promotion before a full rollout

## Limitations
- A transaction is defined as customer + date, since no invoice ID exists
- Rules show association, not causation

## Tools
Python, pandas, mlxtend, matplotlib, seaborn