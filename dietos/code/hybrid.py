import pandas as pd
import numpy as np
from cf_final import get_top_n_recommendations as get_cf_recs
from cbf_6 import recommend_cb_for_user, interactions_df, users, recipes_merged as recipes_df

from cbf_6 import recipes_merged as recipes_df

def hybrid_recommendations(user_id, top_n=10, alpha=0.5, cf_n=150, cbf_n=150):
    user_interacted = interactions_df[interactions_df['User_ID'] == user_id]

    if user_interacted.empty:
        cbf_only = recommend_cb_for_user(user_id, top_n=top_n)
        cbf_only = cbf_only.rename(columns={'score': 'cbf_score'})
        cbf_only['cf_score'] = 0.0
        cbf_only['hybrid_score'] = cbf_only['cbf_score']
        return cbf_only.sort_values(by="hybrid_score", ascending=False)["Recipe_ID"].tolist()

    cf_recs = get_cf_recs(user_id, n=cf_n)
    cb_recs_df = recommend_cb_for_user(user_id, top_n=cbf_n)

    user = users[users["User_ID"] == user_id].iloc[0]
    disease_col = str(user.get("Disease")).lower() if pd.notna(user.get("Disease")) else None
    if disease_col and disease_col in recipes_df.columns:
        allowed_recipes = set(recipes_df[recipes_df[disease_col] == 1]["Recipe_ID"])
        cf_recs = [(rid, score) for (rid, score) in cf_recs if rid in allowed_recipes]

    cf_df = pd.DataFrame(cf_recs, columns=['Recipe_ID', 'cf_score'])
    cb_df = cb_recs_df.rename(columns={'score': 'cbf_score'})[['Recipe_ID', 'cbf_score']]

    merged = pd.merge(cf_df, cb_df, on='Recipe_ID', how='outer')

    cf_base = cf_df['cf_score'].min() * 0.9 if not cf_df.empty else 0
    cbf_base = cb_df['cbf_score'].min() * 0.9 if not cb_df.empty else 0

    merged['cf_score'] = merged['cf_score'].astype(float).fillna(cf_base)
    merged['cbf_score'] = merged['cbf_score'].astype(float).fillna(cbf_base)

    merged['hybrid_score'] = alpha * merged['cf_score'] + (1 - alpha) * merged['cbf_score']
    top_recommendations = merged.sort_values(by='hybrid_score', ascending=False).head(top_n)

    return top_recommendations["Recipe_ID"].tolist()

def evaluate_hybrid_system(user_ids, k=10, alpha=0.5, cf_n=150, cbf_n=150):
    hits = []
    rr = []
    precision_list = []
    recall_list = []

    for uid in user_ids:
        true_items = interactions_df[interactions_df['User_ID'] == uid]['Recipe_ID'].tolist()
        if not true_items:
            continue

        try:
            recs = hybrid_recommendations(uid, top_n=k, alpha=alpha, cf_n=cf_n, cbf_n=cbf_n)
        except Exception:
            continue

        predicted_items = recs['Recipe_ID'].tolist()
        hit = any(item in true_items for item in predicted_items)
        hits.append(int(hit))

        rank = 0
        for i, item in enumerate(predicted_items):
            if item in true_items:
                rank = i + 1
                break
        rr.append(1.0 / rank if rank > 0 else 0.0)

        true_set = set(true_items)
        pred_set = set(predicted_items)
        intersection = pred_set.intersection(true_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(true_set) if true_set else 0
        precision_list.append(precision)
        recall_list.append(recall)

    metrics = {
        f"HitRate@{k}": np.mean(hits),
        f"MRR@{k}": np.mean(rr),
        f"Precision@{k}": np.mean(precision_list),
        f"Recall@{k}": np.mean(recall_list),
    }

    return metrics
"""
def main():
    user_ids = users['User_ID'].unique()
    print("🔍 Running Hybrid Evaluation...")
    metrics = evaluate_hybrid_system(user_ids, k=10, alpha=0.2, cf_n=150, cbf_n=150)
    print("\n📊 Hybrid Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    recs = hybrid_recommendations(20, 10)
    print(f"\n📋 Top 10 Hybrid Recommendations for User 20:\n")
    for _, row in recs.iterrows():
        print(f"Recipe_ID: {int(row['Recipe_ID'])} - Hybrid Score: {row['hybrid_score']:.4f}")

if __name__ == "__main__":
    main()
"""

