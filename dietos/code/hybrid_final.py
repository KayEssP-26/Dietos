import pandas as pd
import numpy as np
from cf_final import get_top_n_recommendations as get_cf_recs
from cbf_6 import recommend_cb_for_user

def hybrid_recommendations(user_id, top_n=10):
    """
    Generate hybrid recommendations by combining CF and CBF scores.
    
    Parameters:
        user_id (int): The user ID for which to generate recommendations.
        top_n (int): Number of top recommendations to return.
        alpha (float): Weight for CF score. (1 - alpha) is weight for CBF.
    
    Returns:
        pd.DataFrame: Top N recommended recipes with hybrid scores.
    """
    from cbf_5 import interactions_df
    from cbf_5 import recipes as recipes_df

    user_interacted = interactions_df[interactions_df['User_ID'] == user_id]
    if user_interacted.empty:
        print(f"No interactions found for user {user_id}. Using CBF only.")
        cbf_only = recommend_cb_for_user(user_id, top_n=top_n)
        cbf_only = cbf_only.rename(columns={'normalized_score': 'hybrid_score'})
        cbf_only['cf_score'] = 0.0
        cbf_only['cbf_score'] = cbf_only['hybrid_score']
        cbf_only = cbf_only[["Recipe_ID", "cf_score", "cbf_score", "hybrid_score"]]
        return pd.merge(cbf_only, recipes_df, on="Recipe_ID", how="left")

    cf_recs = get_cf_recs(user_id, n=150)
    cb_recs_df = recommend_cb_for_user(user_id, top_n=150)

    cf_df = pd.DataFrame(cf_recs, columns=['Recipe_ID', 'cf_score'])
    cb_df = cb_recs_df.rename(columns={'normalized_score': 'cbf_score'})[['Recipe_ID', 'cbf_score']]

    merged = pd.merge(cf_df, cb_df, on='Recipe_ID', how='outer')

    cf_base = cf_df['cf_score'].min() * 0.8 if not cf_df.empty else 0
    cbf_base = cb_df['cbf_score'].min() * 0.8 if not cb_df.empty else 0

    merged['cf_score'] = merged['cf_score'].fillna(cf_base)
    merged['cbf_score'] = merged['cbf_score'].fillna(cbf_base)

    alpha = 0.6

    merged['hybrid_score'] = alpha * merged['cf_score'] + (1 - alpha) * merged['cbf_score']
    top_recommendations = merged.sort_values(by='hybrid_score', ascending=False).head(top_n)

    # Merge with recipe names
    top_recommendations = pd.merge(top_recommendations, recipes_df, on="Recipe_ID", how="left")

    return top_recommendations[["Recipe_ID", "RecipeName", "cf_score", "cbf_score", "hybrid_score"]]


from cbf_5 import interactions_df, users
from collections import defaultdict

def evaluate_hybrid_system(user_ids, k=10):
    hits = []
    rr = []
    precision_list = []
    recall_list = []

    for uid in user_ids:
        true_items = interactions_df[interactions_df['User_ID'] == uid]['Recipe_ID'].tolist()
        if not true_items:
            continue

        try:
            recs = hybrid_recommendations(uid, top_n=k)
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

if __name__ == "__main__":
    user_ids = users['User_ID'].unique()
    hybrid_metrics = evaluate_hybrid_system(user_ids, k=10)
    print("\nHybrid Evaluation Metrics:\n", hybrid_metrics)

# Example usage
if __name__ == "__main__":
    user_id = 32
    top_hybrid_recs = hybrid_recommendations(user_id, top_n=10)
    print("Hybrid Recommendations for user", user_id)
    print(top_hybrid_recs)