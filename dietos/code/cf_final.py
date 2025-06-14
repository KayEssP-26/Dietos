import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNWithZScore
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load your data
interactions = pd.read_csv("user_recipe_interactions_v2_boosted.csv")
users = pd.read_csv("new_users_modified.csv")
recipes = pd.read_csv("final_recipes.csv")
recipe_disease = pd.read_csv("recipes_with_disease_labels_v2.csv")

# Prepare the Surprise dataset
reader = Reader(rating_scale=(interactions['Final_Rating'].min(), interactions['Final_Rating'].max()))
data = Dataset.load_from_df(interactions[['User_ID', 'Recipe_ID', 'Final_Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 🔁 Use KNNWithZScore (user-based)
sim_options = {
    'name': 'cosine',       # You can also try 'cosine', 'msd'
    'user_based': True       # Set to False for item-based
}
cf_model = KNNWithZScore(sim_options=sim_options)
cf_model.fit(trainset)

# Top-N recommendation function (with normalized scores)
def get_top_n_recommendations(user_id, n=10, normalize_scores=True, epsilon=1e-3):
    all_recipe_ids = interactions['Recipe_ID'].unique()
    user_interacted_recipes = interactions[interactions['User_ID'] == user_id]['Recipe_ID'].unique()
    recipes_to_predict = [rid for rid in all_recipe_ids if rid not in user_interacted_recipes]

    predictions = []
    for recipe_id in recipes_to_predict:
        pred = cf_model.predict(user_id, recipe_id)
        predictions.append((recipe_id, pred.est))
        
    if normalize_scores and predictions:
        scores = np.array([score for _, score in predictions])
        if np.ptp(scores) > 1e-5:
            scores = scores.reshape(-1, 1)
            scaler = MinMaxScaler()
            normalized_scores = scaler.fit_transform(scores).flatten()
            predictions = [(recipe_id, norm_score) for (recipe_id, _), norm_score in zip(predictions, normalized_scores)]
        else:
            base_score = scores[0]
            predictions = [
                (recipe_id, base_score + 1e-4 * i)  # Epsilon increment
                for i, (recipe_id, _) in enumerate(predictions)
            ]

    # Sort and return top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


# Ranking metrics evaluation function
from surprise import accuracy

def evaluate_ranking_metrics(model, testset, k=10, relevance_threshold=4.0):
    from collections import defaultdict
    predictions = model.test(testset)
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, est, true_r))

    precisions = []
    recalls = []
    hit_rates = []
    reciprocal_ranks = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_k = user_ratings[:k]
        relevant_items = [iid for iid, _, true_r in user_ratings if true_r >= relevance_threshold]
        recommended_relevant = [1 for iid, _, true_r in top_k if true_r >= relevance_threshold]
        n_relevant = len(relevant_items)
        n_recommended_relevant = sum(recommended_relevant)
        precisions.append(n_recommended_relevant / k if k > 0 else 0)
        recalls.append(n_recommended_relevant / n_relevant if n_relevant > 0 else 0)
        hit_rates.append(1.0 if n_recommended_relevant > 0 else 0.0)
        rr = 0.0
        for rank, (iid, _, true_r) in enumerate(top_k, start=1):
            if true_r >= relevance_threshold:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    metrics = {
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'HitRate@K': np.mean(hit_rates),
        'MRR@K': np.mean(reciprocal_ranks)
    }
    return metrics
"""
top_recs = get_top_n_recommendations(44, 10)

print("\nTop 10 Recipe Recommendations for User 44 (Recipe_ID and Score):\n")
for rid, score in top_recs:
    print(f"Recipe_ID: {rid} - Score: {score:.4f}")
"""