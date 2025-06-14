import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNWithZScore
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# Load your data
interactions = pd.read_csv("user_recipe_interactions_v2_boosted.csv")
users = pd.read_csv("new_users_modified.csv")
recipes = pd.read_csv("final_recipes.csv")
recipe_disease = pd.read_csv("recipes_with_disease_labels_v2.csv")

# Merge recipe metadata
recipes_merged = pd.merge(recipes, recipe_disease, on="Recipe_ID")
recipes_merged["cleanedingredients"] = recipes_merged["cleanedingredients"].fillna("").str.lower()

# Prepare Surprise dataset
reader = Reader(rating_scale=(interactions['Final_Rating'].min(), interactions['Final_Rating'].max()))
data = Dataset.load_from_df(interactions[['User_ID', 'Recipe_ID', 'Final_Rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# CF model
sim_options = {
    'name': 'cosine',
    'user_based': True
}
cf_model = KNNWithZScore(sim_options=sim_options)
cf_model.fit(trainset)

# Helper: check health constraints
def is_suitable_recipe(recipe_row, user_row):
    # Disease filter
    disease = str(user_row.get("Disease")).lower() if pd.notna(user_row.get("Disease")) else None
    if disease and disease in recipe_row and recipe_row[disease] != 1:
        return False

    # Diet filter
    diet = user_row.get("User_Diet")
    if pd.notna(diet):
        allowed_diets = {
            "vegetarian": ["vegetarian"],
            "eggitarian": ["vegetarian", "eggetarian"],
            "non vegetarian": ["vegetarian", "eggetarian", "non vegetarian"]
        }
        if recipe_row.get("Diet", "").lower() not in allowed_diets.get(diet.lower(), []):
            return False

    # Allergy & non-preferred food
    blacklist = []
    for col in ["Allergies", "non_preferred_food"]:
        if pd.notna(user_row.get(col)):
            blacklist += [x.strip() for x in user_row[col].lower().split(", ")]
    ingredients = recipe_row.get("cleanedingredients", "")
    if any(b in ingredients for b in blacklist):
        return False

    return True

# Top-N CF recommendations with filtering
def get_recs(user_id, n=10, normalize_scores=True):
    if user_id not in trainset._raw2inner_id_users:
        return []

    all_recipe_ids = interactions['Recipe_ID'].unique()
    user_interacted = interactions[interactions['User_ID'] == user_id]['Recipe_ID'].unique()
    candidate_ids = [rid for rid in all_recipe_ids if rid not in user_interacted]

    predictions = []
    for rid in candidate_ids:
        pred = cf_model.predict(user_id, rid)
        predictions.append((rid, pred.est))

    if not predictions:
        return []

    # Normalize scores if needed
    scores = np.array([score for _, score in predictions])
    if normalize_scores:
        if np.ptp(scores) > 1e-5:
            scaler = MinMaxScaler()
            scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        else:
            scores = scores + np.linspace(1e-5, 1e-4, len(scores))  # Add small noise

        predictions = [(rid, score) for (rid, _), score in zip(predictions, scores)]

    # Filter by user health/diet profile
    user_row = users[users["User_ID"] == user_id]
    if user_row.empty:
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    user_row = user_row.iloc[0]
    safe_predictions = []
    for rid, score in predictions:
        recipe_row = recipes_merged[recipes_merged["Recipe_ID"] == rid]
        if recipe_row.empty or is_suitable_recipe(recipe_row.iloc[0], user_row):
            safe_predictions.append((rid, score))

    # Sort and return
    safe_predictions.sort(key=lambda x: x[1], reverse=True)
    return safe_predictions[:n]

# Optional: CF-only evaluation

def evaluate_ranking_metrics(model, testset, k=10, relevance_threshold=4.0):
    predictions = model.test(testset)
    user_est_true = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, est, true_r))

    precisions, recalls, hit_rates, reciprocal_ranks = [], [], [], []

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

    print("\nCF Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics

top_recs = get_recs(44, 10)


print("\nTop 10 Recipe Recommendations for User 44 (Recipe_ID and Score):\n")
for rid, score in top_recs:
    print(f"Recipe_ID: {rid} - Score: {score:.4f}")
