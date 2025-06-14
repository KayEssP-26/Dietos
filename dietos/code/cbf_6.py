import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load data
recipes = pd.read_csv("final_recipes.csv")
disease = pd.read_csv("recipes_with_disease_labels_v2.csv")
users = pd.read_csv("new_users_modified.csv")
interactions_df = pd.read_csv("user_recipe_interactions_v2_boosted.csv")

# Merge and preprocess
recipes_merged = pd.merge(recipes, disease, on="Recipe_ID")
recipes_merged["cleanedingredients"] = recipes_merged["cleanedingredients"].fillna("").str.lower()

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(recipes_merged["cleanedingredients"])

# User profile vector
from scipy.sparse import csr_matrix

def get_user_profile_vector(user_row):
    if pd.notna(user_row.get("preferred_food")):
        return tfidf_vectorizer.transform([" ".join(user_row["preferred_food"].lower().split(", "))])

    uid = user_row["User_ID"]
    history = interactions_df[interactions_df["User_ID"] == uid]["Recipe_ID"]
    if not history.empty:
        history_ingredients = recipes_merged[recipes_merged["Recipe_ID"].isin(history)]["cleanedingredients"]
        return tfidf_vectorizer.transform([" ".join(history_ingredients)])

    # Fallback: sparse mean vector
    return csr_matrix(tfidf_matrix.mean(axis=0))

# Recipe recommendation function
def recommend_cb_for_user(user_id, top_n=10):
    user = users[users["User_ID"] == user_id].iloc[0]
    filtered = recipes_merged.copy()
    # === Stage 1: Hard filtering ===
    # Disease filter
    disease_col = str(user.get("Disease")).lower() if pd.notna(user.get("Disease")) else None
    if disease_col and disease_col in filtered.columns:
        filtered = filtered[filtered[disease_col] == 1]
    # Diet filter
    diet = user.get("User_Diet")
    if pd.notna(diet):
        diet = diet.lower()
        if diet == "vegetarian":
            filtered = filtered[filtered["Recipe_Diet"].str.lower() == "vegetarian"]
        elif diet == "eggetarian":
            filtered = filtered[filtered["Recipe_Diet"].str.lower().isin(["vegetarian", "eggetarian"])]
        elif diet == "non vegetarian":
            filtered = filtered[filtered["Recipe_Diet"].str.lower().isin(["vegetarian", "eggetarian", "non vegetarian"])]
    # === Stage 2: Penalty scoring ===
    blacklist = []
    for col in ["Allergies", "non_preferred_food"]:
        if pd.notna(user.get(col)):
            blacklist += user[col].lower().split(", ")
    blacklist = [b.strip() for b in blacklist if b.strip()]
    def penalty(ingredients):
        return sum(b in ingredients for b in blacklist)
    filtered["penalty"] = filtered["cleanedingredients"].apply(penalty)
    recipe_vecs = tfidf_vectorizer.transform(filtered["cleanedingredients"])
    user_vec = get_user_profile_vector(user)
    cosine_scores = cosine_similarity(user_vec, recipe_vecs).flatten()
    filtered["cosine_score"] = cosine_scores

    # === Boosting ===
    def boost(row):
        boost_score = 0.0
        if pd.notna(user.get("Preferred_Cuisines")) and pd.notna(row.get("Cuisine")):
            if row["Cuisine"].lower() in user["Preferred_Cuisines"].lower():
                boost_score += 0.3
        if pd.notna(user.get("User_Diet")) and pd.notna(row.get("Diet")):
            if row["Diet"].lower() == user["User_Diet"].lower():
                boost_score += 0.6
        return boost_score
    filtered["boost"] = filtered.apply(boost, axis=1)
    # === Final scoring ===
    filtered["total_score"] = filtered["cosine_score"] + filtered["boost"]
    if filtered["total_score"].nunique() == 1:
        noise = np.random.uniform(0.001, 0.1, size=len(filtered))
        filtered["total_score"] += noise

    scaler = MinMaxScaler()
    filtered["score"] = scaler.fit_transform(filtered[["total_score"]])

    return filtered.sort_values(by="score", ascending=False)[
        ["Recipe_ID", "score"]
    ].head(top_n)

# Evaluation
def evaluate_cb_system(user_ids, interactions_df, k=10):
    hits = []
    rr = []
    precision_list = []
    recall_list = []

    for uid in user_ids:
        true_items = interactions_df[interactions_df['User_ID'] == uid]['Recipe_ID'].tolist()
        if not true_items:
            continue

        try:
            recs = recommend_cb_for_user(uid, top_n=k)
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
        "HitRate@{}".format(k): np.mean(hits),
        "MRR@{}".format(k): np.mean(rr),
        "Precision@{}".format(k): np.mean(precision_list),
        "Recall@{}".format(k): np.mean(recall_list),
    }
    return metrics


# Run
"""
user_ids = users["User_ID"].unique()
metrics = evaluate_cb_system(user_ids, interactions_df, k=10)
print(metrics)"""

#print(recommend_cb_for_user(44, 10))