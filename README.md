
# 🍽️ Dietos: Ingredient-Based Food Recommendation System

**Dietos** is a personalized food recommendation engine built in Python. Unlike most nutrition-based systems, Dietos focuses on **ingredient-level matching** to suggest recipes that align with users' tastes, dietary restrictions, and medical conditions. It employs **Collaborative Filtering (CF)**, **Content-Based Filtering (CBF)**, and a **Hybrid approach** to generate more accurate and safer food recommendations.

---

## 📌 Features

- **Hybrid Recommendation Engine** combining CF and CBF.
- **Ingredient-based logic** over nutritional value for personalization.
- **Health-aware filtering** based on user allergies, diseases (e.g., diabetes), and dietary preferences.
- **Boosting mechanism** for preferred cuisine and diet types.
- **Evaluation metrics**: Hit Rate, MRR, Precision, Recall.
- **Tunable hybrid weight (`alpha`)** to balance CF vs. CBF.

---

## 🧠 Recommender Algorithms

### 1. Content-Based Filtering (CBF)
- Uses **TF-IDF** over cleaned ingredient lists.
- Filters recipes by:
  - Disease compatibility (e.g., diabetes-safe).
  - Allergies and non-preferred foods.
  - Diet type (e.g., vegetarian, eggitarian).
- Boosts scores for matching cuisines and diets.

### 2. Collaborative Filtering (CF)
- Based on user-recipe interaction data (ratings, views, etc.).
- Uses **KNN with Z-score normalization** (User-based CF).
- Filters out already interacted recipes and ranks new ones.

### 3. Hybrid System
- Weighted sum of CF and CBF scores:
  ```
  hybrid_score = α * CF_score + (1 - α) * CBF_score
  ```
- Automatically falls back to CBF when CF data is insufficient.
- Configurable `α` value to balance personalization strategies.

---

## 📁 Project Structure

```bash
├── cbf_final.py             # Content-based filtering module
├── cf_final.py              # Collaborative filtering module
├── hybrid_final.py          # Hybrid recommender logic
├── alphametric.py           # 2D alpha tuning & metric plotting
├── metrics_3d.py            # 3D performance surface plotting
├── final_recipes.csv        # Recipe dataset with ingredients & metadata
├── new_users_modified.csv   # User profile data
├── recipes_with_disease_labels_v2.csv  # Disease-safe recipe labels
├── user_recipe_interactions_v2_boosted.csv  # Interaction & rating data
```
---
## 📊 Example Output

```
Hybrid Recommendations for user 32
| Recipe_ID | RecipeName        | cf_score | cbf_score | hybrid_score |
|-----------|-------------------|----------|-----------|--------------|
| 105       | Grilled Veg Wrap  | 0.78     | 0.82      | 0.80         |
| 203       | Chicken Korma     | 0.76     | 0.71      | 0.74         |
...
```

---

## 🧪 Evaluation Metrics

The system is evaluated using:
- **Hit Rate@K**
- **MRR@K (Mean Reciprocal Rank)**
- **Precision@K**
- **Recall@K**

Metrics can be tuned using the `alpha` parameter and recommendation candidate pool size (`cf_n`, `cbf_n`).

---

## 🔒 Health & Safety Considerations

Dietos ensures:
- **No unsafe ingredients** based on user allergies.
- **Diet compliance** (e.g., avoids meat for vegetarians).
- **Disease-safe recommendations**, using pre-labeled recipe suitability.

---

## 👩‍💻 Built With

- Python
- pandas, NumPy, scikit-learn
- matplotlib, seaborn
- Surprise (for collaborative filtering)

---
