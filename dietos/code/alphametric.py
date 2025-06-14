import matplotlib.pyplot as plt
import numpy as np
from hybrid import evaluate_hybrid_system, users

user_ids = users["User_ID"].unique()
k = 10
cf_n = 20  # You can adjust or fix this based on earlier tuning
cbf_n = 20

# Range of alpha values to test (from 0 = pure CBF to 1 = pure CF)
alphas = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]

# Store results
hit_rates, mrrs, precisions, recalls, valid_alphas = [], [], [], [], []

for alpha in alphas:
    print(f"Evaluating for alpha = {alpha:.2f}")
    try:
        metrics = evaluate_hybrid_system(user_ids, k=k, alpha=alpha, cf_n=cf_n, cbf_n=cbf_n)
        if any(np.isnan(val) for val in metrics.values()):
            print("⚠️ Skipped due to NaN")
            continue

        hit_rates.append(metrics[f"HitRate@{k}"])
        mrrs.append(metrics[f"MRR@{k}"])
        precisions.append(metrics[f"Precision@{k}"])
        recalls.append(metrics[f"Recall@{k}"])
        valid_alphas.append(alpha)
    except Exception as e:
        print(f"⚠️ Skipped alpha={alpha:.2f}: {e}")
        continue

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(valid_alphas, hit_rates, 'o-', label='HitRate@10')
plt.plot(valid_alphas, mrrs, 's-', label='MRR@10')
plt.plot(valid_alphas, precisions, '^-', label='Precision@10')
plt.plot(valid_alphas, recalls, 'd-', label='Recall@10')
plt.xlabel("Alpha (Weight for CF in Hybrid Score)")
plt.ylabel("Metric Score")
plt.title("Hybrid Recommendation Metrics vs. Alpha Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
