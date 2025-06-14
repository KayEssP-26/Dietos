import matplotlib.pyplot as plt
import numpy as np
from hybrid import evaluate_hybrid_system, hybrid_recommendations, users

user_ids = users["User_ID"].unique()
k = 10
alph = 0.65
rec_sizes = list(range(10, 251, 10))

# Store valid metric values and corresponding rec_sizes
hit_rates, mrrs, precisions, recalls, valid_sizes = [], [], [], [], []

for rec_n in rec_sizes:
    print(f"Evaluating for CF/CBF candidate size: {rec_n}")
    try:
        metrics = evaluate_hybrid_system(user_ids, k=k, alpha=alph, cf_n=rec_n, cbf_n=rec_n)
        if any(np.isnan(val) for val in metrics.values()):
            print("skipp")
            continue  # Skip this run if any metric is NaN
        hit_rates.append(metrics[f"HitRate@{k}"])
        mrrs.append(metrics[f"MRR@{k}"])
        precisions.append(metrics[f"Precision@{k}"])
        recalls.append(metrics[f"Recall@{k}"])
        valid_sizes.append(rec_n)
    except Exception as e:
        print(f"Skipped {rec_n}: {e}")
        continue


# Plot
plt.figure(figsize=(10, 6))
plt.plot(valid_sizes, hit_rates, label='HitRate@10')
plt.plot(valid_sizes, mrrs, label='MRR@10')
plt.plot(valid_sizes, precisions, label='Precision@10')
plt.plot(valid_sizes, recalls, label='Recall@10')
plt.xlabel("Number of CF/CBF Candidates Used Before Merging")
plt.ylabel("Score")
plt.title("Hybrid Recommendation Metrics vs. Candidate Pool Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
