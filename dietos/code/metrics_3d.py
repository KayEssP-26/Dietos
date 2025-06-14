import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from hybrid import evaluate_hybrid_system, users

# Setup
user_ids = users["User_ID"].unique()
k = 10
alpha_vals = np.round(np.linspace(0.0, 1.0, 11), 2)       # [0.0, 0.1, ..., 1.0]
rec_sizes = list(range(50, 301, 25))                      # [50, 75, ..., 300]

# Collect metrics
results = []

for alpha in alpha_vals:
    for rec_n in rec_sizes:
        print(f"Evaluating: alpha={alpha}, rec_n={rec_n}")
        try:
            metrics = evaluate_hybrid_system(user_ids, k=k, alpha=alpha, cf_n=rec_n, cbf_n=rec_n)
            if any(np.isnan(val) for val in metrics.values()):
                continue
            results.append({
                "Alpha": alpha,
                "RecSize": rec_n,
                "HitRate": metrics[f"HitRate@{k}"]
            })
        except Exception as e:
            print(f"⚠️ Skipped alpha={alpha}, rec_n={rec_n}: {e}")
            continue

# Convert to DataFrame
df = pd.DataFrame(results)

# Pivot to 2D array for surface plot
pivot = df.pivot(index="Alpha", columns="RecSize", values="HitRate")
alphas = pivot.index.values
rec_sizes = pivot.columns.values
X, Y = np.meshgrid(rec_sizes, alphas)
Z = pivot.values

# Plot 3D surface
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.5)

# Labels
ax.set_xlabel("Candidate Pool Size (rec_n)", labelpad=10)
ax.set_ylabel("Alpha (CF weight)", labelpad=10)
ax.set_zlabel("HitRate@10", labelpad=10)
ax.set_title("HitRate@10 Surface Plot: Alpha vs Candidate Size", pad=20)

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=10, label="HitRate@10")

plt.tight_layout()
plt.show()
