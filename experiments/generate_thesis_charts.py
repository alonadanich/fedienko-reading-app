import os
import numpy as np
import matplotlib.pyplot as plt

# Налаштування академічного стилю (Times New Roman згідно з вимогами диплома)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

OUTPUT_DIR = "thesis_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# РИСУНОК 4.1: Порівняння підходів та шарів Wav2Vec 2.0 (Random Split)
# ==============================================================================
labels_4_1 = [
    "Fine-Tuning\n(3 епохи)",
    "CNN +\nMean",
    "Шар 24\n(Mean)",
    "Шар 24\n(Mean+Std)",
    "Шари\n12+18+24",
    "Шар 18\n(Mean)",
    "Шар 12\n(Mean)"
]
scores_4_1 = [39.79, 69.20, 83.51, 85.95, 86.47, 89.27, 91.80]
colors_4_1 = ["#D9534F", "#8EA9DB", "#5B9BD5", "#5B9BD5", "#5B9BD5", "#4472C4", "#2F5597"]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
bars = ax.bar(labels_4_1, scores_4_1, color=colors_4_1, width=0.58, edgecolor="none", zorder=3)

ax.set_ylabel("Точність класифікації (Accuracy), %", fontsize=12)
ax.set_ylim(0, 103)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 1.5,
        f"{h:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_4_1_wav2vec_layers_benchmark.png"), dpi=300)
plt.close()

# ==============================================================================
# РИСУНОК 4.2: Спеціалізація шарів при маршрутизації (Speaker-Independent, GroupKFold)
# ==============================================================================
layers = [8, 12, 16, 20]
singles_mlp = [81.67, 78.43, 76.52, 75.29]
pairs_mlp = [61.14, 62.93, 61.17, 59.02]

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
ax.plot(layers, singles_mlp, marker="o", linewidth=2.4, markersize=8,
        color="#C00000", label="Одиничні звуки (Singles, MLP)")
ax.plot(layers, pairs_mlp, marker="s", linewidth=2.4, markersize=8,
        color="#2F5597", label="Двобуквені склади (Pairs, базовий MLP)")

for x, y in zip(layers, singles_mlp):
    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, fontweight="bold", color="#900000")

for x, y in zip(layers, pairs_mlp):
    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, fontweight="bold", color="#1F3864")

ax.set_xlabel("Номер прихованого шару трансформера Wav2Vec 2.0", fontsize=12)
ax.set_ylabel("Speaker-Independent точність, %", fontsize=12)
ax.set_xticks(layers)
ax.set_xticklabels([f"Шар {l}" for l in layers])
ax.set_ylim(53, 88)
ax.grid(True, linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, loc="center right", fontsize=10.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_4_2_model_routing_layers.png"), dpi=300)
plt.close()

# ==============================================================================
# РИСУНОК 4.3: Еволюція Speaker-Independent точності та порівняння моделей
# ==============================================================================
configs_4_3 = [
    "CatBoost + SMOTETomek (склади)",
    "Silero VAD + Pitch (+4) + SVM (63 класи)",
    "Шар 12 + базовий лінійний SVM (63 класи)",
    "Шар 12 + PCA(512) + SVM RBF (63 класи)",
    "Шар 12 + Temporal Pooling + MLP (63 класи)",
    "Шари 12+16 (12288 ознак) + PCA + MLP (склади)",
    "Фінал Склади: Шар 12 + Temporal + MLP (tanh)",
    "Фінал Звуки: Шар 8 + Mean + MLP"
]
scores_4_3 = [56.96, 59.40, 61.40, 63.40, 63.56, 64.93, 66.04, 81.67]
colors_4_3 = [
    "#D9534F", "#ED7D31", "#A5A5A5", "#70AD47",
    "#5B9BD5", "#8FAADC", "#2F5597", "#1E7145"
]

fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=300)
y_pos = np.arange(len(configs_4_3))
bars = ax.barh(y_pos, scores_4_3, color=colors_4_3, height=0.6, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(configs_4_3, fontsize=10.5)
ax.set_xlabel("Чесна точність на незнайомих дикторах (Speaker-Independent), %", fontsize=11.5)
ax.set_xlim(0, 92)
ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + 1.0,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.2f}%",
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_4_3_speaker_independent_benchmark.png"), dpi=300)
plt.close()

print("Усі 3 графіки успішно збережено у папку 'thesis_charts'!")