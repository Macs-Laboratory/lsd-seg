from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.manifold import TSNE

from ..metrics import predictive_entropy


def wilcoxon_table(result_paths: list[str], metric: str, output_path: str) -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in result_paths]
    base = dfs[0][metric]
    rows = []
    for index, df in enumerate(dfs[1:], start=1):
        stat, pvalue = wilcoxon(base, df[metric])
        rows.append({"comparison_index": index, "statistic": stat, "pvalue": pvalue})
    output = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def plot_umap_like(descriptors, labels, output_path: str) -> None:
    try:
        import umap

        projection = umap.UMAP(n_components=2, random_state=42).fit_transform(descriptors)
    except Exception:
        projection = TSNE(n_components=2, random_state=42, init="pca").fit_transform(descriptors)
    df = pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1], "label": labels})
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="x", y="y", hue="label", s=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_win_loss_heatmap(frame: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(12, 5))
    sns.heatmap(frame, cmap="RdYlGn", center=0.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def win_loss_heatmap_from_csv(csv_path: str, ours_name: str, output_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    pivot = frame.pivot(index="dataset", columns="model", values="dice")
    if ours_name not in pivot.columns:
        raise ValueError(f"Ours model '{ours_name}' not found in {csv_path}")
    diff = pd.DataFrame({column: pivot[ours_name] - pivot[column] for column in pivot.columns if column != ours_name})
    plot_win_loss_heatmap(diff, output_path)
    return diff


def plot_prototype_evolution(history: list[int], output_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history, linewidth=2)
    plt.xlabel("Update step")
    plt.ylabel("Prototype count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sensitivity(frame: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=frame, x="value", y="dice", hue="parameter", marker="o")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_qualitative_segmentation_grid(
    images: list,
    targets: list,
    baseline_predictions: dict[str, list],
    ours_predictions: list,
    output_path: str,
) -> None:
    rows = len(images)
    columns = 3 + len(baseline_predictions)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    if rows == 1:
        axes = [axes]
    baseline_names = list(baseline_predictions)
    for row_idx in range(rows):
        row_axes = axes[row_idx]
        entries = [images[row_idx], targets[row_idx], *[baseline_predictions[name][row_idx] for name in baseline_names], ours_predictions[row_idx]]
        labels = ["Input", "GT", *baseline_names, "Ours"]
        for ax, entry, label in zip(row_axes, entries, labels):
            ax.imshow(entry, cmap="gray")
            ax.set_title(label)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_routing_uncertainty_panels(
    image,
    target,
    expert_a,
    expert_b,
    variance_map,
    final_prob,
    output_path: str,
) -> None:
    entropy_map = predictive_entropy(torch.as_tensor(final_prob)).squeeze().cpu().numpy()
    panels = [image, target, expert_a, expert_b, variance_map, final_prob]
    titles = ["Input + GT", "GT", "Expert #3", "Expert #8", "Predictive variance", "Final output"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, panel, title in zip(axes.flatten(), panels, titles):
        ax.imshow(panel, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
