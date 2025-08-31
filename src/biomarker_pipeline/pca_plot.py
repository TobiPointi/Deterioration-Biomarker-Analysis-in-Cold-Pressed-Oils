from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def pca_2d_feature_scatter(X: pd.DataFrame, title: str, out_prefix: Path, dpi: int = 440, fmt: str = "png") -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(Xs)

    evr = pca.explained_variance_ratio_.tolist()
    (out_prefix.parent / f"{out_prefix.name}_explained_variance.json").write_text(
        json.dumps({"explained_variance_ratio": evr}, indent=2)
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    for i, label in enumerate(X.index.astype(str)):
        plt.text(scores[i, 0], scores[i, 1], label, fontsize=8)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.{fmt}", dpi=dpi)
    plt.savefig(f"{out_prefix}.pdf")
    plt.close()
