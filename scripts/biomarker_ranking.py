
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Beispiel (toy data):
polyphenol_results = pd.DataFrame({
    "LASSO": {"Polyphenol1": xx.x, "Polyphenol2": xx.x, ...},
    "RF": {"Polyphenol1": 0.xx, "Polyphenol2": 0.xx, ...},
    "Correlation": {"Polyphenol1": x.xx, "Polyphenol2": x.xx, ...},
})

fatty_acid_results = pd.DataFrame({
    "LASSO": {"Fatty acid1": xx.x, "Fatty acid2": xx.x, ...},
    "RF": {"Fatty acid1": 0.xx, "Fatty acid2": 0.xx, ...},
    "Correlation": {"Fatty acid1": x.xx, "Fatty acid2": x.xx, ...},
})

lipid_oxidation_results = pd.DataFrame({
    "LASSO": {"Sec. lipid ox. product1": xx.x, "Sec. lipid ox. product2": xx.x, ...},
    "RF": {"Sec. lipid ox. product1": 0.xx, "Sec. lipid ox. product2": 0.xx, ...},
    "Correlation": {"Sec. lipid ox. product1": x.xx, "Sec. lipid ox. product2": x.xx, ...},
})

# Combine all results into a single dataset
combined = pd.concat(
    [polyphenol_results, fatty_acid_results, lipid_oxidation_results],
    keys=["Polyphenols", "Fatty Acids", "Lipid Oxidation"]
)

# Normalize LASSO, RF, and Correlation using Min-Max scaling
scaler = MinMaxScaler()
normalized_combined = combined.copy()
normalized_combined[["LASSO", "RF", "Correlation"]] = scaler.fit_transform(
    combined[["LASSO", "RF", "Correlation"]].abs()
)

# Calculate weighted score
weights = {"LASSO": 0.4, "RF": 0.4, "Correlation": 0.2}
normalized_combined["Weighted_Score"] = (
    normalized_combined["LASSO"] * weights["LASSO"] +
    normalized_combined["RF"] * weights["RF"] +
    normalized_combined["Correlation"] * weights["Correlation"]
)

# Sort by Weighted_Score
final_ranked_results = normalized_combined.sort_values(by="Weighted_Score", ascending=False)

if __name__ == "__main__":
    print("\n=== Final Ranked Biomarkers Across Methods (Normalized) ===")
    print(final_ranked_results)
