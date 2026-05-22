import pandas as pd

# ============================================
# LOAD DATASET LIST
# ============================================

df = pd.read_csv("datasets_names.csv")

# first column = dataset names
dataset_col = df.columns[0]

# normalize names
df[dataset_col] = (
    df[dataset_col]
    .astype(str)
    .str.strip()
)

# ============================================
# REMOVE DUPLICATES
# ============================================

df = df.drop_duplicates(subset=[dataset_col])

# ============================================
# REMOVE UNWANTED FAMILIES
# ============================================

patterns_to_remove = [

    # Friedman synthetic datasets
    r"^fri_",

    # Random function meta datasets
    r"^rmft",

    # FOREX datasets
    r"^FOREX_",

    # Generated seed datasets
    r"_seed_",

    # Reproduced datasets
    r"reproduced",

    # Upload/test/demo datasets
    r"upload",
    r"test",
    r"dummy",
    r"example",

    # Auto-generated variants
    r"nrows_",
    r"nclasses_",
    r"ncols_",

    # Common synthetic families
    r"synthetic",
    r"artificial",

    # Frequent noisy replications
    r"copy",
    r"clone",

    # OpenML junk variants
    r"iris.*",
    r".*iris",

]

combined_pattern = "|".join(patterns_to_remove)

mask = ~df[dataset_col].str.contains(
    combined_pattern,
    case=False,
    regex=True,
    na=False
)

filtered_df = df[mask]

# ============================================
# SAVE
# ============================================

filtered_df.to_csv(
    "datasets_filtered.csv",
    index=False
)

# ============================================
# REPORT
# ============================================

print(f"Original datasets: {len(df)}")
print(f"Filtered datasets: {len(filtered_df)}")
print(f"Removed datasets: {len(df) - len(filtered_df)}")