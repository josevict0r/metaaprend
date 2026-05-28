import pandas as pd

df = pd.read_csv("meta_features.csv")

df_cls = df[
    (df["nr_class"].notna()) 
]
df_cls = df_cls[df_cls["best_node.histogram.0"].notnull()]
df_cls.to_csv(
    "meta_features_clean.csv",
    index=False
)

print(len(df_cls))