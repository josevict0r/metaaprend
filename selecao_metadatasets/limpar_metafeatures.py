import pandas as pd

df = pd.read_csv("meta_features_completo.csv")
print('Antes:' + str(len(df)))
df_cls = df[
    (df["nr_class"].notna()) 
]
df_cls = df_cls[df_cls["best_node.histogram.0"].notnull()]
df_cls.to_csv(
    "meta_features_clean_2.csv",
    index=False
)

print('Depois:' + str(len(df_cls)))