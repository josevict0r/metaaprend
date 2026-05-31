import pandas as pd

meta_features_df = pd.read_csv(
    'selecao_metadatasets\meta_features_clean_2.csv'
)

performances_df = pd.read_csv(
    'avaliacao_base\classifier_results.csv'
)

performances_df = performances_df.drop(
    columns=['time_seconds'],
    errors='ignore'
)

meta_dataset = pd.merge(
    meta_features_df,
    performances_df,
    on='dataset',
    how='inner'
)

print(meta_dataset.shape)
print(meta_dataset.head())

meta_dataset.to_csv('meta_dataset.csv', index=False)