import pandas as pd

meta_dataset = pd.read_csv('meta_dataset.csv')

# Get the classifier columns (last 6 columns)
classifier_cols = meta_dataset.columns[-6:].tolist()

# Find the classifier with the best (maximum) accuracy for each dataset
meta_dataset['Best'] = meta_dataset[classifier_cols].idxmax(axis=1)

meta_dataset = meta_dataset.drop(['time_seconds'], axis=1)

print(pd.DataFrame(meta_dataset['Best'].value_counts()))

#meta_dataset.to_csv('meta_dataset_bestcol.csv')