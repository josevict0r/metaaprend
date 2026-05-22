import pandas as pd

df = pd.read_csv('filtered_datasets.csv')

print(df.columns)

df = df.drop(['did', 'version', 'uploader', 'status', 'format',
       'MajorityClassSize', 'MaxNominalAttDistinctValues', 'MinorityClassSize',
       'NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances',
       'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues',
       'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures'], axis=1)


df = df.drop_duplicates()

print(df.head)

df.to_csv('datasets_names.csv', index= False)