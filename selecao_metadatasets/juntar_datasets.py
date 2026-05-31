import pandas as pd

um = pd.read_csv('meta_features.csv')
outro = pd.read_csv('tempo.csv')

resultado = pd.concat([um, outro])

resultado.to_csv('meta_features_completo.csv',index=False)