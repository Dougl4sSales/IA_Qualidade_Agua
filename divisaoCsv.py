import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar o arquivo CSV
data = pd.read_csv('dataset/waterQuality1.csv')

# Dividir os dados em conjuntos de treinamento e teste mantendo a proporção das classes
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['is_safe'], random_state=42)

# Salvar os conjuntos de treinamento e teste em arquivos CSV
train_data.to_csv('dataset/train_data.csv', index=False)
test_data.to_csv('dataset/test_data.csv', index=False)

print("Conjuntos de treinamento e teste foram salvos com sucesso.")
