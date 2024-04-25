import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
import tkinter as tk

# Carrega as colunas necessarias a partir do arquivo TXT
with open('colunas_carregadas.txt', 'r') as arquivo:
    colunas_usadas = arquivo.read().splitlines()

# Carregando os dados do arquivo CSV
train_data = pd.read_csv('dataset/test_data.csv', usecols=colunas_usadas)  
test_data = pd.read_csv('dataset/train_data.csv', usecols=colunas_usadas)

# Convertendo e tratando CSV de Treino
train_data = train_data[train_data['ammonia'] != '#NUM!']
train_data['ammonia'] = train_data['ammonia'].astype('float')
train_data['is_safe'] = train_data['is_safe'].astype('int')
# train_data.info()

# Convertendo e tratando CSV de Teste
test_data = test_data[test_data['ammonia'] != '#NUM!']
test_data['ammonia'] = test_data['ammonia'].astype('float')
test_data['is_safe'] = test_data['is_safe'].astype('int')

# Convertendo o dataframe para numerico
train_data = train_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Separando os recursos (features) e o alvo (target)
X_train = train_data.drop('is_safe', axis=1)
y_train = train_data['is_safe'] 

X_test = test_data.drop('is_safe', axis=1)
y_test = test_data['is_safe'] 

# Inicializando e treinando o modelo de IA (XGBoost)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculando o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Dados de entrada se for igual a 0 agua é impropria para consumo
# dados_entrada = [[0,0,0.05,0,0.006,4.43,0.5,1.58,0.13,0.97,0.97,0.17,11.6,1.07,0.006,9.82,0.83,0.06,0.23,0.08]]

# # Dados de entrada se for igual a 1 agua é potavel
# dados_entrada = [[0.47,22.21,0.02,2.07,0.001,0.4,0.21,1.32,1.23,0.56,0.56,0.049,18.67,1.78,0.007,4.51,0.51,0.06,0.2,0.0]]

def fazer_previsao():
    # Dados de entrada fornecidos pelo usuário
    entrada = [[float(entry.get()) for entry in entries]]
    
    # Fazendo previsão nos novos dados
    prediction = model.predict(entrada)
    
    # Exibindo a previsão na caixa de texto
    if prediction == [0]:
        resultado_var.set("Água imprópria para consumo")
        resultado_label.config(fg="red")
    elif prediction == [1]:
        resultado_var.set("Água potável")
        resultado_label.config(fg="green", )

# Configurando a interface gráfica Tkinter
root = tk.Tk()
root.title("Qualidade da Água")

# Criando os campos de entrada para os dados
labels = ['Alumínio:', 'Amônia:', 'Arsênico:', 'Bário:', 'Cádmio:', 
          'Cloramina:', 'Cromo:', 'Cobre:', 'Fluor:', 'Bactérias:', 
          'Vírus:', 'Chumbo:', 'Nitratos:', 'Nitritos:', 'Mercúrio:', 'Perclorato:', 
          'Rádio:', 'Selênio:', 'Prata:', 'Urânio:']
entries = [tk.Entry(root) for _ in range(len(labels))]
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entries[i].grid(row=i, column=1)

# Botão para fazer a previsão
tk.Button(root, text="Fazer Previsão", command=fazer_previsao).grid(row=len(labels)+1, columnspan=2)

# Caixa de texto para mostrar o resultado da previsão
resultado_var = tk.StringVar()
resultado_label = tk.Label(root, textvariable=resultado_var)
resultado_label.grid(row=len(labels)+2, columnspan=2)

root.mainloop()