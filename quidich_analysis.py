import json
import random
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, linregress
# Gerar uma lista de 1600 jogadores com pontos e horas de treino aleatórias
def gerar_dados():
    players = []  # Initialize the players list
    for i in range(1, 1601):
        player = {
            "id": i,
            "points": random.randint(50, 500),
            "training_hours": round(random.uniform(5, 100), 2)
        }
        players.append(player)  # Now you can append to the list
    return players  # Return the list of players
dados= gerar_dados()
# Converter a lista de jogadores em JSON
players_json = json.dumps(dados, indent=4)

# Retornar a string JSON
#players_json
df = json.loads(players_json)
#print(df)
#Criar um dataFrame a partir do json
df = pd.DataFrame(df)
#print(df)
#salvar o dataFrame em um arquivo CSV
df.to_csv('players.csv',index=False)
#abrir o arquivo csv
df = pd.read_csv('players.csv')
print(df)
print(df.columns)
#x
points = df['points'].tolist() 
#print(points)
#y 
training_hours = df['training_hours'].tolist() 
#print(training_hours)
#calculando o valor p
# uma das maneiras de cálculo de valor p e correlação usando person da lib stats do scipy
#correlation, p_value = pearsonr(points, training_hours)
#print("P-value:", p_value)
#p_value = round(p_value, 3)
#print("p-value",p_value)
#print(correlation)

# Calcular o coeficiente angular e o intercepto da regressão linear

#outra maneira de calcular o valor p e coeficiente de correlação, além de calcular o intercepto e inclinação
slope, intercept, rvalue, pvalue, stderr = linregress(training_hours, points)
#corficiente de determinação-r ao quadrado
r_squared = rvalue **2
print("Coeficiente angular:", round(slope, 3))
print("Intercepto:", round(intercept) , 3) 
print("coeficiente de correlacao",round(rvalue), 3) 
print("valor p", round(pvalue), 3) 
print("erro padrao",round(stderr, 3))
print("coeficiente de determinação-r quadrado", round(r_squared, 3))
print("points =intercept + slope*training hours" )
f_statistic =f_oneway(points, training_hours)

f_statistic = np.array(f_statistic)

f_statistic = round(f_statistic[0], 3) 
print(" f-statistc",f_statistic)
#minimo método quadrado
# Mínimos quadrados
# Calcular a média dos valores de X e Y
mean_x = np.mean(training_hours)
mean_y = np.mean(points)

# Calcular a soma dos produtos dos desvios de X e Y
sum_xy = 0
for i in range(len(training_hours)):
    sum_xy += (training_hours[i] - mean_x) * (points[i] - mean_y)

# Calcular a soma dos quadrados dos desvios de X
sum_xx = 0
for i in range(len(training_hours)):
    sum_xx += (training_hours[i] - mean_x) ** 2

# Calcular a inclinação
slope = sum_xy / sum_xx

# Calcular o intercepto
intercept = mean_y - slope * mean_x

print("Inclinação (mínimos quadrados):", round(slope, 3))
print("Intercepto (mínimos quadrados):", round(intercept, 3))
# Calcular os resíduos
residuals = []
for i in range(len(training_hours)):
    predicted_y = intercept + slope * training_hours[i]
    residuals.append(points[i] - predicted_y)

#print("Resíduos:", residuals)
# Calcular a variância da inclinação
variance_slope = stderr ** 2
print("Variância da inclinação:", round(variance_slope, 3))
# Calcular a variância dos resíduos
variance_residuals = np.var(residuals)
print("Variância dos resíduos:", round(variance_residuals, 3))
# Intervalo de confiança da inclinação


# Nível de confiança
confidence_level = 0.95

# Calcular o intervalo de confiança da inclinação
confidence_interval = stats.t.interval(confidence_level, len(training_hours) - 2, loc=slope, scale=stderr)

# Imprimir o intervalo de confiança

print("Intervalo de confiança da inclinação:", confidence_interval)
#o modelo apresentado, não captura bem a relação entre horas de treino e notas, um intervalo de confiança próximo a zero, tendo uma baixa correlação, uma alta variabilidade de resíduos e um baixo ajuste na reta, indicando uma possível relação não-linear. No entanto, tem um baixo erro padrão, e tem uma significância estatística, e inclusive passou no teste de hipótese,sendo assim, podendo ser explicado por demais fatores
