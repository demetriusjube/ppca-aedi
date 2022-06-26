# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulação do Call Center
# Number de empregados
employees = 100
# Custo por empregado
wage = 200
# Número independente de chamadas por empregado
n = 50
# Probability de sucesso de cada Chamada
p = 0.04
# Receita por chamada
revenue = 100
# variável aleatória binomial do problema
conversions = np.random.binomial(n, p, size=employees)
# Print some key metrics of our call center
print('Conversão média por empregado: ' + str(round(np.mean(conversions), 2)))
print('Desvio-padrão da conversão por empregado: ' + str(round(np.std(conversions), 2)))
print('Total de conversões: ' + str(np.sum(conversions)))
print('Total de receitas: ' + str(np.sum(conversions)*revenue))
print('Custo Total: ' + str(employees*wage))
print('Lucro: ' + str(np.sum(conversions)*revenue - employees*wage))

# Call Center Simulation (Higher Conversion Rate)
# Number of employees to simulate
employees = 100
# Cost per employee
wage = 200
# Number of independent calls per employee
n = 55
# Probability of success for each call
p = 0.05
p_1 = 0.04
# Revenue per call
revenue = 100
# Binomial random variables of call center employees
conversions_up = np.random.binomial(n, p, size=employees)
# Simulate 1,000 days for our call center
# Number of days to simulate
sims = 1000
sim_conversions_up = [np.sum(np.random.binomial(n, p, size=employees)) for i in range(sims)]
sim_conversions = [np.sum(np.random.binomial(n, p_1, size=employees)) for i in range(sims)]
sim_profits_up = np.array(sim_conversions_up)*revenue - employees*wage
sim_profits = np.array(sim_conversions)*revenue - employees*wage
# Plot and save the results as a histogram
fig, ax = plt.subplots(figsize=(14,7))
ax = sns.distplot(sim_profits, bins=20, label='Resultado original da simulação de Call Center')
ax = sns.distplot(sim_profits_up, bins=20, label='Melhora da Taxa de Conversão', color='red')
ax.set_xlabel("Lucros",fontsize=16)
ax.set_ylabel("Frequência",fontsize=16)
plt.legend()