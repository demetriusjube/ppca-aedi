#!/usr/bin/env python
# coding: utf-8

# # Revisão de Estatística Descritiva - Aplicação no Mercado Financeiro

# ## Importando as Bibliotecas

# In[76]:


import pandas as pd
from pandas_datareader import data
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy import stats
from scipy import optimize


# ### Baixando os dados de uma ação para da B3

# In[2]:


bb_df = data.DataReader(name='BBAS3.SA',
                           data_source='yahoo', start='2000-01-01')
bb_df


# ### Construindo uma Base de Dados Financeiros com Mais Ações

# In[3]:


acoes = ['BBAS3.SA', 'BRFS3.SA', 'CIEL3.SA', 'PETR3.SA', 'MGLU3.SA', '^BVSP']
acoes


# In[4]:


acoes_df = pd.DataFrame()
for acao in acoes:
    acoes_df[acao] = data.DataReader(acao,
                                     data_source='yahoo', start='2015-01-01')['Close']


# In[5]:


acoes_df


# In[6]:


acoes_df.reset_index(inplace=True)
acoes_df


# ### Visualização dos Dados

# In[8]:


figura = px.line(title = 'Histórico do preço das ações')
for i in acoes_df.columns[1:]:
  figura.add_scatter(x = acoes_df["Date"] ,y = acoes_df[i], name = i)
figura.show()


# Taxa de Retorno de Ações

# $$ \mathbb{E} [ R_i] = log \left( \frac{P_t}{P_{t-1}} \right) $$

# In[9]:


dataset = acoes_df.copy()
dataset


# In[10]:


dataset.drop(labels = ['Date'], axis=1, inplace=True)
dataset


# In[12]:


dataset.shift(1)


# In[11]:


taxas_retorno = np.log(dataset / dataset.shift(1))
taxas_retorno


# In[13]:


taxas_retorno.describe()


# In[21]:


medias = (taxas_retorno[acoes].sum()/len(taxas_retorno[acoes]))*100
medias


# In[14]:


taxas_retorno.mean()*100


# In[31]:


vars_acoes = ((taxas_retorno[acoes] - taxas_retorno.mean()) ** 2).sum() / (len(taxas_retorno[acoes]) - 1)
vars_acoes


# In[29]:


taxas_retorno.var()


# In[15]:


taxas_retorno.std()*100


# In[17]:


dataset_date = acoes_df.copy()
date = dataset_date.filter(["Date"]) 
date


# In[18]:


taxas_retorno_date = pd.concat([date, taxas_retorno], axis=1)
taxas_retorno_date


# In[19]:


figura = px.line(title = 'Histórico de retorno das ações')
for i in taxas_retorno_date.columns[1:]:
  figura.add_scatter(x = taxas_retorno_date["Date"] ,y = taxas_retorno_date[i], name = i)
figura.show()


# In[35]:


taxas_retorno.cov()


# In[36]:


taxas_retorno.corr()


# In[37]:


plt.figure(figsize=(8,8))
sns.heatmap(taxas_retorno.corr(), annot=True);


# #### Montando uma Carteira de Ativos

# In[39]:


taxas_retorno_date["CARTEIRA"] = (taxas_retorno_date["BBAS3.SA"] + taxas_retorno_date["BRFS3.SA"] + 
                                   taxas_retorno_date["CIEL3.SA"] + taxas_retorno_date["PETR3.SA"] + 
                                   taxas_retorno_date["MGLU3.SA"])/5
taxas_retorno_date


# In[40]:


taxas_retorno_port = taxas_retorno_date.filter(["Date", "CARTEIRA", "^BVSP"])
taxas_retorno_port


# In[43]:


figura = px.line(title = 'Comparação de retorno Carteira x Ibovespa')
for i in taxas_retorno_port.columns[1:]:
  figura.add_scatter(x = taxas_retorno_port["Date"] ,y = taxas_retorno_port[i], name = i)
figura.add_hline(y = taxas_retorno_port['CARTEIRA'].mean(), line_color="green", line_dash="dot", )
figura.show()


# In[44]:


taxas_retorno_port_corr = taxas_retorno_date.filter(["CARTEIRA", "^BVSP"])
taxas_retorno_port_corr


# In[45]:


plt.figure(figsize=(8,8))
sns.heatmap(taxas_retorno_port_corr.corr(), annot=True);


# #### Alocação Aleatória de Ativos - Portfólio Markowitz

# In[49]:


acoes_port = acoes_df.copy()
acoes_port.drop(labels = ['^BVSP'], axis=1, inplace=True)
acoes_port


# In[50]:


def alocacao_ativos(dataset, dinheiro_total, seed = 0, melhores_pesos = []):
  dataset = dataset.copy()

  if seed != 0:
    np.random.seed(seed)

  if len(melhores_pesos) > 0:
    pesos = melhores_pesos
  else:  
    pesos = np.random.random(len(dataset.columns) - 1)
    #print(pesos, pesos.sum())
    pesos = pesos / pesos.sum()
    #print(pesos, pesos.sum())

  colunas = dataset.columns[1:]
  #print(colunas)
  for i in colunas:
    dataset[i] = (dataset[i] / dataset[i][0])

  for i, acao in enumerate(dataset.columns[1:]):
    #print(i, acao)
    dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total
  
  dataset['soma valor'] = dataset.sum(axis = 1)

  datas = dataset['Date']
  #print(datas)

  dataset.drop(labels = ['Date'], axis = 1, inplace = True)
  dataset['taxa retorno'] = 0.0

  for i in range(1, len(dataset)):
    dataset['taxa retorno'][i] = np.log(dataset['soma valor'][i] / dataset['soma valor'][i - 1]) * 100

  acoes_pesos = pd.DataFrame(data = {'Ações': colunas, 'Pesos': pesos})

  return dataset, datas, acoes_pesos, dataset.loc[len(dataset) - 1]['soma valor']


# In[51]:


dataset, datas, acoes_pesos, soma_valor = alocacao_ativos(acoes_port, 10000, 10)


# In[52]:


dataset


# In[53]:


acoes_pesos


# In[54]:


datas


# In[55]:


soma_valor


# In[56]:


figura = px.line(x = datas, y = dataset['taxa retorno'], title = 'Retorno diário do portfólio',
                labels=dict(x="Data", y="Retorno %"))
figura.add_hline(y = dataset['taxa retorno'].mean(), line_color="red", line_dash="dot", )
figura.show()


# In[57]:


figura = px.line(title = 'Evolução do patrimônio')
for i in dataset.drop(columns = ['soma valor', 'taxa retorno']).columns:
  figura.add_scatter(x = datas, y = dataset[i], name = i)
figura.show()


# In[58]:


figura = px.line(x = datas, y = dataset['soma valor'], 
                 title = 'Evolução do patrimônio da Carteira',
                 labels=dict(x="Data", y="Valor R$"))
figura.add_hline(y = dataset['soma valor'].mean(), 
                 line_color="green", line_dash="dot", )
figura.show()


# #### Mais estatísticas sobre o portfólio aleatório

# In[59]:


# Retorno
dataset.loc[len(dataset) - 1]['soma valor'] / dataset.loc[0]['soma valor'] - 1


# In[60]:


# Desvio-Padrão
dataset['taxa retorno'].std()


# In[61]:


# Sharpe Ratio
(dataset['taxa retorno'].mean() / dataset['taxa retorno'].std())


# In[62]:


dinheiro_total = 10000
soma_valor - dinheiro_total


# ## Simulação da Fronteira Eficiente

# In[64]:


acoes_port


# In[65]:


log_ret = acoes_port.copy()
log_ret.drop(labels = ["Date"], axis = 1, inplace = True)
log_ret = np.log(log_ret/log_ret.shift(1))
log_ret


# In[66]:


np.random.seed(42)
num_ports = 1000
all_weights = np.zeros((num_ports, len(acoes_port.columns[1:])))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    # Weights
    weights = np.array(np.random.random(5))
    weights = weights/np.sum(weights)
    
    # Save weights
    all_weights[x,:] = weights
    
    # Expected return
    ret_arr[x] = np.sum((log_ret.mean() * weights))
    
    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov(), weights)))
    
    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]


# In[67]:


print("Max Sharpe Ratio: {}". format(sharpe_arr.max()))
print("Local do Max Sharpe Ratio: {}". format(sharpe_arr.argmax()))


# In[68]:


# Pesos do Portfólio do Max Sharpe Ratio
print(all_weights[643,:])


# In[69]:


# salvando os dados do Max Sharpe Ratio
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
print(max_sr_ret)
print(max_sr_vol)


# In[72]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatilidade')
plt.ylabel('Retorno')
plt.scatter(max_sr_vol, max_sr_ret,c='black', s=200) # black dot
plt.show()


# Nós podemos ver no gráfico assima o conjunto de portfólios simulados, pois o peso $w_i$ de cada ativo foi simulado e criamos um conjunto de $n = 1000$ carteiras e escolhemos no ponto vermelho a que tem maior **Sharpe Ratio**, que é a razão retorno sobre a volatilidade. Esse dado nos da uma noção do portfólio ponderado pelo risco.

# In[73]:


def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov(), weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

def neg_sharpe(weights):
# the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1


# In[74]:


cons = ({'type': 'eq', 'fun': check_sum})
bounds = ((0,1), (0,1), (0,1), (0,1), (0,1))
init_guess = ((0.2),(0.2),(0.2),(0.2),(0.2))


# In[77]:


op_results = optimize.minimize(neg_sharpe, init_guess, method="SLSQP", bounds= bounds, constraints=cons)
print(op_results)


# In[85]:


frontier_y = np.linspace(-0.0006, 0.0008, 200)


# In[86]:


def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]


# In[87]:


frontier_x = []

for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = optimize.minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])


# In[89]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatilidade')
plt.ylabel('Retorno')
plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
plt.scatter(max_sr_vol, max_sr_ret,c='black', s=200)
# plt.savefig('cover.png')
plt.show()

