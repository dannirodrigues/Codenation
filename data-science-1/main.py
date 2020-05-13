#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:




from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[4]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[9]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# In[11]:


dataframe.describe()


# In[12]:


df = dataframe.describe()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[32]:


def q1():
    df = dataframe.describe()
    data = df.loc[['25%','50%','75%']]
    da = round(data['normal']-data['binomial'],3)
    return  tuple(da.values)
q1()


# In[33]:



sns.distplot(dataframe['normal'])


# In[34]:



sns.distplot(dataframe['binomial'])


# Para refletir:
# 
# * Você esperava valores dessa magnitude? Sim, pelo fato da amostra ser grande a distribuição tende a ser normal, logo faz sentidos a magnitude teder a zero
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores? Em uma distribuição binomial, para um determinado valor fixo de probabilidade, à medida que o tamanho da amostra aumenta, mais esta distribuição tende a ter o formato de uma distribuição normal. Desta maneira, é possível se obter valores de probabilidades para uma distribuição binomial a partir de uma distribuição normal.

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[41]:


# fit a cdf
np.array(dataframe['normal'])


# In[39]:


def q2():
    
    normal = np.array(dataframe['normal'])
    return float(round((sct.norm.cdf(normal.mean()+normal.std(), loc=20, scale=4)-
           sct.norm.cdf(normal.mean()-normal.std(), loc=20, scale=4)),3))


# In[40]:





# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[59]:


def q3():
    # Média das variáveis normal e binomial:
    m_norm,m_binom = dataframe.mean()

    # Variância das variáveis normal e binomial:
    v_norm,v_binom = dataframe.var()

    # Diferença entre as médias e as variâncias das variáveis binomial e normal:
    return tuple([round(x,3) for x in (m_binom - m_norm,v_binom - v_norm)])
q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude? 
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[47]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[48]:


# Sua análise da parte 2 começa aqui.


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[55]:



def q4():
    aux = pd.DataFrame(stars['mean_profile'].loc[stars['target']==False])
    aux['false_pulsar_mean_profile_standardized'] = (aux-aux.mean())/(aux.std())
    q1 = sct.norm.ppf(0.80, loc=0, scale=1)
    q2 = sct.norm.ppf(0.90, loc=0, scale=1)
    q3 = sct.norm.ppf(0.95, loc=0, scale=1)
    quartis = [q1,q2,q3]
    ecdf = ECDF(aux['false_pulsar_mean_profile_standardized'])
    return (tuple(np.round(ecdf(quartis),3)))


# In[57]:





# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[53]:


def q5():
    aux = pd.DataFrame(stars['mean_profile'].loc[stars['target']==False])
    aux['false_pulsar_mean_profile_standardized'] = (aux-aux.mean())/(aux.std())
    
    q1 = round(np.percentile(np.array(aux['false_pulsar_mean_profile_standardized']),25)-              
               (sct.norm.ppf(0.25, loc=0, scale=1)),3)
        
    q2 = round(np.percentile(np.array(aux['false_pulsar_mean_profile_standardized']),50)-              
               (sct.norm.ppf(0.5, loc=0, scale=1)),3)
   
    q3 = round(np.percentile(np.array(aux['false_pulsar_mean_profile_standardized']),75)-              
               (sct.norm.ppf(0.75, loc=0, scale=1)),3)
    return (q1,q2,q3)


# In[ ]:





# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
