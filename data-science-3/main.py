#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# In[20]:


#pip install loguru


# ## _Setup_ geral

# In[70]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from loguru import logger


# In[73]:


# Algumas configurações para o matplotlib.


from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[23]:


fifa = pd.read_csv("data.csv")


# In[24]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# In[25]:


columns_to_drop


# In[26]:


fifa.head()


# ## Inicia sua análise a partir daqui

# In[27]:


# Sua análise começa aqui.
cons = pd.DataFrame({'colunas': fifa.columns,
                    'tipo': fifa.dtypes,
                    'missing': fifa.isna().sum(),
                    'size':fifa.shape[0],
                    'unicos':fifa.nunique()})

cons['percentual'] = round(cons['missing']/cons['size'],2)


# In[ ]:





# In[29]:


#Removendo os dados faltante 

print('Contagem de colunas com ATÉ 20% de dados faltantes', cons[cons.percentual < 0.2].shape[0])
print('Contagem de colunas com 0% de dados faltantes',  cons[cons.percentual == 0].shape[0])


# In[31]:


fifa.dropna(inplace=True)


# In[35]:


pca = PCA(n_components=1)


# In[37]:


pca.fit(fifa).transform(fifa)


# In[38]:


pca.explained_variance_ratio_


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[45]:


def q1():
    d = pca.explained_variance_ratio_
    return round(d[0],3)
q1() 


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[50]:





# In[74]:


def q2():
    clf = PCA(0.95)
    x_transform = clf.fit(fifa)
    d = x_transform.explained_variance_ratio_
    return round(len(d),3)
   
q2()    
    


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[65]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[66]:





# In[69]:


def q3():
    pca = PCA().fit(fifa)
    components = pca.components_
    coordenadas = components.dot(x)
    return (round(coordenadas[0],3),round(coordenadas[1],3))
q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[72]:


def q4():
    rfe = RFE(LinearRegression(), n_features_to_select = 5, step = 1)
    y_train = fifa['Overall']
    x_train = fifa.drop(columns=['Overall'])
    rfe.fit(x_train, y_train)
    return list(x_train.columns[rfe.get_support()])
q4()   


# In[ ]:




