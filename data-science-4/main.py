#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")

for col in countries.columns:
    try:
        countries[col] = countries[col].str.replace(',','.').astype(float)
    except Exception as e:
        print(e)


# In[ ]:





# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[5]:


def q1():
    regioes = countries.Region.unique()
    regioes.sort()

    return list(regioes)
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[36]:


def q2():
    # Retorne aqui o resultado da questão 2.
    from sklearn.preprocessing import KBinsDiscretizer
    binarizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    try:
        countries['Pop_density'] = countries['Pop_density'].str.replace(',','.').astype(float)
    except:
        pass
    pop_density_array = np.asarray(countries['Pop_density']).reshape(-1,1)
    pop_density_array.shape
    discretized_data = binarizer.fit_transform(pop_density_array)
    df = pd.DataFrame(discretized_data.tolist(), columns=['bins'])
    return int(df[df['bins']==9.0].count()[0])
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[26]:



def q3():
    dummies = pd.get_dummies(countries[['Region', 'Climate']].fillna('NaN'))
    
    return int(dummies.shape[1])
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[20]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[21]:


def q4():
    # Retorne aqui o resultado da questão 4.
    # Questao 4
    # https://www.youtube.com/watch?v=irHhDMbw3xo
    from sklearn.impute import SimpleImputer

    # imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imput_cols = countries.columns[countries.dtypes != 'object'].tolist()
    remaining_cols = countries.columns[countries.dtypes == 'object'].tolist()

    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.compose import ColumnTransformer, make_column_transformer
    from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #resp = scaler.fit_transform(countries[imput_cols])

    imputer = make_column_transformer(
        (make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler()), imput_cols),
        remainder='passthrough'        
    )

    #pipe = make_pipeline(imputer)
    #resp = imputer.fit_transform(countries)
    #pd.DataFrame(resp)
  
    tc = pd.DataFrame(test_country).T
    tc.columns = countries.columns
    for col in tc.columns:
        try:
            tc[col] = tc[col].astype(float)
        except Exception as e:
            print(e)

    imputer.fit(countries)

    resp_df = pd.DataFrame(imputer.transform(tc))
    resp_df.columns = [*imput_cols, *remaining_cols]
    return float(round(resp_df.loc[0, 'Arable'],3))


# In[ ]:





# In[ ]:





# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[22]:


def q5():
    # Retorne aqui o resultado da questão 4.
    desc = countries['Net_migration'].describe()
    iqr = desc['75%']-desc['25%']
    q1 = desc['25%']
    q3 = desc['75%']
    outmin = q1-1.5*iqr
    outmax = q3+1.5*iqr
    return int((countries['Net_migration']<outmin).sum()), int((countries['Net_migration']>outmax).sum()), bool(0)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[27]:


from sklearn.datasets import fetch_20newsgroups


# In[28]:


categorias = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup= fetch_20newsgroups(subset='train', categories=categorias, shuffle=True, random_state=12)


# In[29]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[31]:


def q6():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(newsgroup.data)
    phone_index = vectorizer.vocabulary_.get('phone')
    return int(X.toarray()[:, phone_index].sum()) 
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[24]:


def q7():
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfid_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfid_vectorizer.fit_transform(newsgroup.data)
    phone_index = tfid_vectorizer.vocabulary_.get('phone')
    df = pd.DataFrame(tfidf_matrix.toarray(), columns = tfid_vectorizer.get_feature_names())
    return float(round(df['phone'].sum(),3))
q7()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




