import pandas as pd 

import matplotlib.pyplot as plt 
import streamlit as st
import altair as alt 


def main():
    st.sidebar.image('ufsc.png', width=200)
    st.title('Coronavírus Contexto Santa Catarina')
   

    st.sidebar.markdown('Os dados atualizados você pode encontrar no link abaixo, baixar o csv  e fazer upload ao lado:')
    st.sidebar.markdown('https://brasil.io/dataset/covid19/caso/?search=&date=&state=SC&city=&place_type=&is_last=&city_ibge_code=&order_for_place=')
   
    
    file = st.file_uploader('ESCOLHA A BASE DE DADOS QUE DESEJA ANALISAR (.csv)', type = 'csv')
    if file is not None:
        st.subheader('Coronavirus Santa Catarina')
        df = pd.read_csv(file)
        aux = pd.DataFrame({'colunas': df.columns, 'tipos': df.dtypes})
        colunas_numericas = list(aux[aux['tipos']!= 'object']['colunas'])
        colunas = list(df.columns)
        
        df = df.dropna()
        d = df[df['state']== 'SC']
        st.dataframe(d)

        st.subheader('Base de Dados Tratadas')
        st.markdown('** Números de linhas : **')
        st.markdown(d.shape[0])
        st.markdown('** Números de Colunas : **')
        st.markdown(d.shape[1])


        
      


        select_analise = st.radio('ESCOLHA UMA OPÇÃO ABAIXO:', ('Estado de Santa Catarina', 'Cidades com Campus UFSC'))
        if select_analise == 'Estado de Santa Catarina':
            st.dataframe(d)
            st.markdown('**Análise Gráfica**')
            options = st.multiselect('ESCOLHA AS OPÇÕES QUE DESEJA VISUALIZAR NO GRÁFICO',(df.columns))
            if len(options) >=1:

                if st.checkbox('Grafico Linear'): 
                    sc = d[options]
                    tem = d.groupby('date')[options].sum()
                    tem.plot(kind='line',figsize = (15,10)).set_yscale('linear')
                    plt.grid(True)
                    plt.legend(loc=2, prop={'size': 18})
                    plt.title('Situação Santa Cantarina Escala Logarítimica',fontsize=24)
                    st.pyplot()
                    
                elif st.checkbox('Grafíco Logarítimico'):
                    sc = d[options]
                    tem = d.groupby('date')[options].sum()
                    tem.plot(kind='line',figsize = (15,10)).set_yscale('log')
                    plt.legend(loc=2, prop={'size': 18})
                    plt.grid(True)
                    plt.title('Situação Santa Cantarina Escala Logarítimica',fontsize=24)
                    st.pyplot()
                elif st.checkbox('Gráfico Linear x Gráfico Linear'):
                    tem = d.groupby('date')[options].sum()
                    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
                    axes[0].plot(tem)
                    axes[0].set_yscale('linear')
                    axes[0].set_title('Escala Linear')
                    axes[0].grid(True)
                    
                    axes[1].plot(tem)
                    axes[1].set_yscale('log')
                    axes[1].set_title('Escala Logarítimica')
                    axes[1].grid(True)
                   
                    fig.tight_layout()
                    st.pyplot()
                        
        



      
if __name__=='__main__':
    main()