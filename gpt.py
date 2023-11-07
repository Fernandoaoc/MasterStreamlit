import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pyvis.network as net
from pyvis.options import Options
import plotly.express as px
import numpy as np
import funcoes as fn
import streamlit.components.v1 as components

# Configurações do Streamlit
st.set_page_config(
    page_title="FII's",
    layout="wide",
    page_icon=':money:'
)
st.title("Análise de FII Brasileiros")
# Função para exibir a rede complexa
def exibir_rede():
    """st.subheader("Visualização da Rede de FII")
    nt = net.Network(notebook=True, height="600px", width="100%", bgcolor="#222222")
    nt.from_nx(G)
    nt.show("network.html")"""
    st.subheader("Visualização da Rede de FII")
    nt = net.Network(notebook=True, height="100%", width="100%")
    nt.from_nx(G)
    #nt.show(f'Rede_{ano}-{trimestre}.html')
    html_file = open(f'Rede_{ano}-{trimestre}.html')
    source_code = html_file.read()
    components.html(source_code)

# Função para exibir as métricas da rede
def exibir_metricas():
    st.subheader("Métricas da Rede Complexa")
    st.write(f"Número de nós: {len(G.nodes)}")
    st.write(f"Número de arestas: {len(G.edges)}")
    st.write(f"Coeficiente de agrupamento médio: {nx.average_clustering(G)}")
    st.write(f"Grau médio: {nx.average_degree_connectivity(G)}")
    st.write(f"Diâmetro: {nx.diameter(G)}")

# Função para exibir o mapa de localização
def exibir_mapa():
    st.subheader("Mapa com Localizações de Imóveis")
    # Carregar dados de localizações de imóveis (pode ser substituído pelos seus próprios dados)
    #data = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
    data = pd.read_csv("arquivosCSV/imoveis_2016.csv")
    # Exibir mapa utilizando a biblioteca Streamlit
    #st.map(data)

    #px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(data,
                            lat="latitude",
                            lon="longitude",
                            hover_name="nome_imovel", hover_data=["nome_fundo","cnpj_fundo"],
                            color="nome_fundo",
                            zoom=3)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.layout.update(showlegend=False)
    st.plotly_chart(fig)

# Título do dashboard
st.title("Dashboard de Rede Complexa")

# Carregar dados de rede (pode ser substituído pelos seus próprios dados)
#G = nx.karate_club_graph()
ano=2016
trimestre=4
data = fn.prepara_data(ano=ano, quartil=trimestre)
G, neti = fn.rede(data, 'rede', ano, trimestre)

# Menu na parte esquerda da página
menu_opcoes = ["Rede", "Métricas da Rede", "Mapa de Localização"]
opcao_selecionada = st.sidebar.selectbox("Selecione uma opção", menu_opcoes)

# Exibir conteúdo correspondente à opção selecionada na parte central da página




if opcao_selecionada == "Rede":
    exibir_rede()
elif opcao_selecionada == "Métricas da Rede":
    exibir_metricas()
elif opcao_selecionada == "Mapa de Localização":
    exibir_mapa()

