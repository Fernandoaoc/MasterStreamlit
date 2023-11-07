import statistics

import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
import seaborn as sns
from networkx.algorithms import community
from operator import itemgetter
import pandas_datareader as pdr
import itertools
import streamlit as st
import requests
from PIL import Image


def prepara_data(ano, quartil):
    data_trim = pd.read_csv(f'arquivosCSV\\inf_trimestral_fii_ativo_{ano}.csv', delimiter=';', encoding='ISO-8859-1')
    data_comp = pd.read_csv(f'arquivosCSV\\inf_mensal_fii_complemento_{ano}.csv', delimiter=';', encoding='ISO-8859-1')

    data_trim['Data_Referencia'] = pd.to_datetime(data_trim['Data_Referencia'])
    data_trim['Data_Referencia'] = data_trim['Data_Referencia'].dt.to_period('M')

    data_comp['Data_Referencia'] = pd.to_datetime(data_comp['Data_Referencia'])
    data_comp['Data_Referencia'] = data_comp['Data_Referencia'].dt.to_period('M')

    if quartil == 1:
        mensal1 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-3']
        mensal2 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-2']
        mensal3 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-1']
        mensal = pd.concat([mensal1, mensal2, mensal3], ignore_index=True)
        mensal.reset_index(inplace=True, drop=True)

        trimestre = data_trim.loc[data_trim['Data_Referencia'] == f'{ano}-3']
        trimestre.reset_index(inplace=True, drop=True)

    if quartil == 2:
        mensal1 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-6']
        mensal2 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-5']
        mensal3 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-4']
        mensal = pd.concat([mensal1, mensal2, mensal3], ignore_index=True)
        mensal.reset_index(inplace=True, drop=True)

        trimestre = data_trim.loc[data_trim['Data_Referencia'] == f'{ano}-6']
        trimestre.reset_index(inplace=True, drop=True)

    if quartil == 3:
        mensal1 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-9']
        mensal2 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-8']
        mensal3 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-7']
        mensal = pd.concat([mensal1, mensal2, mensal3], ignore_index=True)
        mensal.reset_index(inplace=True, drop=True)

        trimestre = data_trim.loc[data_trim['Data_Referencia'] == f'{ano}-9']
        trimestre.reset_index(inplace=True, drop=True)

    if quartil == 4:
        mensal1 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-12']
        mensal2 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-11']
        mensal3 = data_comp.loc[data_comp['Data_Referencia'] == f'{ano}-10']
        mensal = pd.concat([mensal1, mensal2, mensal3], ignore_index=True)
        mensal.reset_index(inplace=True, drop=True)

        trimestre = data_trim.loc[data_trim['Data_Referencia'] == f'{ano}-12']
        trimestre.reset_index(inplace=True, drop=True)

    if quartil == 5:
        for i in range(0, len(data_trim)):
            data_trim['CNPJ_Fundo'][i] = str(data_trim['CNPJ_Fundo'][i]).replace('.', '').replace('-', '').replace('/',
                                                                                                                   '')
            data_trim['CNPJ_Emissor'][i] = str(data_trim['CNPJ_Emissor'][i]).replace('.', '').replace('-', '').replace(
                '/', '')

        for i in range(0, len(data_comp)):
            data_comp['CNPJ_Fundo'][i] = str(data_comp['CNPJ_Fundo'][i]).replace('.', '').replace('-', '').replace('/',
                                                                                                                   '')

        data_merge = pd.merge(data_trim, data_comp, left_on=['CNPJ_Fundo'], right_on=['CNPJ_Fundo'],
                              suffixes=('_trim', '_mensal'))
        data_fii = data_merge.loc[data_merge['Tipo'] == 'FII']

        data_fii.reset_index(inplace=True, drop=True)
        return data_fii


    data_merge = pd.merge(trimestre, mensal, left_on=['CNPJ_Fundo'], right_on=['CNPJ_Fundo'],
                          suffixes=('_trim', '_mensal'))
    data_merge['P/VPA'] = data_merge['Patrimonio_Liquido'] / (
                data_merge['Cotas_Emitidas'] * data_merge['Valor_Patrimonial_Cotas'])

    for i in range(0, len(data_merge)):
        data_merge['CNPJ_Fundo'][i] = str(data_merge['CNPJ_Fundo'][i]).replace('.', '').replace('-', '').replace('/',
                                                                                                                 '')
        data_merge['CNPJ_Emissor'][i] = str(data_merge['CNPJ_Emissor'][i]).replace('.', '').replace('-', '').replace(
            '/', '')

    data_fii = data_merge.loc[data_merge['Tipo'] == 'FII']
    # data_fii = data_merge
    data_fii.reset_index(inplace=True, drop=True)

    if "não aplicavel" in data_fii['Emissor']:
        for i in range(len(data_fii['Emissor'])):
            if data_fii['Emissor'][i] == "não aplicavel":
                data_fii['Emissor'][i] = np.nan
            data_fii.dropna(subset=['Emissor'], inplace=True)
            data_fii.reset_index(inplace=True)

    return data_fii

# data = prepara_data(2017,4)
# data.to_csv('paladio.csv', index=False)


def colorir(var):
   tipo = ['Outras Cotas de FI', 'Cotas de Sociedades', 'Ações de Sociedades','FII', 'Outros Ativos Financeiros', 'LCI', 'Ações', 'FIA', 'LIG',
       'CRI', 'FIDC', 'FIP', 'CEPAC']
   cor = ['#d0572f','#d0802f','#d0a82f','#d0d02f','#a8d02f','#80d02f','#57d02f','#2fd02f','#2fd057','#2fd080','#2fd0a8','#2fd0d0','#2fa8d0']

   if var ==  tipo[0]:
      return cor[0]
   if var ==  tipo[1]:
      return cor[1]
   if var ==  tipo[2]:
      return cor[2]
   if var ==  tipo[3]:
      return cor[3]
   if var ==  tipo[4]:
      return cor[4]
   if var ==  tipo[5]:
      return cor[5]
   if var ==  tipo[6]:
      return cor[6]
   if var ==  tipo[7]:
      return cor[7]
   if var ==  tipo[8]:
      return cor[8]
   if var ==  tipo[9]:
      return cor[9]
   if var ==  tipo[10]:
      return cor[10]
   if var ==  tipo[11]:
      return cor[11]
   if var ==  tipo[12]:
      return cor[12]


def rede(df, nome, ano, mes):
    # Rede do ultimo trimestre de 2016
    from turtle import color

    net = Network(height='500px', width='100%', directed=True, notebook=True)

    G = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, source='CNPJ_Fundo', target='CNPJ_Emissor',
                                edge_attr=['Quantidade', 'Cotas_Emitidas'])
    valor_medio_dict = {}
    i = 0

    Cotas_Emitidas_dict = {}
    Tipo_dict = {}
    Patrimonio_Liquido_dict = {}
    Valor_Patrimonial_Cotas_dict = {}
    Percentual_Rentabilidade_Efetiva_Mes_dict = {}
    size_dict = {}
    color_dict = {}

    for pro in range(0, len(df)):

        Cotas_Emitidas_dict[df['CNPJ_Fundo'][pro]] = df['Cotas_Emitidas'][pro]
        Tipo_dict[df['CNPJ_Fundo'][pro]] = df['Tipo'][pro]
        Patrimonio_Liquido_dict[df['CNPJ_Fundo'][pro]] = df['Patrimonio_Liquido'][pro]
        Valor_Patrimonial_Cotas_dict[df['CNPJ_Fundo'][pro]] = df['Valor_Patrimonial_Cotas'][pro]
        Percentual_Rentabilidade_Efetiva_Mes_dict[df['CNPJ_Fundo'][pro]] = df['Percentual_Rentabilidade_Efetiva_Mes'][
            pro]
        color_dict[df['CNPJ_Fundo'][pro]] = colorir(df['Tipo'][pro])

        if df['Valor_Patrimonial_Cotas'][pro] >= 0:
            size_dict[df['CNPJ_Fundo'][pro]] = abs(np.log(df['Valor_Patrimonial_Cotas'][pro]) * 8)
        else:
            size_dict[df['CNPJ_Fundo'][pro]] = 5


    nx.set_node_attributes(G, Cotas_Emitidas_dict, 'Cotas_Emitidas')
    nx.set_node_attributes(G, Patrimonio_Liquido_dict, 'Patrimonio_Liquido')
    nx.set_node_attributes(G, Valor_Patrimonial_Cotas_dict, 'Valor_Patrimonial_Cotas')
    nx.set_node_attributes(G, Tipo_dict, 'Tipo')
    nx.set_node_attributes(G, Percentual_Rentabilidade_Efetiva_Mes_dict, 'Percentual_Rentabilidade_Efetiva_Mes')
    nx.set_node_attributes(G, size_dict, 'size')
    nx.set_node_attributes(G, color_dict, 'color')

    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')

    pr_dict = nx.pagerank(G)
    betweenness_dict = nx.betweenness_centrality(G)
    eigenvector_dict = nx.eigenvector_centrality(G)

    nx.set_node_attributes(G, pr_dict, 'pagerank')
    nx.set_node_attributes(G, betweenness_dict, 'betweenness')
    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')

    communities = community.greedy_modularity_communities(G)
    modularity_dict = {}  # Create a blank dictionary
    for i, c in enumerate(
            communities):  # Loop through the list of communities, keeping track of the number for the community
        for name in c:  # Loop through each person in a community
            modularity_dict[
                name] = i  # Create an entry in the dictionary for the person, where the value is which group they belong to.

    # Now you can add modularity information like we did the other metrics
    nx.set_node_attributes(G, modularity_dict, 'modularity')

    # First get a list of just the nodes in that class
    class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]

    # Then create a dictionary of the eigenvector centralities of those nodes
    class0_eigenvector = {n: G.nodes[n]['eigenvector'] for n in class0}

    # Then sort that dictionary and print the first 5 results
    class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)

    return G, net


def imprime_rede(G, net,ano,trimestre):
    net.from_nx(G)
    #net.show_buttons()
    return net.show(f'Rede_{ano}-{trimestre}.html')


def figura_componentes(G,ano,mes):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(18, 18))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title(f"Componentes Conectados da rede {ano} - {mes}", fontsize=24)
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot", fontsize=18)
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram", fontsize=18)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    fig.patch.set_facecolor('white')
    #plt.savefig(f'images\\componentes_conectados_{ano}-{mes}.png', dpi='figure')
    st.pyplot(fig)


def exibir_metricas(G):
    #st.subheader("Métricas da Rede Complexa")
    st.write(f"Número de nós: {len(G.nodes)}")
    st.write(f"Número de arestas: {len(G.edges)}")
    dict_degree = nx.degree(G)
    st.write(f"Degree : {statistics.mean([dict_degree.values])}")
    #st.write(f"Closeness centrality: {nx.closeness_centrality(G)}")
    #st.write(f"Betweenness centrality: {nx.betweenness_centrality(G)}")


def correlatio_grafic(data, ano, mes):
    correlation = data.corr()
    fig = plt.figure(figsize=(10, 6))
    plt.title(f'Matriz Correlação {ano}')
    sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
    plt.savefig(f'images\\graficos\\Matriz Correlação {ano}_{mes}.jpg', dpi='figure')
    image = Image.open(f'images\\graficos\\Matriz Correlação {ano}_{mes}.jpg')

    st.image(image)

    # st.pyplot(fig)

def graficos(data, ano, mes, G):

    correlation = data.corr()
    fig = plt.figure(figsize=(10, 6))
    plt.title(f'Matriz Correlação {ano}')
    sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
    #plt.savefig(f'graficos\\Matriz Correlação {ano}_{mes}.png', dpi='figure')
    st.pyplot(fig)

    A = nx.to_pandas_edgelist(G, source='CNPJ_Fundo', target='CNPJ_Emissor')
    corrA = A.corr()

    f, ax = plt.subplots(figsize=(10, 8))
    plt.title(f'Matriz Correlação Gerada depois da Rede {ano}')
    sns.heatmap(corrA, annot=True, fmt=".1f", linewidths=.5)
    # plt.savefig(f'graficos\\Matriz Correlação Gerada depois da Rede {ano}_{mes}.png', dpi='figure')


def pagerank(G, ano):
    # # Calcula o pagerank de cada nó
    # # Cria a rede Karate Club
    # #G = nx.karate_club_graph()
    # bc = nx.betweenness_centrality(G)
    #
    # # Ordenando os nós pelo valor de betweenness centrality
    # sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    #
    # # Exibindo os 5 nós mais espalhados
    # print("Os 5 nós mais espalhados:")
    # for node in sorted_nodes[:5]:
    #     print(f"Node {node[0]}: {node[1]:.4f}")
    #
    # # Plotando a rede com os nós coloridos pelo valor de betweenness centrality
    # node_color = [bc[node] for node in G.nodes()]
    # nx.draw(G, pos=nx.spring_layout(G), node_color=node_color, cmap=plt.cm.Blues)
    #
    # st.pyplot()
    pr = nx.pagerank(G)
    pr_sorted = {}
    for i in sorted(pr, key=pr.get, reverse=True):
        pr_sorted[i] = pr[i]

    top_10 = {k: pr_sorted[k] for k in list(pr_sorted)[:10]}
    x = list(top_10.keys())
    y = list(top_10.values())

    area = []
    for s in top_10.values():
        if s != 0:
            area.append(s * 1e4)
        else:
            area.append(s + 100)

    nome = []
    trimestre = []
    for k in top_10:
        if df_fii.loc[df_fii['CNPJ_fundo'] == k].empty == False:
            b = df_fii.loc[df_fii['CNPJ_fundo'] == k]
            b.reset_index(inplace=True, drop=True)
            nome.append(b['CODIGO'][0])
        else:
            cnpj = f'{k[:2]}.{k[2:5]}.{k[5:8]}/{k[8:12]}-{k[12:]}'  # 11.827.568/0001-05
            nome.append(f'{cnpj}')

    for j in range(len(x)):
        trimestre.append(f'{ano}-{mes}')

    # f, ax1 = plt.subplots(figsize=(16,8))
    # fig = plt.figure(figsize=(16,9))
    # fig.tight_layout()
    # fig.patch.set_facecolor('white')
    import seaborn as sns
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))
    # ax.scatter(x, y, c=np.random.rand(10), s=area, alpha=0.5)

    for i, txt in enumerate(nome):
        plt.annotate(txt, (x[i], y[i]))

    df = pd.DataFrame({'Nome': nome, 'CNPJ': x, 'metrica': y, 'size': area, 'Data': trimestre})

    plt.scatter(x, y, c=np.random.rand(10), s=area)
    plt.title(f'{metrica} - Top 10 - {ano}_{mes}', fontsize=24)
    plt.xlabel('CNPJ Fundo')
    plt.ylabel(f'{metrica}')
    plt.xticks(rotation=45)
    st.pyplot(fig)


import requests
import pandas as pd


def lista_fundos():
    #A url que você quer acesssar
    url = "https://clubedospoupadores.com/fundos-imobiliarios/cnpj.html"
    #Informações para fingir ser um navegador
    header = {
      "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"
    }
    #juntamos tudo com a requests
    r = requests.get(url, headers=header)
    #E finalmente usamos a função read_html do pandas
    lista_fii = pd.read_html(r.text)

    df_fii = lista_fii[0]
    df_fii.rename(columns={'ABCP11':'CODIGO',
                  'FUNDO DE INVESTIMENTO IMOBILIÁRIO GRAND PLAZA SHOPPING':'NOME',
                  '1.201.140.000.190':'CNPJ_fundo', 'Rio Bravo Investimentos DTVM Ltda':'ADMINISTRADOR',
                  '72.600.026.000.181':'CNPJ_admin'}, inplace=True)
    df_fii.append({'CODIGO':'ABCP11', 'NOME':'FUNDO DE INVESTIMENTO IMOBILIÁRIO GRAND PLAZA SHOPPING', 'CNPJ_fundo':'1.201.140.000.190', 'ADMINISTRADOR':'Rio Bravo Investimentos DTVM Ltda', 'CNPJ_admin':'72.600.026.000.181'}, ignore_index=True)

    #'CODIGO', 'NOME', 'CNPJ_fundo', 'ADMINISTRADOR', 'CNPJ_admin'
    #['ABCP11', 'FUNDO DE INVESTIMENTO IMOBILIÁRIO GRAND PLAZA SHOPPING','1.201.140.000.190', 'Rio Bravo Investimentos DTVM Ltda','72.600.026.000.181']
    return df_fii


def graficosRedes(G, df, ano, mes, metrica):
    if metrica == 'Degree_Centrality':
        pr = nx.degree_centrality(G)
        pr_sorted = {}

    if metrica == 'Betweenness_Centrality':
        pr = nx.betweenness_centrality(G)
        pr_sorted = {}

    if metrica == 'Closeness_Centrality':
        pr = nx.closeness_centrality(G)
        pr_sorted = {}

    if metrica == 'Eigenvector_Centrality':
        pr = nx.eigenvector_centrality(G)
        pr_sorted = {}

    if metrica == 'PageRank':
        pr = nx.pagerank(G)
        pr_sorted = {}

    for i in sorted(pr, key=pr.get, reverse=True):
        pr_sorted[i] = pr[i]

    top_10 = dict(itertools.islice(pr_sorted.items(), 10))
    x = list(top_10.keys())
    y = list(top_10.values())

    area = []
    for s in top_10.values():
        if s != 0:
            area.append(s * 1e4)
        else:
            area.append(s + 100)

    nome = []
    trimestre = []
    for cnpj in top_10:
        cnpj_str = str(cnpj)
        cnpj_formatado = "{}.{}.{}.{}.{}".format(cnpj_str[:2], cnpj_str[2:5], cnpj_str[5:8], cnpj_str[8:11],
                                                 cnpj_str[11:])

        if df.loc[df['CNPJ_fundo'] == cnpj_formatado].empty == False:
            b = df.loc[df['CNPJ_fundo'] == cnpj_formatado]
            b.reset_index(inplace=True, drop=True)
            nome.append(b['CODIGO'][0])
        else:
            nome.append(f'NoCode - {cnpj_formatado}')

    for j in range(len(x)):
        trimestre.append(f'{ano}-{mes}')

    for i, txt in enumerate(nome):
        plt.annotate(txt, (x[i], y[i]))

    df_para_grafico = pd.DataFrame({'Nome': nome, 'CNPJ': x, 'metrica': y, 'size': area, 'Data': trimestre})

    return df_para_grafico



def gera_df_para_rede(df_para_grafico, metrica):
    year = [2016, 2017, 2018, 2019, 2020, 2021]
    #medidas = ['Degree_Centrality', 'PageRank', 'Betweenness_Centrality', 'Closeness_Centrality','Eigenvector_Centrality']
    quartil = [1,2,3,4]
    apendDF = []
    for ano in year:
        if ano == 2016:
            data = prepara_data(f'{ano}', 4)
            G, n = rede(data,'Rede', f'{ano}', f'{4}')
            #n.from_nx(G)
            #n.show_buttons()
            #n.show(f'rede_html\\Rede_{ano}-4.html')
            df = graficosRedes(ano, 4, metrica)
            apendDF.append(df)
        else:
            for mes in quartil:
                data = prepara_data(f'{ano}', mes)
                G, n = rede(data,'Rede', f'{ano}', f'{mes}')
                #n.from_nx(G)
                #n.show_buttons()
                #n.show(f'rede_html\\Rede_{ano}-{mes}.html')
                df = graficosRedes(ano, mes, metrica)
                apendDF.append(df)
    concat_degree = pd.concat(apendDF)

    return concat_degree



def grafico_tempo_metrica(df,ano,mes, metrica):

    fig = px.scatter(df, title=f"{metrica} {ano} - {mes}", x=df.index,
                     y="metrica", animation_frame='Data', animation_group='Nome',
                     size='size', hover_name="Nome", text=df['Nome'], height=600, width=900, color='Nome',
                     log_x=False, size_max=55, range_y=[0., 0.70])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons
    fig.update_layout(legend_title=metrica)

    #fig.layout.updatemenus[0].buttons[0].args[1]['frame']['redraw'] = True

    fig.write_html(f'images\\10_maiores_{metrica}_Centrality_scatter_{ano}_{mes}.html')
    #fig.show()
    return fig


def get_subgraph_for_node(rede, cnpj):
    vizinhos = list(rede.neighbors(cnpj))

    # Criar um subgrafo contendo apenas os vizinhos do nó específico
    subgrafo = G.subgraph(vizinhos + [cnpj])

    # Desenhar o grafo
    pos = nx.spring_layout(subgrafo)  # Layout para posicionar os nós
    nx.draw(subgrafo, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=1000)

    # Mostrar o grafo
    plt.show()