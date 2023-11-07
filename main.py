import funcoes as fn
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px
import networkx as nx
from sklearn.preprocessing import StandardScaler


def load_data_mapa(tipo_dado, ano, trimestre=1):
    if tipo_dado == 'data':
        data = pd.read_csv(f"arquivosCSV/mapas/info_imoveis_{ano}.csv")
        return data
    if tipo_dado == 'info':
        info = pd.read_csv(f'arquivosCSV\\inf_trimestral_fii_ativo_{ano}.csv', delimiter=';', encoding='ISO-8859-1')
        return info
    if tipo_dado == 'prepara_data':
        prepara_data = fn.prepara_data(ano=ano, quartil=trimestre)

        return prepara_data


# Leitura dos dados
def exibir_mapa(ano, select, legenda, tipo_dado='data', nome=''):
    # st.subheader("Mapa com Localizações de Imóveis")
    scaler = StandardScaler()

    if tipo_dado == 'single':
        df = load_data_mapa(ano=ano, tipo_dado='data')
        data = df.loc[df['nome_fundo'] == nome]
        zoom = 5
        select = "segmento_atuacao"
    else:
        data = load_data_mapa(tipo_dado, ano)
        df_max_scaled = data
        zoom = 3
    fig = px.scatter_mapbox(data,
                            lat="latitude",
                            lon="longitude",
                            hover_name="nome_imovel",
                            hover_data=["nome_fundo", "cnpj_fundo", "quantidade_cotas_emitidas", "nome_administradores",
                                        "segmento_atuacao"],
                            color=select,
                            zoom=zoom)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.layout.update(showlegend=legenda)
    return fig


def info_imoveis(ano):
    data = load_data_mapa('data', ano)
    total_imoveis = len(data['Endereco'])


st.set_page_config(page_title='Análise de FIIs Brasileiros', layout='wide', page_icon=':money:')
# st.title("FII'S")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title("FII's - Fundos de Investimento Imobiliário")
with row0_2:
    st.text("")
    st.text("")
    st.text("")
    st.markdown('Dashboard de visualização de fundos de investimento imobiliários para fins acadêmicos')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown(
        "Os Fundos de investimento imobiliários chamam a atenção de investidores, também chamam a atenção de pesquisadores. Diante deste aumento significativo de investidores e estudos, no presente trabalho serão analisados os fundos de investimento listados na B3 a partir do 4º trimestre de 2016.")

with st.sidebar:
    st.title("Período da Rede")
    st.text("")
    st.markdown('Selecione o trimestre e o ano desejado para montar a rede e exibir os gráficos ao lado')
    # trimestre = st.select_slider('Trimestre:', [1, 2, 3, 4], value=4)
    ano = st.select_slider('Ano:', [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    match ano:
        case 2016:
            trimestre = 4
        case 2022:
            trimestre = st.select_slider('Trimestre:', [1, 2], value=1)
        case 2023:
            trimestre = st.select_slider('Trimestre:', [1, 2, 3], value=1)
        case _:
            trimestre = st.select_slider('Trimestre:', [1, 2, 3, 4], value=1)

    data = load_data_mapa('prepara_data', ano, trimestre)
    # G, net = fn.rede(data, 'rede', ano, trimestre)
    # fn.imprime_rede(G, net, ano, trimestre)
    # html_file = open(f'Rede_{ano}-{trimestre}.html')

# st.table(data)

# st.subheader("Rede")
row1_spacer1, row1_1, row1_spacer2 = st.columns((.2, 7.1, .2))
with row1_1:
    st.subheader("Rede")
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns((.2, 1.3, .4, 5.4, .2))
with row2_1:
    st.markdown(f'A rede de fundos para o {trimestre}º trimestre do ano de {ano} é visualizada ao lado')
    # if st.button("Rede Info"):
    #     G, net = fn.rede(data, 'rede', ano, trimestre)
    #     fn.exibir_metricas(G)

with row2_2:
    G, net = fn.rede(data, 'rede', ano, trimestre)
    fig = fn.imprime_rede(G, net, ano, trimestre)
    html_file = open(f'Rede_{ano}-{trimestre}.html')
    source_code = html_file.read()
    components.html(source_code, height=520, width=700)

    # if st.button("Rede"):
    # G, net = fn.rede(data, 'rede', ano, trimestre)
    # fig = fn.imprime_rede(G, net, ano, trimestre)
    # html_file = open(f'Rede_{ano}-{trimestre}.html')
    # source_code = html_file.read()
    # components.html(source_code, height=520, width=700)

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.subheader("Gráficos:")
row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3 = st.columns((.2, 1.3, .4, 5.4, .2))
# with st.sidebar:
# st.title("Opções de Gráficos")
opcoes = ['Correlação', 'Componentes', 'Pagerank']
texto = 'Qual o gráfico você deseja visualizar?'
with row4_1:
    # st.markdown("Selecione no menu lateral, qual o gráfico você deseja visualizar.")
    st.markdown(f'<div style="text-align: justify;"> {texto}</div>', unsafe_allow_html=True)
    selecao = st.selectbox('Gráfico:', opcoes)

    if selecao == 'Correlação':
        # st.markdown('Representação gráfica de valores simultâneos de duas variáveis relacionadas a um mesmo processo, mostrando o que acontece com uma variável quando a outra se altera')
        texto = 'Representação gráfica de valores simultâneos de duas variáveis relacionadas a um mesmo processo, mostrando o que acontece com uma variável quando a outra se altera'
    if selecao == 'Componentes':
        # st.markdown('Visualização da componente principal, histograma de grau e a distribuição de grau por nó.')
        texto = 'Visualização da componente principal, histograma de grau e a distribuição de grau por nó.'
    if selecao == 'Pagerank':
        # st.markdown('Pagerank calcula uma classificação dos nós na rede com base na estrutura dos links de entrada.')
        texto = 'Pagerank calcula uma classificação dos nós na rede com base na estrutura dos links de entrada.'
    st.markdown(f'<div style="text-align: justify;"> {texto}</div>', unsafe_allow_html=True)
with row4_2:
    # G, net = fn.rede(data, 'rede', ano, trimestre)
    if selecao == 'Correlação':
        fn.correlatio_grafic(data, ano, trimestre)
    if selecao == 'Componentes':
        fn.figura_componentes(G, ano, trimestre)
    if selecao == 'Pagerank':
        fn.pagerank(G, ano)

# with st.sidebar:
#     st.title("Opções de Mapas")
#     selecao = st.radio(
#         "Filtro do mapa de imóveis",
#         ('Nome do Fundo', 'Segmento de atuação', 'Administradoras de Fundos'))
#     if selecao == 'Nome do Fundo':
#         color = "nome_fundo"
#         legenda = False
#     if selecao == 'Segmento de atuação':
#         color = "segmento_atuacao"
#         legenda = True
#     if selecao == 'Administradoras de Fundos':
#         color = "nome_administradores"
#         legenda = True

row5_spacer1, row5_1, row5_spacer2 = st.columns((.2, 7.1, .2))
with row5_1:
    st.subheader("Mapa Imóveis")
row6_spacer1, row6_1, row6_spacer2, row6_2, row6_spacer3 = st.columns((.2, 1.3, .4, 5.4, .2))
with row6_1:
    st.markdown(
        "Os imóveis podem ser agrupados pelo seu segmento de atuação, o nome do fundo, bem como seu administrador.")
    selecao_imoveis = st.radio(
        "Filtro de imóveis",
        ('Nome do Fundo', 'Segmento de atuação', 'Administradoras de Fundos'))
    if selecao_imoveis == 'Nome do Fundo':
        color = "nome_fundo"
        legenda = False
    if selecao_imoveis == 'Segmento de atuação':
        color = "segmento_atuacao"
        legenda = True
    if selecao_imoveis == 'Administradoras de Fundos':
        color = "nome_administradores"
        legenda = True
with row6_2:
    # if st.sidebar.button('Mostrar mapa'):
    fis = exibir_mapa(ano=ano, select=color, legenda=legenda)
    st.plotly_chart(fis)

# with st.sidebar:
# node_info = net.nodes[option]
info = load_data_mapa('data', ano)
info.sort_values(by='nome_fundo', na_position="last", ascending=True, inplace=True)
info.dropna(subset=['nome_fundo'], inplace=True)
nome = info['nome_fundo'].unique()
medidas = ['Degree_Centrality', 'PageRank', 'Betweenness_Centrality', 'Closeness_Centrality', 'Eigenvector_Centrality']

row7_spacer1, row7_1, row7_spacer2 = st.columns((.2, 7.1, .2))
with row7_1:
    st.subheader("Imóveis")
row8_spacer1, row8_1, row8_spacer2, row8_2, row8_spacer3 = st.columns((.1, 1.3, .4, 5.4, .2))
with row8_1:
    st.markdown("Localização dos Imóveis por Fundo")
    nome_select = st.selectbox("Selecione o FII", nome)
with row8_2:
    # if st.sidebar.button('Mostrar Mapa'):
    figura = exibir_mapa(ano=ano, select=color, legenda=legenda, tipo_dado='single', nome=nome_select)
    st.plotly_chart(figura)

row9_spacer1, row9_1, row9_spacer2 = st.columns((.2, 7.1, .2))
with row9_1:
    st.subheader("Metricas")
row10_spacer1, row10_1, row10_spacer2, row10_2, row10_spacer3 = st.columns((.1, 1.3, .4, 5.4, .2))
with row10_1:
    st.markdown("Localização dos Imóveis por Fundo")
    metrica_select = st.selectbox("Métrica", medidas)
with row10_2:
    # if st.sidebar.button('Mostrar Mapa'):
    lista_fundos = fn.lista_fundos()
    print(lista_fundos.head(3))
    year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    # quartil = [1, 2, 3, 4]
    # apendDF = []
    df = fn.graficosRedes(G, lista_fundos, ano, trimestre, metrica_select)
    # for ano in year:
    #     if ano == 2016:
    #         data = fn.prepara_data(f'{ano}', 4)
    #         G, n = fn.rede(data, 'Rede', f'{ano}', f'{4}')
    #         # n.from_nx(G)
    #         # n.show_buttons()
    #         # n.show(f'rede_html\\Rede_{ano}-4.html')
    #         df = fn.graficosRedes(G, lista_fundos, ano, 4, metrica_select)
    #         apendDF.append(df)
    #     else:
    #         for mes in quartil:
    #             data = fn.prepara_data(f'{ano}', mes)
    #             G, n = fn.rede(data, 'Rede', f'{ano}', f'{mes}')
    #             # n.from_nx(G)
    #             # n.show_buttons()
    #             # n.show(f'rede_html\\Rede_{ano}-{mes}.html')
    #             df = fn.graficosRedes(G, lista_fundos, ano, mes, metrica_select)
    #             apendDF.append(df)
    # concat_degree = pd.concat(apendDF)

    fig = fn.grafico_tempo_metrica(df, ano, trimestre, metrica_select)
    st.plotly_chart(fig)

row11_spacer1, row11_1, row11_spacer2 = st.columns((.2, 7.1, .2))
with row11_1:
    st.subheader("caminho mais curto")
row12_spacer1, row12_1, row12_spacer2, row12_2, row12_spacer3 = st.columns((.1, 1.3, .4, 5.4, .2))
with row12_1:
    st.markdown("Localização dos Imóveis por Fundo")
    lista_fundos = fn.lista_fundos()
    listinha = data['CNPJ_Fundo'].unique()
    lista_nome = []
    for cnpj in listinha:
        if cnpj[0] == '0':
            cnpj = cnpj[1:]
            print(cnpj)
        cnpj_str = str(cnpj)
        cnpj_formatado = "{}.{}.{}.{}.{}".format(cnpj_str[:1], cnpj_str[1:4], cnpj_str[4:7], cnpj_str[7:10],
                                                 cnpj_str[10:])

        if lista_fundos.loc[lista_fundos['CNPJ_fundo'] == cnpj_formatado].empty == False:
            b = lista_fundos.loc[lista_fundos['CNPJ_fundo'] == cnpj_formatado]
            b.reset_index(inplace=True, drop=True)
            lista_nome.append(b['CODIGO'][0])
        else:
            lista_nome.append(f'NoCode - {cnpj_formatado}')
    cnpj_source = st.selectbox("origem CNPJ", lista_nome)
    cnpj_target = st.selectbox("target CNPJ", data['CNPJ_Emissor'].unique())
with row12_2:
    # if st.sidebar.button('Mostrar Mapa'):
    caminho_mais_curto = nx.shortest_path(G, source=cnpj_source, target=cnpj_target)
    texto = f'O caminho mais curto {caminho_mais_curto}'
