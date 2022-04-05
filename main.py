import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib
from PIL import Image

import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

import streamlit.components.v1 as components

st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="random",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
        'About': "# This project is part of the final exam of the DEV.F data science course."
     }
 )

# st.set_page_config(layout="wide")
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()





with header:
    st.title('Análisis de sentimientos y Modelo de prediccción de quejas de clientes de "Blue aerolinea"')
    st.markdown('En este proyecto tomamos un extracto de todos los mensajes que diariamente recibe una aerolinea, provenientes de diversas fuentes como: Facebook, twitter, portal propio, por telefono')
    st.markdown('Aplicamos un analisis de sentimiento usando la libreria "pysentimiento" que una vez que analiza las valiables nos arroja 3 resultados NEG (negativo), NEU (neutral) y POS (positivo). A su vez dependiendo el resultado que nos da este analisis podemos categorizar los NEG como quejas, los NEU como comentarios o preguntas y los POS como felicitaciones')

    st.sidebar.title("Análisis de sentimientos y Modelo de prediccción de quejas de clientes de 'Blue aerolinea'")
    st.sidebar.markdown("Esta es un tablero de Streamlit para analizar "
                "sentimientos de los mensajes recibidos por una aerolínea")

    
    data_url = ('data/data_final.csv')
    data = pd.read_csv(data_url)
    data['tipo'] = data['tipo'].replace('comentario o pregunta', 'comentario_o_pregunta',regex=True)
    print(data['tipo'])

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(data_url)
        # data['tweet_created'] = pd.to_datetime(data['tweet_created'])
        return data

    data = load_data()

with dataset:
    st.header('Base de datos de mensajes recibidos de clientes de una aerolinea')
    st.markdown('Se muestra un extracto de la base de datos cruda')
    # aerolinea = pd.read_csv('data/data_final.csv')
    st.write(data.head(8))

    st.sidebar.subheader("Mensaje aleatorio")
    random_msj = st.sidebar.radio('Sentiment', ('felicitacion', 'comentario o pregunta', 'queja'))
    st.sidebar.markdown(data.query("tipo == @random_msj")[["descripcion"]].sample(n=1).iat[0, 0])

    st.sidebar.markdown("### Numero de mensajes por tipo de sentimiento")
    select = st.sidebar.selectbox('Tipo de visualización', ['Bar plot', 'Pie chart'], key='1')
    sentiment_count = data['tipo'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Messages':sentiment_count.values})
    # if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Numero de mensajes por tipo de sentimiento")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Messages', color='Messages', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Messages', names='Sentiment')
        st.plotly_chart(fig)

    st.sidebar.subheader("De donde proviene la mayor cantidad de mensaje recibidos")
    each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
    tipo_count = data.groupby('fuente')['tipo'].count().sort_values(ascending=False)
    tipo_count = pd.DataFrame({'fuente':tipo_count.index, 'descripcion':tipo_count.values.flatten()})
    # if not st.sidebar.checkbox("Close", True, key='2'):
    if each_airline == 'Bar plot':
        st.subheader("De donde proviene la mayor cantidad de mensaje recibidos")
        fig_1 = px.bar(tipo_count, x='fuente', y='descripcion', color='descripcion', height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("De donde proviene la mayor cantidad de mensaje recibidos")
        fig_2 = px.pie(tipo_count, values='descripcion', names='fuente')
        st.plotly_chart(fig_2)

    @st.cache(persist=True)
    def plot_sentiment(fuente1):
        df = data[data['fuente']==fuente1]
        count = df['tipo'].value_counts()
        count = pd.DataFrame({'Tipo':count.index, 'Descripcion':count.values.flatten()})
        return count


    st.sidebar.subheader("Tipos de mensaje por cada fuente donde se reciben")
    choice = st.sidebar.multiselect('Elegir fuente', ('Facebook','Twitter','Portal','Email','Phone'))
    if len(choice) > 0:
        st.subheader("Tipos de mensaje por cada fuente donde se reciben")
        breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
        fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
        if breakdown_type == 'Bar plot':
            for i in range(1):
                for j in range(len(choice)):
                    fig_3.add_trace(
                        go.Bar(x=plot_sentiment(choice[j]).Tipo, y=plot_sentiment(choice[j]).Descripcion, showlegend=False),
                        row=i+1, col=j+1
                    )
            fig_3.update_layout(height=600, width=800)
            st.plotly_chart(fig_3)
        else:
            fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
            for i in range(1):
                for j in range(len(choice)):
                    fig_3.add_trace(
                        go.Pie(labels=plot_sentiment(choice[j]).Tipo, values=plot_sentiment(choice[j]).Descripcion, showlegend=True),
                        i+1, j+1
                    )
            fig_3.update_layout(height=600, width=800)
            st.plotly_chart(fig_3)

    st.subheader('Por cada tipo de mensaje se muestra el horario en que más se reciben')
    tipo = pd.DataFrame(data['tipo'].value_counts())
    # st.bar_chart(tipo)

    line_chart_data = data.copy()
    line_chart_data['time'] = pd.to_datetime(line_chart_data['time'])
    line_chart_data['hour'] = line_chart_data['time'].dt.hour

    hour_cross_tab = pd.crosstab(line_chart_data['hour'], line_chart_data['tipo'])
    fig = px.line(hour_cross_tab)
    fig.update_layout(
        showlegend=True,
        width=700,
        height=400,
        margin=dict(l=1,r=1,b=1,t=1),)
    st.write(fig)

    st.subheader('Por cada fuente se muestra el horario en que más se reciben')
    hour_cross_tab1 = pd.crosstab(line_chart_data['hour'], line_chart_data['fuente'])
    fig1 = px.line(hour_cross_tab1)
    fig1.update_layout(
        showlegend=True,
        width=700,
        height=400,
        margin=dict(l=1,r=1,b=1,t=1),)
    st.write(fig1)

    #Define stop words
    stop_words_sp = set(stopwords.words('spanish'))
    # stop_words_en = set(stopwords.words('english'))
    stop_words = stop_words_sp #| stop_words_en
    #add words that aren't in the NLTK stopwords list
    new_stopwords = ["buenas", "buen","blueaerolinea","saludos cordiales","com mx","www","gmail com","mailto","gmail", "com","hotmail","image004","hola","tarde","gmail com","muchas","gracias", "buenos","día","cid","mx","ustede","correo", "electronico",
                          "saludos","cordiales", "formato","pdf","solo","ustedes","quedo","espera","noche", "blue", "aerobu","adjunto podrá", "términos", "condiciones","correo electrónico","saber", "si","podrás encontrar","quedo","espera","noche",
                          "adjunto","podrá", "png","podrás","encontrar","grupos","quedo","favor","hoy", "quisiera", "tardes","adjunto podrá", "términos condiciones","correo electrónico","saber", "si","image","6c756af0","noches","01d57aa3",
                      "com", "nbsp","cognitoforms","dia","quedo","gracias","vea","saludos","favor", "días","hola","image004","xx ","tarde","gmail com","muchas","SECTUR_GobOax", "VIVAELTEATRO","TigresFemenil","CasaOaxacaRest","HLaureles","SECTUR_GobOax HLaureles","puedo", "hacer"]
    new_stopwords_list = stop_words.union(new_stopwords)
    not_stopwords = {}
    final_stop_words = set([word for word in new_stopwords_list if word not in not_stopwords])

    st.sidebar.header("Word Cloud")
    word_sentiment1 = st.sidebar.radio('Display word cloud for what sentiment?', ('queja', 'comentario_o_pregunta', 'felicitacion'))
    # if not st.sidebar.checkbox("Close", True, key='3'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment1))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df1 = data[data['tipo']==word_sentiment1]
    words1 = ' '.join(df1['descripcion'].values)
    # processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    # wordcloud1 = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(words1)
    wordcloud1 = WordCloud(stopwords=final_stop_words, background_color='white', width=1000, height=640).generate(words1)
    plt.imshow(wordcloud1)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

#     #--------------------------------wordclouds separados begin-------------
#     #dividimos el dataset en quejas, comentarios y felicitaciones
#     quejas = data[data['tipo'].str.contains("queja")]
#     comentarios = data[data['tipo'].str.contains("comentario o pregunta")]
#     felicitaciones = data[data['tipo'].str.contains("felicitacion")]
#
#     #Define stop words
#     stop_words_sp = set(stopwords.words('spanish'))
#     # stop_words_en = set(stopwords.words('english'))
#     stop_words = stop_words_sp #| stop_words_en
#     #add words that aren't in the NLTK stopwords list
#     new_stopwords = ["buenas", "buen","blueaerolinea","saludos cordiales","com mx","www","gmail com","mailto","gmail", "com","hotmail","image004","hola","tarde","gmail com","muchas","gracias", "buenos","día","cid","mx","ustede","correo", "electronico",
#                           "saludos","cordiales", "formato","pdf","solo","ustedes","quedo","espera","noche", "blue", "aerobu","adjunto podrá", "términos", "condiciones","correo electrónico","saber", "si","podrás encontrar","quedo","espera","noche",
#                           "adjunto","podrá", "png","podrás","encontrar","grupos","quedo","favor","hoy", "quisiera", "tardes","adjunto podrá", "términos condiciones","correo electrónico","saber", "si","image","6c756af0","noches","01d57aa3",
#                       "com", "nbsp","cognitoforms","dia","quedo","gracias","vea","saludos","favor", "días","hola","image004","xx ","tarde","gmail com","muchas","gracias", "buenos","día","cid","mx","ustede","puedo", "hacer"]
#     new_stopwords_list = stop_words.union(new_stopwords)
#     not_stopwords = {}
#     final_stop_words = set([word for word in new_stopwords_list if word not in not_stopwords])
#
#     st.subheader('Palabras más frecuentes en quejas')
#
#     #creamos diccionario de las 100 palabras mas usadas en quejas sin stopwords
#     quejas['desc_sw'] = quejas['descripcion'].apply(lambda x: ' '.join([word for word in x.split() if word not in (final_stop_words)]))
#     q_w= dict(Counter(" ".join(quejas["desc_sw"]).split()).most_common(100))
#
#     #creamos wordcloud para quejas
#     wordcloud_q = WordCloud(width = 1000, height = 500).generate_from_frequencies(q_w)
#     plt.figure(figsize=(15,8))
#     plt.imshow(wordcloud_q)
#     plt.axis("off")
#     st.pyplot()
#
#     st.subheader('Palabras más frecuentes en felicitaciones')
#
#     #creamos diccionario de las 100 palabras mas usadas en quejas sin stopwords
#     felicitaciones['desc_sw'] = felicitaciones['descripcion'].apply(lambda x: ' '.join([word for word in x.split() if word not in (final_stop_words)]))
#     f_w= dict(Counter(" ".join(felicitaciones["desc_sw"]).split()).most_common(100))
#
#     #creamos wordcloud para quejas
#     wordcloud_f = WordCloud(width = 1000, height = 500).generate_from_frequencies(f_w)
#     plt.figure(figsize=(15,8))
#     plt.imshow(wordcloud_f)
#     plt.axis("off")
#     st.pyplot()
#
#     st.subheader('Palabras más frecuentes en comentarios o preguntas')
#
#     #creamos diccionario de las 100 palabras mas usadas en quejas sin stopwords
#     comentarios['desc_sw'] = comentarios['descripcion'].apply(lambda x: ' '.join([word for word in x.split() if word not in (final_stop_words)]))
#     c_w= dict(Counter(" ".join(comentarios["desc_sw"]).split()).most_common(100))
#
#     #creamos wordcloud para quejas
#     wordcloud_c = WordCloud(width = 1000, height = 500).generate_from_frequencies(c_w)
#     plt.figure(figsize=(15,8))
#     plt.imshow(wordcloud_c)
#     plt.axis("off")
#     st.pyplot()
# #--------------------------------wordclouds separados end-------------


# with features:
#     st.header('Creación y selección de features')

#     st.markdown('* **First Feature:** bla bla')
#     st.markdown('* **Second Feature:** bla bla')

with model_training:

    import streamlit.components.v1 as components
    st.header('Elegimos el modelo a entrenar ')
    st.subheader('Latent Dirichlet Allocation(LDA) from Gensim package')
    st.text('Perplexity:  -8.574979549521306')
    st.text('Coherence Score:  0.6029944051045057')

    st.header("Visualizacion de temas")

    HtmlFile = open("data/lda.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code,height = 900)

    st.subheader('Mallet(LDA) implementation model from Gensim package')
   
    from PIL import Image
    image = Image.open('data/graph.png')

    st.image(image, caption='Number topics graph')

    st.write('Num Topics = 2  has Coherence Value of 0.5241')
    st.write('Num Topics = **8  has Coherence Value of 0.7051**')
    st.write('Num Topics = 14  has Coherence Value of 0.6961')
    st.write('Num Topics = 20  has Coherence Value of 0.6958')
    st.write('Num Topics = 26  has Coherence Value of 0.6989')
    st.write('Num Topics = 32  has Coherence Value of 0.6862')
    st.write('Num Topics = 38  has Coherence Value of 0.6832')

    code = '''[
     (0,
     '0.027*"reservacion" + 0.022*"pago" + 0.016*"boleto" + 0.016*"compra" + '
     '0.015*"pagina" + 0.014*"boletos" + 0.013*"tarjeta" + 0.012*"reserva" + '
     '0.012*"costo" + 0.011*"comprar"'),
     (1,
     '0.132*"vuelo" + 0.041*"mexico" + 0.036*"cancun" + 0.035*"monterrey" + '
     '0.032*"ciudad" + 0.022*"regreso" + 0.019*"julio" + 0.019*"pm" + '
     '0.019*"cdmx" + 0.017*"vuelos"'),
     (2,
     '0.029*"reservacion" + 0.024*"covid" + 0.022*"nombre" + 0.017*"clave" + '
     '0.016*"cambio" + 0.015*"prueba" + 0.011*"motivo" + 0.011*"reembolso" + '
     '0.010*"solicito" + 0.010*"anexo"'),
     (3,
     '0.051*"vuelo" + 0.023*"viajar" + 0.020*"vuelos" + 0.019*"necesito" + '
     '0.018*"mas" + 0.017*"puede" + 0.013*"quiero" + 0.012*"cambiar" + '
     '0.011*"poder" + 0.010*"gustaria"'),
     (4,
     '0.044*"cotizacion" + 0.035*"fecha" + 0.033*"dias" + 0.021*"confirmacion" + '
     '0.018*"mty" + 0.017*"cuenta" + 0.016*"presente" + 0.014*"pasajeros" + '
     '0.012*"enviar" + 0.011*"nueva"'),
     (5,
     '0.023*"maleta" + 0.019*"aeropuerto" + 0.017*"equipaje" + 0.013*"abordar" + '
     '0.012*"avion" + 0.010*"aerolinea" + 0.010*"tenia" + 0.009*"hora" + '
     '0.009*"habia" + 0.008*"llegar"'),
     (6,
     '0.032*"atencion" + 0.032*"respuesta" + 0.032*"numero" + 0.021*"mas" + '
     '0.020*"informacion" + 0.019*"servicio" + 0.018*"mismo" + 0.013*"caso" + '
     '0.013*"dias" + 0.013*"aerobus"'),
     (7,
     '0.024*"nombre" + 0.023*"factura" + 0.022*"clave" + 0.020*"datos" + '
     '0.017*"pasajero" + 0.015*"solicito" + 0.013*"apoyo" + 0.011*"reservacion" + '
     '0.011*"rfc" + 0.010*"facturacion"')]'''
    st.code(code, language='python')

    st.markdown('Se muestra el top de mensajes')
    top_mensajes = pd.read_csv('data/top_mensajes.csv')
    st.write(top_mensajes)

    # st.markdown('Se muestra un extracto del numero y porcentaje de documentos por tema')
    # docs_tema = pd.read_csv('data/dominant_topics.csv')
    # st.write(docs_tema.head(8))

