# --------------------LIBRERÍAS----------------------------#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors
import seaborn as sns
import streamlit as st
import streamlit as st
import folium
from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import geopandas as gpd
from branca.colormap import LinearColormap
from langdetect import detect, DetectorFactory
from joblib import Parallel, delayed
from textblob import TextBlob
import spacy
from wordcloud import WordCloud
from PIL import Image
from io import BytesIO
import geopandas as gpd
import streamlit.components.v1 as components

# --------------------CONFIGURACIÓN DE LA PÁGINA----------------------------#
st.set_page_config(page_title='Study on AirBnB data from Vancouver city.', layout='wide', page_icon='img/Flag_of_Vancouver.svg.png')
logo1 = 'img/covlogo-share.png'
logo2 = 'img/Airbnb_Logo_Bélo-1600x1000.png'

# --------------------COLUMNAS----------------------------#
col1, col2, col3 = st.columns(3)
with col1 :
    st.image(logo1, width=300)
    st.write('')
with col2 :
    st.title('AirBnB data from Vancouver city.')
    st.text('This app shows an analysis of Airbnb listings.')
with col3 :
    st.image(logo2, width=300)
    st.write('')

# --------------------DATA A UTILIZAR EN TODA LA APP----------------------------#
df = pd.read_csv('listings_cleaned.csv.gz')
cal = pd.read_csv('cal.csv.gz')
reviews_details = pd.read_csv('reviews_details.csv.gz')
st.dataframe(df.head(5))
st.write('Sample of main dataframe.')

# --------------------TABS----------------------------#
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Geographical Visual Representation of Data",
        "Types of Properties, Accommodations and Number of Guests",
        "Neighborhood Data",
        "Hosts' Relevant Information",
        "Listings Forecast",
        'Reviews Scores',
        'Comments Review'
    ]
)

# --------------------TAB 1----------------------------#
with tab1:
    lats = df['latitude'].tolist()
    lons = df['longitude'].tolist()
    locations = list(zip(lats, lons))

    map1 = folium.Map(location=[49.2327, -123.1207], zoom_start=11)
    FastMarkerCluster(data=locations).add_to(map1)

    vancouver_geojson = "neighbourhoods.geojson"
    vancouver_gdf = gpd.read_file(vancouver_geojson)
    mean_prices = df.loc[df['accommodates'] == 2].groupby('neighbourhood')['price_winsorized'].mean()

    vancouver_gdf = vancouver_gdf.join(mean_prices, on='neighbourhood')

    vancouver_gdf.dropna(subset=['price_winsorized'], inplace=True)

    price_dict = vancouver_gdf.set_index('neighbourhood')['price_winsorized'].round().to_dict()

    color_scale = LinearColormap(['green', 'yellow', 'red'], vmin=min(price_dict.values()), vmax=max(price_dict.values()), caption='Average price')

    def style_function(feature):
        return {
            'fillColor': color_scale(price_dict.get(feature['properties']['neighbourhood'], 0)),
            'color': 'black',
            'weight': 1,
            'dashArray': '5, 5',
            'fillOpacity': 0.5
        }

    def highlight_function(feature):
        return {
            'weight': 3,
            'fillColor': color_scale(price_dict.get(feature['properties']['neighbourhood'], 0)),
            'fillOpacity': 0.8
        }

    map3 = folium.Map(location=[49.2327, -123.1207], zoom_start=11)

    folium.GeoJson(
        data=vancouver_gdf,
        name='Vancouver',
        tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'price_winsorized'], labels=True, sticky=False),
        style_function=style_function,
        highlight_function=highlight_function
    ).add_to(map3)

    map3.add_child(color_scale)

    min_price = df['price_winsorized'].min()
    max_price = df['price_winsorized'].max()

    color_scale = LinearColormap(['green', 'yellow', 'red'], vmin=min_price, vmax=max_price, caption='Price Range')

    vancouver_heatmap = folium.Map(location=[49.2327, -123.1207], zoom_start=11)

    HeatMap(data=df[['latitude', 'longitude', 'price_winsorized']],
            radius=20,
            gradient={0: 'green', 0.5: 'yellow', 1: 'red'},
            min_opacity=0.2).add_to(vancouver_heatmap)

    vancouver_heatmap.add_child(color_scale)

    st.title("Average Price Map by Neighborhood")
    st_folium(map3, returned_objects=[], width=700, height=500)
    st.title("Map of Accommodations Location")
    st_folium(map1, width=700, height=500)
    st.title("Heat Map of Accommodation Prices")
    st_folium(vancouver_heatmap, returned_objects=[], width=700, height=500)
    

# --------------------TAB 2----------------------------#
with tab2:
    prop = df.groupby(['property_type','room_type']).room_type.count()
    prop = prop.unstack()
    prop['total'] = prop.iloc[:,0:3].sum(axis = 1)
    prop = prop.sort_values(by=['total'])
    prop = prop[prop['total']>=75]
    prop = prop.drop(columns=['total'])
    columns_order = ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room']
    prop = prop[columns_order]

    fig = px.bar(prop, barmode='stack', orientation='h',
                color_discrete_sequence=['#00348F', '#005AF5', '#5C98FF', '#C2D8FF'],
                width=1000, height=600)
    fig.update_layout(
       title='Main Property Types in Vancouver',
       xaxis_title='Number of listings',
       yaxis_title='',
       legend_title='',
       font=dict(size=14),
       yaxis=dict(tickmode='linear'),
       margin=dict(l=200)
    )
    st.plotly_chart(fig, use_container_width=True)

    freq = df['room_type'].value_counts().sort_values(ascending=False)

    fig1 = px.bar(freq, orientation='h', color=freq.index,
                 labels={'y': 'Room Type', 'x': 'Number of Listings'},
                 color_discrete_sequence=['#00348F', '#005AF5', '#5C98FF', '#C2D8FF'])
    fig1.update_layout(title="Number of Listings by Room Type",
                      xaxis_title="Number of Listings",
                      yaxis_title="",
                      showlegend=False,
                      height=400, width=800)
    st.plotly_chart(fig1, use_container_width=True)

    color_scale = plotly.colors.make_colorscale(['#99BEFF', '#00348F'])

    feq = df['accommodates'].value_counts().sort_index().reset_index()
    feq.columns = ['Accommodates', 'Number of listings']
    color_sequence = plotly.colors.sample_colorscale(color_scale, feq['Accommodates'].nunique())

    fig2 = px.bar(feq, x='Accommodates', y='Number of listings', 
                 color='Accommodates',
                 color_continuous_scale=color_sequence,
                 width=700, height=500)
    fig2.update_layout(title={'text':"Accommodates Allowed"},
                      xaxis_title='Accommodates', yaxis_title='Number of listings',
                      font=dict(size=14),
                      coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# --------------------TAB 3----------------------------#
with tab3:
    feq1 = df['neighbourhood'].value_counts().sort_values(ascending=False)
    feq1 = feq1[feq1>100]

    colors = ['#000F29', '#00163D', '#001E52', '#002566', '#002D7A', '#00348F', '#003CA3', '#0043B8', '#004BCC', '#0052E0', '#005AF5', '#0A64FF', '#1F71FF', '#337EFF', '#478BFF', '#5C98FF', '#70A5FF', '#99BEFF']

    fig3 = px.bar(feq1, x=feq1.values, y=feq1.index, orientation='h', 
                 color=feq1.index, color_discrete_sequence=colors)

    fig3.update_layout(
        title="Number of Listings by Neighborhood",
        xaxis_title="",
        yaxis_title="",
        font=dict(size=12),
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

    color_scale = plotly.colors.make_colorscale(['#99BEFF', '#00348F'])

    feq2 = df[df['accommodates']==2]
    feq2 = feq2.groupby('neighbourhood')['price_winsorized'].mean().sort_values(ascending=True)

    fig4 = px.bar(feq2, orientation='h', height=700, color=feq2.values,
                 color_continuous_scale=color_scale,
                 title='Average Daily Price for a 2-person Accommodation')

    fig4.update_layout(
        xaxis_title="Average Daily Price in CAD",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig4, use_container_width=True)

    feq3 = df[df['number_of_reviews'] >= 10].groupby('neighbourhood')['review_scores_location'].mean().sort_values(ascending=True)

    fig5 = px.bar(feq3, x='review_scores_location', y=feq3.index, orientation='h', color='review_scores_location', color_continuous_scale=color_scale, height=700, title="Neighborhood Average Review Score")
    fig5.update_layout(xaxis_title="Score (scale 1-5)", yaxis_title="", coloraxis_colorbar_title="Review Scores", yaxis=dict(tickmode='array', tickvals=feq3.index, ticktext=feq3.index))

    st.plotly_chart(fig5, use_container_width=True)

# --------------------TAB 4----------------------------#
with tab4:
    feq4 = df[df['number_of_reviews'] >= 10]
    feq4_sorted = feq4['host_response_rate'].sort_values(ascending=True)

    fig6 = go.Figure()

    fig6.add_trace(go.Histogram(
        x=feq4_sorted,
        nbinsx=35,
        marker=dict(
            color='#00348F',
            line=dict(
                color='#FFFFFF',
                width=1.5
            )
        )
    ))

    fig6.update_layout(
        title='Response rate (minimum 10 reviews)',
        xaxis_title='Response Rate',
        yaxis_title='Number of Listings',
        bargap=0.2,
        showlegend=False,
        width=900,  
        height=600,
        font=dict(size=16)  
    )
    st.plotly_chart(fig6, use_container_width=True)

    response_time_counts = df.dropna(subset=['host_response_time'])['host_response_time'].value_counts().reset_index()
    response_time_counts.columns = ['response_time', 'count']

    fig7 = px.bar(response_time_counts, x='response_time', y='count', labels={'response_time': 'Response time', 'count': 'Listings'}, color_discrete_sequence=['#00348F'])
    fig7.update_layout(title='Response time (minimum 10 reviews)', xaxis_title="", yaxis_title="", font=dict(size=16))
    
    st.plotly_chart(fig7, use_container_width=True)

    color_discrete_map = {'f': '#0A64FF', 't': '#001E52'}

    listings_frequencies = df['host_is_superhost'].value_counts(normalize=True).reset_index()
    listings_frequencies.columns = ['Superhost', 'Percentage']
    listings_frequencies['Percentage'] = listings_frequencies['Percentage'] * 100

    fig8 = px.bar(listings_frequencies, x='Superhost', y='Percentage',
                 labels={'Superhost': '', 'Percentage': 'Percentage (%)'},
                 color='Superhost',
                 color_discrete_map=color_discrete_map)

    fig8.update_traces(texttemplate='%{y:.2f}%', textposition='inside')
    fig8.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig8.update_layout(legend_title='Superhost', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig8.update_layout(
        title_text="Superhost Distribution",
        height=400,
        width=1000,
        font_size=12,
        showlegend=False
    )

    fig8.update_xaxes(tickvals=['f', 't'], ticktext=['Not Superhost', 'Superhost'])

    st.plotly_chart(fig8, use_container_width=True)

    fig_scatter = px.strip(df, x='host_is_superhost', y='price_winsorized',
                           labels={'host_is_superhost': 'Superhost', 'price_winsorized': 'Price in CAD'},
                           color='host_is_superhost',
                           color_discrete_map={'t': '#001E52', 'f': '#478BFF'})

    fig_scatter.update_xaxes(
        categoryorder='array',
        categoryarray=['t', 'f'],
        ticktext=['Not Superhost', 'Superhost'],
        tickvals=['f', 't'],
        title=''
    )

    fig_scatter.update_layout(title='Price Distribution by Superhost Status', showlegend=False)

    st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------TAB 5----------------------------#
with tab5:
    cal['date'] = pd.to_datetime(cal['date'])
    sum_available = cal[cal.available == "t"].groupby(['date']).size().to_frame(name= 'available').reset_index()

    sum_available['weekday'] = sum_available['date'].dt.day_name()

    sum_available = sum_available.set_index('date')

    fig9 = px.line(sum_available, y='available', title='Number of Listings Available by Date')

    fig9.update_layout(
        yaxis_title='Listings Available',
        xaxis_title='Date'
    )

    fig9.update_traces(line=dict(color='#001E52'))

    st.plotly_chart(fig9, use_container_width=True)

    numeric_columns = cal.select_dtypes(include=[np.number]).columns
    average_price = cal[(cal.available == "t") & (cal.accommodates == 2)].groupby(['date'])[numeric_columns].mean().astype(np.int64).reset_index()
    average_price['weekday'] = average_price['date'].dt.day_name()
    average_price = average_price.set_index('date')
    fig10 = px.line(average_price, x=average_price.index, y='price_x', title='Average Price of Available 2-person Accommodation by Date')
    fig10.update_traces(text=average_price['weekday'], line=dict(color='#001E52'))
    fig10.update_layout(xaxis_title='Date', yaxis_title='Price')

    st.plotly_chart(fig10, use_container_width=True)

# --------------------TAB 6----------------------------#
with tab6:
    listings10 = df[df['number_of_reviews']>=10]

    location_rev = px.histogram(listings10, x='review_scores_location',
                 barmode='group', category_orders={'review_scores_location': sorted(listings10['review_scores_location'].unique())})
    location_rev.update_layout(title="Location", title_x=0.5, xaxis_title="", yaxis_title="", font_size=14)
    location_rev.update_traces(marker_color='#001E52')
    st.plotly_chart(location_rev, use_container_width=True)

    cleanliness_rev = px.histogram(listings10, x='review_scores_cleanliness',
                  barmode='group', category_orders={'review_scores_cleanliness': sorted(listings10['review_scores_cleanliness'].unique())})
    cleanliness_rev.update_layout(title="Cleanliness", title_x=0.5, xaxis_title="", yaxis_title="", font_size=14)
    cleanliness_rev.update_traces(marker_color='#003CA3')
    st.plotly_chart(cleanliness_rev, use_container_width=True)

    value_rev = px.histogram(listings10, x='review_scores_value',
                  barmode='group', category_orders={'review_scores_value': sorted(listings10['review_scores_value'].unique())})
    value_rev.update_layout(title="Price Value", title_x=0.5, xaxis_title="", yaxis_title="", font_size=14)
    value_rev.update_traces(marker_color='#005AF5')
    st.plotly_chart(value_rev, use_container_width=True)

    communication_rev = px.histogram(listings10, x='review_scores_communication',
                  barmode='group', category_orders={'review_scores_communication': sorted(listings10['review_scores_communication'].unique())})
    communication_rev.update_layout(title="Communication", title_x=0.5, xaxis_title="", yaxis_title="", font_size=14)
    communication_rev.update_traces(marker_color='#003CA3')
    st.plotly_chart(communication_rev, use_container_width=True)

    checkin_rev = px.histogram(listings10, x='review_scores_checkin',
                  barmode='group', category_orders={'review_scores_checkin': sorted(listings10['review_scores_checkin'].unique())})
    checkin_rev.update_layout(title="Check In", title_x=0.5, xaxis_title="", yaxis_title="", font_size=14)
    checkin_rev.update_traces(marker_color='#005AF5')
    st.plotly_chart(checkin_rev, use_container_width=True)

    accuracy_rev = px.histogram(listings10, x='review_scores_accuracy',
                  barmode='group', category_orders={'review_scores_accuracy': sorted(listings10['review_scores_accuracy'].unique())})
    accuracy_rev.update_layout(title="Accuracy", title_x=0.5, xaxis_title="", yaxis_title="", font_size=14)
    accuracy_rev.update_traces(marker_color='#001E52')
    st.plotly_chart(accuracy_rev, use_container_width=True)

# --------------------TAB 7----------------------------#
with tab7:
    DetectorFactory.seed = 0

    def detect_language(comment):
        try:
            if pd.notnull(comment) and comment.strip() != "":
                return detect(comment)
            else:
                return "unknown"
        except:
            return "unknown"

    sample_data = reviews_details.sample(frac=0.2, random_state=1)  # Usa una muestra del 10% de los datos

    sample_data['language'] = Parallel(n_jobs=-1)(delayed(detect_language)(comment) for comment in sample_data['comments'])

    comments_by_language = sample_data['language'].value_counts().reset_index()
    comments_by_language.columns = ['language', 'count']

    fig11 = px.bar(comments_by_language, x='language', y='count', 
                 title='Number of Comments by Language', 
                 labels={'language': 'Language', 'count': 'Number of Comments'},
                 color='language',
                 color_discrete_sequence=['#001E52'])

    fig11.update_layout(
        xaxis_title='Language',
        yaxis_title='Number of Comments',
        title={
            'text': 'Number of Comments by Language',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(tickangle=90),
        height=400,
        width=1000,
        showlegend=False
    )
    st.plotly_chart(fig11, use_container_width=True)

    english_comments = sample_data[sample_data['language'] == 'en']


    def get_sentiment_polarity(comment):
        return TextBlob(comment).sentiment.polarity


    english_comments['sentiment_polarity'] = english_comments['comments'].apply(get_sentiment_polarity)


    english_comments['sentiment'] = english_comments['sentiment_polarity'].apply(
        lambda x: 'positive' if x > 0.1 else ('neutral' if -0.1 <= x <= 0.1 else 'negative')
    )

    def create_sentiment_histogram(sentiment_type, color):
        filtered_comments = english_comments[english_comments['sentiment'] == sentiment_type]
        fig = px.histogram(filtered_comments, x='sentiment_polarity',
                           title=f'Distribution of {sentiment_type.capitalize()} Comments',
                           labels={'sentiment_polarity': 'Range', 'count': 'Count of Comments'},
                           color_discrete_sequence=[color])
        fig.update_layout(
            xaxis_title='Range',
            yaxis_title='Count of Comments',
            title={
                'text': f'Distribution of {sentiment_type.capitalize()} Comments',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        return fig
    
    negative_histogram = create_sentiment_histogram('negative', '#FF0000')
    neutral_histogram = create_sentiment_histogram('neutral', '#FFA500')
    positive_histogram = create_sentiment_histogram('positive', '#00FF00')

    st.plotly_chart(negative_histogram, use_container_width=True)
    st.plotly_chart(neutral_histogram, use_container_width=True)
    st.plotly_chart(positive_histogram, use_container_width=True)

    fig_polarity = px.histogram(english_comments, x='sentiment_polarity',
                   title='Distribution of Sentiment Polarity of English Comments',
                   labels={'sentiment_polarity': 'Sentiment Polarity', 'count': 'Count of Comments'},
                   color_discrete_sequence=['#001E52'])

    fig_polarity.update_layout(
        xaxis_title='Sentiment Polarity',
        yaxis_title='Count of Comments',
        height=400,
        width=1000,
        title={
            'text': 'Distribution of Sentiment Polarity of English Comments',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        }
    )
    st.plotly_chart(fig_polarity, use_container_width=True)

    st.image('img/wordcloud_vancouver.png')