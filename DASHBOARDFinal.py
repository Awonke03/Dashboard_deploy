import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash 
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import dash_table
import statsmodels.api as sm
from matplotlib import pyplot as plt

crop_recommendation_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/Crop_recommendation.csv'
train_data_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/Train.csv'
land_area_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/LandArea2020.xls'
nc_data_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/NC%20Data.xlsx'
gdp_data_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/CountriesGdp.csv'
agricultural_land_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/Agricultural%20land%20in%20south%20as%20a%20share%20of%20lanD%20IN%20SA%20.xlsx'
agriculture_rural_url = 'https://raw.githubusercontent.com/Awonke03/Dashboard_deploy/main/agriculture-and-rural-development_zaf.csv'


df = pd.read_csv(crop_recommendation_url)
data = pd.read_csv(train_data_url)
df4 = pd.read_excel(land_area_url)
df5 = pd.read_excel(nc_data_url)
df3 = pd.read_csv(gdp_data_url)
df1 = pd.read_excel(agricultural_land_url)
df2 = pd.read_csv(agriculture_rural_url)



df2.head()

summary_stats = df2.describe()
print("Summary Statistics:")
summary_stats

df2.drop(columns=['Country Name', 'Country ISO3'],inplace=True)
df2.sample(5)

df2 = df2.drop(df.index[0])
df2.head()

#df2 = df2.sort_values(by=['Indicator Name', 'Year'])
df2.tail(5)


# In[8]:


df2['Year'] = df2['Year'].astype(int)
df_2008 = df2[df2['Year'] == 2008]


# In[9]:


df2.info()


# In[10]:


missing_values = df2.isnull().sum()
print("\nMissing Values:")
print(missing_values)


# In[11]:


df2['Indicator Name'].unique()


# In[12]:


df2['Value'] = pd.to_numeric(df2['Value'])


# In[13]:


df2[df2['Indicator Name'] == 'Agricultural raw materials imports (% of merchandise imports)']


# In[14]:


data.head()


# In[15]:


df


# In[16]:


df_summary = pd.pivot_table(df,index=['label'],aggfunc='mean')
df_summary.head()


# In[17]:


'''consumer_key = "Os2XHEdWjIYfqUYXluK5JoeIP"
consumer_secret = "iEsbaLbp6LSoRTyrmVtkFJ580LkMWBumr9ucMpD0QiKjcF9PJy"
access_token = "1676194519295950849-W7jkNKhqA77ijX2sfsUPh51GKSHxE1"
access_token_secret = "dOgtjNoksxJJiZ5hCe8ZZLOrjf4bUNkCBOEp7QVh038Gu"

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)
search_query = "agriculture"

tweets = api.search_tweets(q=search_query, count=100)
tweet_texts = [tweet.text for tweet in tweets]
all_tweets_text = " ".join(tweet_texts)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tweets_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()'''


# In[18]:


imports_df = df2[df2['Indicator Name'] == 'Agricultural raw materials imports (% of merchandise imports)']

exports_df = df2[df2['Indicator Name'] == 'Agricultural raw materials exports (% of merchandise exports)']

trace_imports = go.Scatter(x=imports_df['Year'], y=imports_df['Value'], mode='lines', name='Imports')
trace_exports = go.Scatter(x=exports_df['Year'], y=exports_df['Value'], mode='lines', name='Exports')

layout = go.Layout(title='Agricultural Raw Materials Imports and Exports',
                   xaxis=dict(title='Year'),
                   yaxis=dict(title='Percentage of Merchandise Imports/Exports'))

trace_imports_exports = go.Figure(data=[trace_imports, trace_exports], layout=layout)

trace_imports_exports.show()


# In[19]:


Economicallyactive_df = df2[df2['Indicator Name'] == 'Economically active population in agriculture (number)']

Economicallyactive = px.line(Economicallyactive_df, x='Year', y='Value', title='Economically active population in agriculture',
              labels={'Year': 'Year', 'Value': 'Value'})
Economicallyactive.show()


# In[20]:


female_employ_df = df2[df2['Indicator Name'] == 'Employment in agriculture, female (% of female employment) (modeled ILO estimate)']

male_employ_df = df2[df2['Indicator Name'] == 'Employment in agriculture, male (% of male employment) (modeled ILO estimate)']

Employment_df= df2[df2['Indicator Name'] == 'Employment in agriculture (% of total employment) (modeled ILO estimate)']


# In[21]:


Economicallyactivepredict = px.line(Employment_df, x='Year', y='Value', title='Employment in agriculture (% of total employment) (modeled ILO estimate)',
              labels={'Year': 'Year', 'Value': 'Value'})
Economicallyactivepredict.show()


# In[22]:


x = df_summary.index
y1 = df_summary['temperature']
y2 = df_summary['humidity']
y3 = df_summary['rainfall']

color1 = 'rgb(255, 102, 102)'  
color2 = 'rgb(102, 204, 102)' 
color3 = 'rgb(102, 178, 255)'  

work_exp_clusters = go.Figure()

work_exp_clusters.add_trace(go.Bar(x=x, y=y1, name='Temperature', marker_color=color1, offsetgroup=0))
work_exp_clusters.add_trace(go.Bar(x=x, y=y2, name='Humidity', marker_color=color2, offsetgroup=1))
work_exp_clusters.add_trace(go.Bar(x=x, y=y3, name='Rainfall', marker_color=color3, offsetgroup=2))

work_exp_clusters.update_layout(title="Temperature-Humidity-Rainfall values comparison between crops",
                  xaxis=dict(title="Crop"),
                  yaxis=dict(title="Environmental Values"),
                  xaxis_tickangle=-45,
                  barmode='group',
                  bargap=0.2,  
                  legend=dict(x=0, y=1),
                  margin=dict(l=50, r=50, t=50, b=50)) 
work_exp_clusters.show()


# In[23]:


x = df_summary.index
y1 = df_summary['N']
y2 = df_summary['P']
y3 = df_summary['K']

color1 = 'rgb(255, 102, 102)'  # Red
color2 = 'rgb(102, 204, 102)'  # Green
color3 = 'rgb(102, 178, 255)'  #
prof_clusters= go.Figure()
prof_clusters.add_trace(go.Bar(x=x, y=y1, name='Nitrogen', marker_color=color1))
prof_clusters.add_trace(go.Bar(x=x, y=y2, name='Phosphorous', marker_color=color2, base=y1))
prof_clusters.add_trace(go.Bar(x=x, y=y3, name='Potasium', marker_color=color3, base=[sum(i) for i in zip(y1, y2)]))

prof_clusters.update_layout(title="N-P-K values comparison between crops",
                  xaxis=dict(title="Crop"),
                  yaxis=dict(title="Nutrient Value"),
                  xaxis_tickangle=-45,
                  legend=dict(x=1, y=0.5), 
                  barmode='stack',
                  bargap=0.15, 
                  margin=dict(l=50, r=50, t=50, b=50)) 

prof_clusters.show()


# In[24]:


SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '12rem',
    'padding': '2rem 1rem',
    'background-color': 'rgba(120, 120, 120, 0.4)',
}

CONTENT_STYLE = {
    'margin-left': '15rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
}

dropdown_options = [{'label': str(id), 'value': id} for id in data['ID']]

sidebar = html.Div(
    [
        html.Hr(),
        html.P('Agriculture Optimazation', className='text-center p-3 border border-dark'),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("SA Land", href="/SA-land", className="nav-link"),
                
                dbc.NavLink('Effects', href="/Effects", active='exact'),
            ],
            vertical=True,
            pills=True,
        ),
        html.Div(
            [
                html.P('Leveraging Technology', style={'text-align': 'center', 'margin': '0'}),
                html.P('To help and improve the ', style={'text-align': 'center', 'margin': '0'}),
                html.P('Agricutltural department', style={'text-align': 'center', 'margin': '0'}),
            ],
            style={'padding': '10px 0', 'text-align': 'center'}
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id='page-content', children=[], style=CONTENT_STYLE)


# ## CHARTS

# In[25]:


train_df = data


# In[26]:


green_scale = [[0, '#f7fcf5'], [0.25, '#c7e9c0'], [0.5, '#41ab5d'], [0.75, '#238b45'], [1, '#005a32']]

fig_gender = go.Figure(data=go.Choropleth(
    locations=df4['Country'],
    locationmode='country names', 
    z=df4['Land Area (1000 hectares)'],  
    colorscale=green_scale, 
    colorbar_title='Land Area (1000 hectares)',  
))

fig_gender.update_layout(
    title='Land Area by Country (Choropleth Map)',
    geo=dict(
        showframe=False,  
        projection_type='equirectangular' 
    )
)

fig_gender.show()


# In[27]:


fig_gender_bar = px.bar(
    df3,
    x='Country',
    y='GDP in Africa 2021 in %;',
    color='GDP in Africa 2021 in %;', 
    color_continuous_scale='Viridis',
    title='The contribution of agriculture to the GDP in Africa by Country',
    labels={'GDP in Africa 2021 in %;': 'GDP in Africa 2021 %'},
    template='plotly_white'
)


fig_gender.update_layout(
    xaxis=dict(title='Country'),
    yaxis=dict(title='GDP in Africa 2021 in %;'),
)
fig_gender_bar.show()


# In[28]:


'''prof_cluster_heatmap = go.Figure(data=go.Heatmap(z=df.corr(), x=df.columns, y=df.columns, colorscale='viridis'
))

prof_cluster_heatmap.update_layout(title='Correlation between different features', title_font=dict(size=15),
                  width=1050, height=500)

prof_cluster_heatmap.show()'''


# In[ ]:





# In[29]:


age_scatter = go.Figure()

age_scatter.add_trace(go.Scatter(
    x=df1['Year '],  
    y=df1['Share of land'],
    mode='lines+markers',
    marker=dict(
        size=10,
        color=list(range(len(df1))), 
        colorscale='Viridis', 
        colorbar=dict(title='Year')
    ),
    line=dict(color='darkblue', width=2), 
    name='Share of land'
))

age_scatter.update_layout(
    title='Agricultural land as a share of Land area in South Africa from 2000 to 2020',
    xaxis=dict(title='Index'),
    yaxis=dict(title='Share of agricultural land'),
    showlegend=True,
    template='plotly_white'  
)

age_scatter.show()


# In[30]:


northern_cape_df_sorted = df5.sort_values(by='Agricultural Potential', ascending=False)

sunburst2 = go.Figure(data=[
    go.Bar(name='Agricultural Potential', x=northern_cape_df_sorted['City/Town in the NC'], y=northern_cape_df_sorted['Agricultural Potential']),
    go.Bar(name='Size of Economy', x=northern_cape_df_sorted['City/Town in the NC'], y=northern_cape_df_sorted['Size of Economy'])
])

sunburst2.update_layout(barmode='group', title='Agricultural Potential and Size of Economy of Towns in the Northern Cape')

sunburst2.show()


# ### DASHBOARD PAGES

# #### HOMEPAGE

# In[31]:
homepage = html.Div([
    html.H2('The Alpha Team', className='text-center p-3'),

    html.Div([
        html.Div([
            dbc.Row([
                dcc.Graph(figure=fig_gender, config={'displayModeBar': False}),
            ]),
        ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='customer-distribution'),
   
        html.Div([
            dbc.Row([
                dcc.Graph(figure=fig_gender_bar, config={'displayModeBar': False}),
            ]),
        ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='customer-distribution'),
    ]),

    html.Div([
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=sunburst2, config={'displayModeBar': False}),
                width=12
            ),
        ]),
    ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='sunburst-charts'),

    html.Div([
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=work_exp_clusters, config={'displayModeBar': False}),
                width=6
            ),
            dbc.Col(
                dcc.Graph(figure=prof_clusters, config={'displayModeBar': False}),
                width=6
            ),
        ]),
    ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='customer-distribution'),
])


# Contributions Page

# In[32]:
contributions = html.Div([
    html.H2('The Alpha Team', className='text-center p-3'),

    html.Div([
        html.Div([
            dbc.Row([
                dcc.Graph(figure=Economicallyactive, config={'displayModeBar': False}),
            ]),
        ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='customer-distribution'),

        html.Div([
            dbc.Row([
                dcc.Graph(figure=Economicallyactivepredict, config={'displayModeBar': False}),
            ]),
        ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='customer-distribution'),
    ]),

    html.Div([
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=trace_imports_exports, config={'displayModeBar': False}),
                width=900
            )
        ]),
    ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='customer-distribution'),


    html.Div([
        dbc.Row([
            dcc.Graph(figure=prof_cluster_heatmap, config={'displayModeBar': False}),
        ]), 
    ], className='p-3 mb-2 bg-light text-dark', style={'border': '1px solid #ccc'}, id='sunburst-charts'),
])





# #### DASH APP

# In[33]:


CONTENT_STYLE = {
    'margin-left': '15rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
}


# In[34]:



app = JupyterDash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
app.css.append_css({
    'external_url': 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
})



app.layout = html.Div([
    dcc.Location(id='url'),
    sidebar,
    content
])



@app.callback(
    Output('tableview-table', 'data'),
    Input('search-textbox', 'value'),
    Input('search-by-dropdown', 'value')
)
def update_table(search_value, search_column):
    if search_value and search_column:
        filtered_data = data[data[search_column].str.contains(search_value, case=False)]
        return filtered_data.to_dict('records')
    else:
        return data.to_dict('records')


# In[37]:


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    print("Current pathname:", pathname)
    if pathname == "/SA-land":
        return homepage
    elif pathname == "/":
        return homepage
    elif pathname == '/Effects':
        return contributions
    else:
        return html.P("Page not found")  


# In[38]:
app=app.server


if __name__ == "__main__":
    app.run_server(debug=True, port=8020)
  

