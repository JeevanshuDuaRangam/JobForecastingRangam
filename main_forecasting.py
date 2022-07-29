# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 07:00:38 2022

@author: Jeevanshudua
"""



import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="MyApp")

def convert_negative(num):
    if num<0:
        return 0
    else:
        return num
   
@st.cache(ttl=24*60*60)
def findGeocode(city):
	geolocator = Nominatim(user_agent="your_app_name")
	return geolocator.geocode(city)
			


@st.cache(ttl=24*60*60)
def fetch_data():

    df = pd.read_csv(r"job_forecasting.csv")
    
    return df

def get_metric_remote(df):
    new_df = df.groupby(["Date", "IsRemoteLocation"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName','CityName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText", "ClientName"], axis = 1)
    perc = int(len(new_df[new_df['IsRemoteLocation']==True])/len(new_df)*100)
    return perc
    
def get_top_job_titles(df):
    new_df = df.groupby(["JobTitleText"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.iloc[:10,:]
    return create_pie_chart(new_df, "RequirementID", "JobTitleText" )

def get_top_clients(df):
    new_df = df.groupby(["ClientName"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.iloc[:10,:]
    return create_pie_chart(new_df, "RequirementID", "ClientName" )

def get_top_cities(df):
    longitude = []
    latitude = []
    new_df = df.groupby(["CityName"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    citynames = new_df.index.tolist()[:50]
    
    for i in citynames:
        if findGeocode(i) != None:
             loc = findGeocode(i)
             latitude.append(loc.latitude)
             longitude.append(loc.longitude)
        else:
            latitude.append(np.nan)
            longitude.append(np.nan)
        
    data = pd.DataFrame({"lat":latitude, "lon": longitude})
    return data
    
def get_title_remote(df, job_type):
    new_df = df.groupby(["Date", "IsRemoteLocation"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName','CityName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText", "ClientName"], axis = 1)
    if job_type == "Remote Jobs":
        is_remote = True 
        new_df = new_df[new_df['IsRemoteLocation']==is_remote]
        new_df = df.groupby(["Date", "IsRemoteLocation", "JobTitleText"]).count()
        new_df = new_df.reset_index()
        new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
        new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
        new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
        new_df = new_df.groupby(["JobTitleText"]).count()
        new_df = new_df["RequirementID"].sort_values(ascending = False)
        titles = new_df.index.tolist()[:5]
    return titles

def get_client_remote(df, job_type):
    new_df = df.groupby(["Date", "IsRemoteLocation"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'CategoryName','CityName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText", "ClientName"], axis = 1)
    if job_type == "Remote Jobs":
        is_remote = True 
        new_df = new_df[new_df['IsRemoteLocation']==is_remote]
        new_df = df.groupby(["Date", "IsRemoteLocation","ClientName" ]).count()
        new_df = new_df.reset_index()
        new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
        new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
        new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
        new_df = new_df.groupby(["ClientName"]).count()
        new_df = new_df["RequirementID"].sort_values(ascending = False)
        client_names = new_df.index.tolist()[:5]
    return client_names
    
def get_titles_cities(df, city_name):
    new_df = df.groupby(["Date", "CityName", "JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "CityName", "JobTitleText", "RequirementID"]]
    new_df = new_df[new_df["CityName"]==city_name]
    new_df = new_df.groupby(["JobTitleText"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    titles = new_df.index.tolist()[:5]
    return titles

def get_titles_clients(df, client_name):
    new_df = df.groupby(["Date", "ClientName", "JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "ClientName", "JobTitleText", "RequirementID"]]
    new_df = new_df[new_df["ClientName"]==client_name]
    new_df = new_df.groupby(["JobTitleText"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    titles = new_df.index.tolist()[:5]
    return titles

def get_clients_titles(df, job_title):
    
    new_df = df.groupby(["Date","JobTitleText", "ClientName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "ClientName", "JobTitleText", "RequirementID"]]
    new_df = new_df[new_df["JobTitleText"]==job_title]
    new_df = new_df.groupby(["ClientName"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    clientnames = new_df.index.tolist()[:5]   
    return clientnames


def create_pie_chart(df, values, names):
    fig = px.pie(df, values= values, names= names,)
    return fig
     

def create_remote_plot(df, job_type):
    new_df = df.groupby(["Date", "IsRemoteLocation"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName','CityName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText", "ClientName"], axis = 1)
    if job_type == "Remote Jobs":
        is_remote = True 
        new_df = new_df[new_df['IsRemoteLocation']==is_remote]
    elif job_type == "Non-Remote Jobs":
        is_remote = False 
        new_df = new_df[new_df['IsRemoteLocation']==is_remote]
    return create_forecast(new_df)
    

def create_prophet_remote_plot(df, is_remote):
    new_df = df.groupby(["Date", "IsRemoteLocation"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName','CityName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText", "ClientName"], axis = 1)
    if job_type == "Remote Jobs":
        is_remote = True 
        new_df = new_df[new_df['IsRemoteLocation']==is_remote]
    elif job_type == "Non-Remote Jobs":
        is_remote = False 
        new_df = new_df[new_df['IsRemoteLocation']==is_remote]
    return evaluate_model(new_df) 
 
    
def create_city_plot(df, city_name):
    new_df = df.groupby(["Date", "CityName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText","IsRemoteLocation", "ClientName"], axis = 1)
    data = new_df[new_df["CityName"]==city_name]     
    return create_forecast(data)

def create_prophet_city_plot(df, city_name):   
    new_df = df.groupby(["Date", "CityName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText","IsRemoteLocation", "ClientName"], axis = 1)
    data = new_df[new_df["CityName"]== city_name]
    return evaluate_model(data) 

 
def create_job_title_plot(df, job_title):
    new_df = df.groupby(["Date","JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['ClientName','Date','CreatedDate','CategoryName', 'ClientName','JobTypeText', 'IsRemoteLocation'], axis = 1)
    data = new_df[new_df["JobTitleText"]==job_title]
    return create_forecast(data)


def create_prophet_jobtitle_plot(df, job_title):       
    new_df = df.groupby(["Date","JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['ClientName','Date','CreatedDate','CategoryName', 'ClientName','JobTypeText', 'IsRemoteLocation'], axis = 1)
    data = new_df[new_df["JobTitleText"]==job_title]
    return evaluate_model(data) 
    

def create_client_plot(df, client_name):
    
    new_df = df.groupby(["Date", "ClientName", "CategoryName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df =new_df.groupby([new_df.index,"ClientName"]).sum().reset_index()
    new_df = new_df.rename(columns = {new_df.columns[0]:"Date"} )
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(["Date","CreatedDate", "JobTitleText", "JobTypeText","IsRemoteLocation"], axis = 1)
    data = new_df[new_df["ClientName"]== client_name]
    return create_forecast(data) 
    

def create_prophet_client_plot(df, client_name):
    new_df = df.groupby(["Date", "ClientName", "CategoryName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df =new_df.groupby([new_df.index,"ClientName"]).sum().reset_index()
    new_df = new_df.rename(columns = {new_df.columns[0]:"Date"} )
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(["Date","CreatedDate", "JobTitleText", "JobTypeText","IsRemoteLocation"], axis = 1)
    data = new_df[new_df["ClientName"]== client_name]
    return evaluate_model(data) 
    
def evaluate_model(data):   
    data = data.iloc[:,1]
    data = pd.DataFrame(data)
    data = data.reset_index()
    data.columns = ["ds", "y"]
    #data["y"] = data["y"].apply(np.log)
    model = Prophet()
    try:
        model.fit(data)
    except ValueError:
        st.warning("Not Enough Data Found")
    else:
    
        forecast = model.make_future_dataframe(12, freq = "M")  
        forecast = model.predict(forecast)
        #forecast["yhat"] = forecast["yhat"].apply(np.exp)
        forecast["yhat"] = forecast["yhat"].apply(int)
        fig_1 = model.plot(forecast, figsize=(20, 12))
        ax = fig_1.gca()
        ax.set_title("Projection: Line Represents Projection by the Model and Points are Actual Value.", size=24)
        ax.set_xlabel("Date:", size=12)
        ax.set_ylabel("Requirements: ", size=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        fig_2 = model.plot_components(forecast,figsize=(20, 12))
        ax = fig_2.gca()
        ax.set_title("Variation Trends: ", size=8)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        se = np.square(forecast.loc[:, 'yhat'] - data.loc[:, "y"])
        mse = np.mean(se)
        rmse = np.sqrt(mse)
    
        #return fig_1, fig_2, rmse
        return st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse), st.caption('RMSE Score is lower the better.')



def get_text(lis):
    
    s = ''
    for i in lis:
        s += "- " + i + "\n" 
    return st.markdown(s)




#@st.cache
def create_forecast(data):
    data = data.iloc[:,1]
    data = pd.DataFrame(data)
    data = data.reset_index()
    data.columns = ["ds", "y"]
    #data["y"] = data["y"].apply(np.log)
    model = Prophet()
    try:
        model.fit(data)
    except ValueError:
        st.warning("Not Enough Data Found")
    else:
        
        forecast = model.make_future_dataframe(12, freq = "M")  
        forecast = model.predict(forecast.tail(12))
        forecast = forecast[['ds', 'yhat']]
        forecast["yhat"] = forecast["yhat"].apply(lambda x: convert_negative(x))
        forecast["yhat"] = forecast["yhat"].apply(int)
        forecast.columns = ["Date", "Requirements"]
        forecast = forecast.set_index(pd.DatetimeIndex(forecast['Date']))
        forecast = forecast.drop(["Date"], axis = 1)
        fig = px.bar(forecast, text_auto='.2s', height=800, width = 1200)
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig = go.Figure(data=[go.Bar(x= forecast.index, y=forecast.iloc[:,0],
            hovertext=["Requirements","Requirements","Requirements","Requirements",
            "Requirements","Requirements","Requirements","Requirements","Requirements",
            "Requirements","Requirements","Requirements"])])
        fig.update_layout(
        title='Forecast',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Number of Requirements: ',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
        )
        
        st.plotly_chart(fig)

@st.cache(ttl=24*60*60 )
def get_citynames():
    return df.CityName.unique()
@st.cache
def get_jobtitles(ttl=24*60*60):
    return df.JobTitleText.unique()
@st.cache
def get_clientnames(ttl=24*60*60):
    return df.ClientName.unique()
@st.cache
def get_remotedata(ttl=24*60*60):
    return df

       
if __name__ == '__main__':
    
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .reportview-container .main footer {visibility: hidden;}
    </style>
    
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    tab = st.sidebar.selectbox("What do you want to Search ?" ,
                               ("Home","Remote Jobs","Cities", "Job Titles", "Rangam Clients"))
    try:
        df = fetch_data()
    except:
        st.warning("Unable to Fetch Data, Please Contact Rangam")
    else:
        #with tab1:
            
        if tab == "Home":
            metric = st.columns(1)
            
            #with metric:
            st.title("Status for Remote Jobs")
            with st.expander("Status for Remote Jobs"):
                
                st.metric("Remote Jobs","{0}%".format(get_metric_remote(df)) )
        #top_cities = st.columns(1)
        
        #with top_cities:
            st.title("Top Cities")
            with st.container():
                st.map(get_top_cities(df))
                
                
            top_clients, top_jobs = st.columns((1,1))
            
            with top_clients:
                st.title("Top Clients")    
                with st.expander("Top Clients"):
                    
                    st.plotly_chart(get_top_clients(df))
                    st.info("Use the Clients bar to search for the Requirement Forecasting")
                
            with top_jobs:
                st.title("Top Jobs")
                with st.expander("Top Jobs"):
                    st.plotly_chart(get_top_job_titles(df))
                    st.info("Use the JobTitles bar to search for the Requirement Forecasting")
                
        if tab == "Cities":
            city_name = str()
            st.title("Job Forecasting Based on Cities")
            city_names = list(get_citynames())
            city_names.append("Select")
            city_name = st.selectbox('Select City name: ', city_names, index = city_names.index("Select"))
            #city_name = st.text_input("Enter City Name", value.strip())
            st.write("Selected Option",city_name)
            create_city_plot(df, city_name)
            with st.expander("Check Top Job Requirements for these Cities"):
                get_text(get_titles_cities(df, city_name))
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
            with st.expander("See explanation for Job Forecasting For Cities"):
                create_prophet_city_plot(df, city_name)
        
        if tab == "Job Titles":
        #with tab2:
            job_title = str()
            st.title("Job Forecasting Based on Job Titles: ")  
            job_titles = list(get_jobtitles())   
            job_titles.append("Select")         
            job_title = st.selectbox('Select JobTitle: ', job_titles, index = job_titles.index("Select"))
            #job_title = st.text_input("Enter Job Title", value.strip())
            st.write("Selected Option",job_title)
            create_job_title_plot(df, job_title)
            with st.expander("Check Top Job Requirements for these Cities"):   
                 get_text(get_clients_titles(df, job_title))
                 st.info("Use the Clients bar to search for the Requirement Forecasting")
            with st.expander("See explanation for Job Forecasting For Job Titles"):
                 create_prophet_jobtitle_plot(df, job_title)
            
        if tab == "Rangam Clients":
        #with tab3:
            client_name = str()
            st.title("Job Forecasting Based on Rangam Clients: ")
            client_names = list(get_clientnames())
            client_names.append("Select")
            client_name = st.selectbox('Select Rangam Clients', client_names, index = client_names.index("Select"))
            #client_name = st.text_input("Enter Rangam Clients", value.strip())
            st.write("Selected Option",client_name)
            create_client_plot(df, client_name)
            with st.expander("Check Job Requirements for these Clients"):
                get_text(get_titles_clients(df, client_name))
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
            with st.expander("See explanation for Job Forecasting For Rangam Clients"):
                 create_prophet_client_plot(df, client_name)
                 
        
        if tab == "Remote Jobs":
            #with tab4:
            
            job_type = "Remote Jobs"
            #job_types = ["Remote Jobs", "Select"]
            #job_type = st.selectbox('Select Rangam Clients', job_types, index = job_types.index("Select"))
            #st.write("Selected Option",job_type)
            #if job_type == "Remote Jobs"
            
            st.session_state.load_state = False
            chrt,mtrc  = st.columns([4,1])
            #with chrt:
            #with mtrc:    
            #st.metric("Remote Jobs","{0}%".format(get_metric_remote(df, job_type)) )
            st.title("Status for Work From Home Jobs")

            create_remote_plot(get_remotedata(), job_type)
            
            with st.expander("Check Top Clients and Job Titles for Remote Jobs"): 
                titles, citynames = st.columns(2)
                with titles:
                    st.title("Top Jobs:")
                    get_text(get_title_remote(df, job_type))
                    st.info("Use the JobTitles bar to search for the Requirement Forecasting")
                with citynames: 
                    st.title("Top Clients:")
                    get_text( get_client_remote(df, job_type))
                    st.info("Use the Clients bar to search for the Requirement Forecasting")
            with st.expander("See explanation for Remote Jobs"):
                 create_prophet_remote_plot(get_remotedata(), job_type)
            
                     

    
    



