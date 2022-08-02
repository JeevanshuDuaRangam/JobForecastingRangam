# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 02:10:12 2022

@author: Jeevanshudua
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 07:00:38 2022

@author: Jeevanshudua
"""



import pandas as pd
import streamlit as st
import numpy as np
from prophet import Prophet
#import plotly.express as px
import plotly.graph_objects as go


def convert_negative(num):
    if num<0:
        return 0
    else:
        return num

@st.cache(ttl=24*60*60)
def fetch_data():
    import pandas as pd
    df = pd.read_csv(r"job_forecasting.csv")
    
    return df
#Home Page

#Get Top Categories
def get_metric_category(df):
    new_df = df.groupby(["CategoryName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df[["CategoryName", "RequirementID"]]
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    return create_bar_chart(new_df.CategoryName, new_df.RequirementID)
    
#Get Top Cities
def get_top_cities(df):
    new_df = df.groupby(["CityName", "Latitude", "Longitude"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.drop(["CityName","RequirementID"], axis = 1)
    new_df.columns = ["lat", "lon"]
    return new_df

#Get Top Cities
def get_category_by_title(df, job_title):
    new_df = df[["CategoryName","RequirementID"]][df["JobTitleText"] == job_title]
    new_df = new_df.groupby("CategoryName").sum()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    return create_bar_chart(new_df.CategoryName, new_df.RequirementID)


#Get Top Job Titles
def get_top_job_titles(df, low, high):
    new_df = df.groupby(["JobTitleText"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "JobTitleText")

#Get Top Job Clients
def get_top_clients(df,low, high):
    new_df = df.groupby(["ClientName"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "ClientName")
    
#Get Titles based on Cities
def get_titles_cities(df, city_name,low,high):
    new_df = df.groupby(["Date", "CityName", "JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "CityName", "JobTitleText", "RequirementID"]]
    new_df = new_df[new_df["CityName"]==city_name]
    new_df = new_df.groupby(["JobTitleText"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "JobTitleText" )

#Get Job Title based on Clients
def get_titles_clients(df, client_name,low,high):
    new_df = df.groupby(["Date", "ClientName", "JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "ClientName", "JobTitleText", "RequirementID"]]
    new_df = new_df[new_df["ClientName"]==client_name]
    new_df = new_df.groupby(["JobTitleText"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "JobTitleText" )

#Get Clients based on Job Titles
def get_clients_titles(df, job_title,low,high):
    
    new_df = df.groupby(["Date","JobTitleText", "ClientName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "ClientName", "JobTitleText", "RequirementID"]]
    new_df = new_df[new_df["JobTitleText"]==job_title]
    new_df = new_df.groupby(["ClientName"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "ClientName" )

#Get Title based on Clients
def get_cetegory_clients(df, client_name):
    new_df = df.groupby(["Date", "ClientName", "CategoryName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.loc[:, ["Date", "ClientName", "CategoryName", "RequirementID"]]
    new_df = new_df[new_df["ClientName"]==client_name]
    new_df = new_df.groupby(["CategoryName"]).count()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    new_df = new_df[["CategoryName", "RequirementID"]]
    return create_bar_chart(new_df.CategoryName, new_df.RequirementID)
    

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

    
def create_pie_chart(df, values, names, width=600, height=600):
    labels = df[names]
    values = df[values]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial', textposition='outside', hole=.5
                            )])
    
    fig.update_layout(
    showlegend = False,
    autosize=False,
    width=width,
    height=height)
    
    return fig

def create_bar_chart(x, y):
  
    fig = go.Figure(data=[go.Bar(x= x, y=y,
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
    return fig
     
def evaluate_model(data):   
    data = data.iloc[:,1]
    data = pd.DataFrame(data)
    data = data.reset_index()
    data.columns = ["ds", "y"]
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
    
        return st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse), st.caption('RMSE Score is lower the better.')



def get_text(lis):    
    s = ''
    for i in lis:
        s += "- " + i + "\n" 
    return st.markdown(s)


def create_forecast(data):
    data = data.iloc[:,1]
    data = pd.DataFrame(data)
    data = data.reset_index()
    data.columns = ["ds", "y"]
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
        
        return st.plotly_chart(fig)

@st.cache(ttl=24*60*60)
def get_citynames():
    return df.CityName.unique()
@st.cache(ttl=24*60*60)
def get_jobtitles():
    return df.JobTitleText.unique()
@st.cache(ttl=24*60*60)
def get_clientnames():
    return df.ClientName.unique()
@st.cache(ttl=24*60*60)
def get_remotedata():
    return df
@st.cache(ttl=24*60*60)
def get_categories():
    return df.CategoryName.unique()

       
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
                               ("Home","Cities", "Job Titles", "Rangam Clients"))
    try:
        df = fetch_data()
    except:
        st.warning("Unable to Fetch Data, Please Contact Rangam")
    else:
               
        if tab == "Home":
            
            with st.expander("Status"):
                
                st.plotly_chart(get_metric_category(df))
                
        
            with st.expander("Top Cities"):
            #with st.container():
                st.map(get_top_cities(df), zoom = 1)

          
            with st.expander("Top Clients"):
            #with st.container():
                range_clients = st.slider('Select a range of Top Clients',0, 100, (0,10))
                st.write('Range:',range_clients)
                st.plotly_chart(get_top_clients(df,int(range_clients[0]), int(range_clients[1])))
                st.info("Use the Clients bar to search for the Requirement Forecasting")

            with st.expander("Top Jobs"):
            #with st.container():
                values = st.slider('Select a range of Top Jobs',0, 100, (0,10))
                st.write('Range:', int(str(values[0])), int(str(values[1])))
                st.plotly_chart(get_top_job_titles(df, int(str(values[0])), int(str(values[1]))))
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
                
        
        if tab == "Cities":
            city_name = str()
            st.title("Job Forecasting Based on Cities")
            city_names = list(get_citynames())
            city_names.append("Select")
            city_name = st.selectbox('Select City name: ', city_names, index = city_names.index("Select"))
            st.write("Selected Option",city_name)
            create_city_plot(df, city_name)
            st.title(("Check Top Job Requirements for these Cities"))
            with st.container():
                values = st.slider('Select a range of Top Jobs in the City',0, 100, (0,10))
                st.write('Range:',  int(str(values[0])), int(str(values[1])))
                st.plotly_chart(get_titles_cities(df, city_name,  int(str(values[0])), int(str(values[1]))))
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
            with st.expander("See explanation for Job Forecasting For Cities"):
                create_prophet_city_plot(df, city_name)
        
        if tab == "Job Titles":
            job_title = str()
            st.title("Job Forecasting Based on Job Titles: ")  
            job_titles = list(get_jobtitles())
            job_titles.append("Select")         
            job_title = st.selectbox('Select JobTitle: ', job_titles, index = job_titles.index("Select"))
            st.write("Selected Option",job_title)
            create_job_title_plot(df, job_title)
            st.title("Check Top Job Requirements for these Cities")
            with st.container():
                 values = st.slider('Select a range of Top Clients for the Jobs',0, 100, (0,10))
                 st.write('Range:',  int(str(values[0])), int(str(values[1])))
                 st.plotly_chart(get_clients_titles(df, job_title,  int(str(values[0])), int(str(values[1]))))
                 st.info("Use the Clients bar to search for the Requirement Forecasting")
                 
            with st.expander("Analyze the Requirements based on Rangam Category"):    
                 st.plotly_chart(get_category_by_title(df, job_title))
            with st.expander("See explanation for Job Forecasting For Job Titles"):
                 create_prophet_jobtitle_plot(df, job_title)
            
        if tab == "Rangam Clients":
        
            client_name = str()
            st.title("Job Forecasting Based on Rangam Clients: ")
            client_names = list(get_clientnames())
            client_names.append("Select")
            client_name = st.selectbox('Select Rangam Clients', client_names, index = client_names.index("Select"))
            st.write("Selected Option",client_name)
            create_client_plot(df, client_name)
            st.title("Check Job Requirements for these Clients")
            with st.container():
                values = st.slider('Select a range of Top Jobs for the Clients',0, 100, (0,10))
                st.write('Range:',  int(str(values[0])), int(str(values[1])))
                st.plotly_chart(get_titles_clients(df, client_name, int(str(values[0])), int(str(values[1]))))
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
            with st.container():
                st.plotly_chart(get_cetegory_clients(df, client_name))
            
            with st.expander("See explanation for Job Forecasting For Rangam Clients"):
                 create_prophet_client_plot(df, client_name)
                 
        

            
                     

    
    


