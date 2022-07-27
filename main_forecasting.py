# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 07:00:38 2022

@author: Jeevanshudua
"""


import pandas as pd
import streamlit as st
from DataFetching import fetch_data, create_forecast,evaluate_model


#@st.cache(persist=True, allow_output_mutation=True,suppress_st_warning=True)
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
    

    
#@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
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
    #return st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse)

    

@st.cache(show_spinner=False,suppress_st_warning=True)
def create_city_plot(new_df, city_name):
    new_df = df.groupby(["Date", "CityName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText","IsRemoteLocation", "ClientName"], axis = 1)
    data = new_df[new_df["CityName"]==city_name]
    return create_forecast(data)

@st.cache(show_spinner=False, suppress_st_warning=True)
def create_prophet_city_plot(df, city_name):   
    new_df = df.groupby(["Date", "CityName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['Date',"CreatedDate",'ClientName','CategoryName', 'ZIPCode', 'StateName',"JobTitleText", "JobTypeText","IsRemoteLocation", "ClientName"], axis = 1)
    data = new_df[new_df["CityName"]== city_name]
    return evaluate_model(data) 
    #return st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse)


@st.cache(show_spinner=False,suppress_st_warning=True )     
def create_job_title_plot(df, job_title):
    new_df = df.groupby(["Date","JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['ClientName','Date','CreatedDate','CategoryName', 'ClientName','JobTypeText', 'IsRemoteLocation'], axis = 1)
    data = new_df[new_df["JobTitleText"]==job_title]
    return create_forecast(data)


@st.cache(show_spinner=False, suppress_st_warning=True)
def create_prophet_jobtitle_plot(df, job_title):
       
    new_df = df.groupby(["Date","JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df.drop(['ClientName','Date','CreatedDate','CategoryName', 'ClientName','JobTypeText', 'IsRemoteLocation'], axis = 1)
    data = new_df[new_df["JobTitleText"]==job_title]
    return evaluate_model(data) 
    #return st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse)


@st.cache(show_spinner=False,suppress_st_warning=True)
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
    

@st.cache(show_spinner=False, suppress_st_warning=True)
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
    #return st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse)

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
        

    tab = st.sidebar.selectbox("What do you want to Search ?" ,
                               ("Remote Jobs","Cities", "Job Titles", "Rangam Clients"))
    try:
        df = fetch_data()
    except:
        st.write("Unable to Fetch Data, Please Contact Rangam")
    else:
        #with tab1:
        if tab == "Cities":
            city_name = str()
            st.title("Job Forecasting Based on Cities")
            city_names = list(get_citynames())
            city_names.append("Select")
            city_name = st.selectbox('Select City name: ', city_names, index = city_names.index("Select"))
            #city_name = st.text_input("Enter City Name", value.strip())
            st.write("Selected Option",city_name)
            create_city_plot(df, city_name)
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
            with st.expander("See explanation for Job Forecasting For Rangam Clients"):
                 create_prophet_client_plot(df, client_name)
                 
        
        if tab == "Remote Jobs":
            #with tab4:
            st.title("Status for Work From Home Jobs")
            job_type = "Remote Jobs"
            #job_types = ["Remote Jobs", "Select"]
            #job_type = st.selectbox('Select Rangam Clients', job_types, index = job_types.index("Select"))
            #st.write("Selected Option",job_type)
            #if job_type == "Remote Jobs"
            st.session_state.load_state = False
            create_remote_plot(get_remotedata(), job_type)
            with st.expander("See explanation for Remote Jobs"):
                 create_prophet_remote_plot(get_remotedata(), job_type)
                     
    
    
    
     
    
    



