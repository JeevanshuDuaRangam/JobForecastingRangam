# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 06:41:46 2022

@author: Jeevanshudua
"""

import pyodbc
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

"""
server = '192.168.5.29' 
database = 'sourcepros_07112021' 
username = 'jeevanshu' 
password = 'mar@2022' 
connection_string = 'DRIVER={ODBC Driver 11 for SQL Server}; SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password

@st.cache(ttl=24*60*60)
def fetch_data(connection_string = connection_string):
    cnxn = pyodbc.connect(connection_string)
    Query = "
    SELECT DISTINCT CM.ClientName, RM.RequirementID, RM.CreatedDate, CAT.CategoryName, Job.JobTitleText, jbt.JobTypeText, RM.IsRemoteLocation,
    ZCM.ZIPCode,CIM.CityName, sm.StateName
    FROM DBO.RequirementMaster RM WITH (NOLOCK)
    LEFT OUTER JOIN RequirementLocation RL WITH (NOLOCK) ON RM.RequirementID = RL.RequirementID
    LEFT OUTER JOIN AddressMASter AM WITH (NOLOCK) ON RL.AddressID = AM.AddressID
    LEFT OUTER JOIN ZIPCodeMASter ZCM WITH (NOLOCK) ON AM.ZIPCodeID = ZCM.ZIPCodeID
    LEFT OUTER JOIN CityMASter CIM WITH (NOLOCK) ON ZCM.CityID = CIM.CityID
    LEFT OUTER JOIN StateMASter SM WITH (NOLOCK) ON CIM.StateID = SM.StateID
    LEFT JOIN RequirementUserTxn rmTxn WITH (NOLOCK) ON rmTxn.RequirementID = RM.RequirementID
    LEFT OUTER JOIN ClientMaster CM WITH (NOLOCK) ON CM.ClientID = RM.ClientID
    LEFT OUTER JOIN MSPMaster msp With(NOLOCK) on msp.MSPMasterid = cm.MSPMasterid
    LEFT OUTER JOIN CategoryMaster CAT With(NOLOCK) on CAT.CategoryId = RM.CategoryId
    LEFT OUTER JOIN ClubMaster Club With(NOLOCK) on CAT.Clubid = Club.Clubid
    LEFT OUTER JOIN JobTitleMaster job WITH(NOLOCK) on job.JobTitleId = RM.JobTitleId
    LEFT OUTER JOIN JobTypeMaster jbt WITH(NOLOCK) on jbt.JobTypeId = RM.JobTypeId
    WHERE CAST(RM.CreatedDate as date) between cast('2014-01-01' as date) and cast('2022-06-30' as date)
    "
    #cursor.execute(Query).fetchall()
    df = pd.read_sql_query(Query, cnxn)
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], format="%Y-%m")
    df['Date'] = df['CreatedDate'].map(lambda x: '{}-{}'.format(x.year, x.month))
    
    return df
"""


def convert_negative(num):
    if num<0:
        return 0
    else:
        return num
    
@st.cache(ttl=24*60*60)
def fetch_data():

    df = pd.read_csv(r"job_forecasting.csv")
    
    return df

#@st.cache
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
        st.write("Not Enough Data Found")
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
        st.write("Score: The lesser the better the projection.")
        rmse = np.sqrt(mse)
    
        #return fig_1, fig_2, rmse
        st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2), st.write("RMSE:",rmse)

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
        st.write("Not Enough Data Found")
    else:
        
        forecast = model.make_future_dataframe(12, freq = "M")  
        forecast = model.predict(forecast.tail(12))
        forecast = forecast[['ds', 'yhat']]
        forecast["yhat"] = forecast["yhat"].apply(lambda x: convert_negative(x))
        forecast["yhat"] = forecast["yhat"].apply(int)
        forecast.columns = ["Date", "Requirements"]
        forecast = forecast.set_index(pd.DatetimeIndex(forecast['Date']))
        forecast = forecast.drop(["Date"], axis = 1)
        fig = px.bar(forecast, text_auto='.2s', height=800, width = 1000)
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
    