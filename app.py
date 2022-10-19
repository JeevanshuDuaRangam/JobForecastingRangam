# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 02:10:12 2022

@author: Jeevanshudua
"""




import pandas as pd
import streamlit as st
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import date





def convert_negative(num):
    if num<0:
        return 0
    else:
        return num

@st.cache(ttl=24*60*60)
def fetch_data():
    """
    

    Parameters
    ----------
    connection_string : TYPE, optional
        DESCRIPTION. The default is connection_string.

    Returns
    -------
    df : DATAFRAME
        Returns the Dataframe for Job Forecasting.

    """
    
    
    df = pd.read_csv(r"job_forecasting.csv")
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], format="%Y-%m")
    df['Date'] = df['CreatedDate'].map(lambda x: '{}-{}'.format(x.year, x.month))
    
    return df

#Get Top Categories
def get_metric_category(df):
    """
    
    Parameters
    ----------
    df : Dataframe
        Job Forecasting Dataframe.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    new_df = df.groupby(["CategoryName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df[["CategoryName", "RequirementID"]]
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    return create_bar_chart(new_df.CategoryName, new_df.RequirementID)
    
#Get Top Cities
def get_top_cities(df):
    """
    
    Parameters
    ----------
    df : Dataframe
        Job Forecasting Requirement.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    new_df = df.groupby(["CityName", "Latitude", "Longitude"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.drop(["CityName","RequirementID"], axis = 1)
    new_df.columns = ["lat", "lon"]
    return new_df

#Get Top Cities
def get_category_by_title(df, job_title):
    """
    

    Parameters
    ----------
    df : Dataframe
        Job Forecasting Dataframe.
    job_title : String
        Pass the Job Title Directly by from thr Job Title column.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    new_df = df[["CategoryName","RequirementID"]][df["JobTitleText"] == job_title]
    new_df = new_df.groupby("CategoryName").sum()
    new_df = new_df.sort_values(by = "RequirementID", ascending = False)
    new_df = new_df.reset_index()
    return create_bar_chart(new_df.CategoryName, new_df.RequirementID)


#Get Top Job Titles
def get_top_job_titles(df, low, high):
    """


    Parameters
    ----------
    df : Dataframe
        Job Forecasting Dataframe.
    low : int
        Lowest value for Top Job Titles.
    high : int
        Lowest value for Top Job Titles.

    Returns
    -------
    Piechart on the Streamlit App. 

    """
    
    new_df = df.groupby(["JobTitleText"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "JobTitleText")

#Get Top Job Clients
def get_top_clients(df,low, high):
    """


    Parameters
    ----------
    df : Dataframe
        Job Forecasting Dataframe.
    low : int
        Lowest value for Top Job Titles.
    high : int
        Lowest value for Top Job Titles.

    Returns
    -------
    Piechart on the Streamlit App. 

    """
    new_df = df.groupby(["ClientName"]).count()
    new_df = new_df.loc[:, ["RequirementID"]]
    new_df = new_df.sort_values(ascending = False, by = "RequirementID")
    new_df = new_df.reset_index()
    new_df = new_df.iloc[low:high,:]
    return create_pie_chart(new_df, "RequirementID", "ClientName")
    
#Get Titles based on Cities
def get_titles_cities(df, city_name,low,high):
    
    """

    Parameters
    ----------
    df : Dataframe
        Job Forecasting Dataframe.
    city_name : string
        City name from the Job Forecasting data.
    low : int
        Lowest value for Top Job Titles.
    high : int
        Lowest value for Top Job Titles.

    Returns
    -------
    Piechart on the Streamlit App. 

    """
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
    """

    Parameters
    ----------
    df : Dataframe
        Job Forecasting Dataframe.
    client_name : string
        Client name from the Job Forecasting data.
    low : int
        Lowest value for Top Job Titles.
    high : int
        Lowest value for Top Job Titles.

    Returns
    -------
    Piechart on the Streamlit App. 

    """
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
    """

    Parameters
    ----------
    df : Dataframe
        Job Forecasting DataFrame.
    job_title : string
        City name from the Job Forecasting
    low : int
        Lowest value for Top Job Titles.
    high : int
        Lowest value for Top Job Titles.

    Returns
    -------
    Piechart on the Streamlit App. 

    """
    
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
def get_category_clients(df, client_name):
    
    """
    
    Parameters
    ----------
    df : Job Forecasting Dataframe
        DESCRIPTION.
    client_name : string
        Client name from the Job Forecasting dataframe column.  
        

    Returns
    -------
    Forecast chart on Streamlit Application

    """
    
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
    

def create_city_plot(df, city_name, use = "forecast" ):
    """ 

    Parameters
    ----------
    
    
    df : Job Forecasting Dataframe
        DESCRIPTION.
    
    city_name : string
        City name from the Job Forecasting dataframe.
    use : string, optional
        forecast for Forcating and evaluate for Model Evaluation.
        The default is "forecast", and other is "evalaute".
        

    Returns
    -------
    For "forecast" returns the forecast chart.
    For "evaluate" returns the evaluate model.

    """
    
    new_df = df.groupby(["Date", "CityName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df[['CityName', 'RequirementID']]
    data = new_df[new_df["CityName"]==city_name]     
    if use == "forecast":
        return create_forecast(data) 
    if use == "evaluate":
        return evaluate_model(data)  

 
def create_job_title_plot(df, job_title, use = "forecast" ):
    """
    Parameters
    ----------
    
    
    df : Job Forecasting Dataframe
        DESCRIPTION.
    
    job_title : string
        Job Title from the Job Forecasting dataframe.
    use : string, optional
        forecast for Forcating and evaluate for Model Evaluation.
        The default is "forecast", and other is "evalaute".
        

    Returns
    -------
    For "forecast" returns the forecast chart.
    For "evaluate" returns the evaluate model.

    """
    new_df = df.groupby(["Date","JobTitleText"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df[['JobTitleText', 'RequirementID']]
    data = new_df[new_df["JobTitleText"]==job_title]
    if use == "forecast":
        return create_forecast(data) 
    if use == "evaluate":
        return evaluate_model(data)
    

def create_client_plot(df, client_name, use = "forecast" ):
    """

    Parameters
    ----------
    
    
    df : Job Forecasting Dataframe
        DESCRIPTION.
    
    client_name : string
        Client name from the Job Forecasting dataframe.
    use : string, optional
        forecast for Forcating and evaluate for Model Evaluation.
        The default is "forecast", and other is "evalaute".
        

    Returns
    -------
    For "forecast" returns the forecast chart.
    For "evaluate" returns the evaluate model.

    """
    
    new_df = df.groupby(["Date", "ClientName", "CategoryName"]).count()
    new_df = new_df.reset_index()
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df =new_df.groupby([new_df.index,"ClientName"]).sum().reset_index()
    new_df = new_df.rename(columns = {new_df.columns[0]:"Date"} )
    new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y-%m")
    new_df = new_df.set_index(pd.DatetimeIndex(new_df['Date']))
    new_df = new_df[['ClientName', 'RequirementID']]
    data = new_df[new_df["ClientName"]== client_name]
    if use == "forecast":
        return create_forecast(data) 
    if use == "evaluate":
        return evaluate_model(data)
    

    
def create_pie_chart(df, values, labels, width=600, height=600):
    """
    

    Parameters
    ----------
    df : dataframe
        Job Forecasting dataframe.        
    values : string
        column name to be used as values for the labels for Pie Charts.
    labels : string
        column name to be used as labels for Pie Charts. 
        
    width : int, optional
        DESCRIPTION. The default is 600.
    height : int, optional
        DESCRIPTION. The default is 600.

    Returns
    -------
    fig : plotly chart
        

    """
    
    labels = df[labels]
    values = df[values]
    total = values.sum()
    st.write("Total",str(total)," Requirements that were found.", )
    
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
    """
    

    Parameters
    ----------
    x : int
        values to be shown on x-axis. In the case of foracasting: Dates.
    y : int
        values to be shown on y-axis. In the case of forecasting: Requirements.

    Returns
    -------
    fig : bar chart figure.

    """
    
  
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
    bargroupgap=0.1, # gap between bars of the same location coordinate.
    width=670,
    height=450)
    return fig
     
def evaluate_model(data):   
    """

    Parameters
    ----------
    data : dataframe
        Data should be in a Data as index format and one column for the forecast values.

    Returns
    -------
    rmse : root mean square error 
        Error Matrix to decide forecasting result.
    forecast_plot : Component Plot
        forecast plot for evaluation

    """
    
    data = data.iloc[:,1]
    data = pd.DataFrame(data)
    data = data.reset_index()
    data.columns = ["ds", "y"]
    model = Prophet()
    try:
        model.fit(data)
    except ValueError:
        st.warning("{} requirement(s) are not enough for forecasting".format(str(len(data))))
    else:
    
        forecast = model.make_future_dataframe(52, freq = "W")  
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
    
        return  st.write("RMSE:",rmse), st.caption('RMSE Score is lower the better.'),st.write("Forecast Plot: ",fig_1) ,st.write("Component Plot: ",fig_2)





def create_forecast(data):
    """
    

    Parameters
    ----------
    data : dataframe
        Job Forecasting dataframe.

    Returns
    -------
    fig : plotly figure for the Forecast directly Streamlit App.        

    """
    
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
        today = date.today()
        forecast = model.make_future_dataframe(52, freq = "W")  
        forecast = model.predict(forecast.tail(52))
        forecast = forecast[['ds', 'yhat']]
        forecast["yhat"] = forecast["yhat"].apply(lambda x: convert_negative(x))
        forecast["yhat"] = forecast["yhat"].apply(int)
        forecast.columns = ["Date", "Requirements"]
        
        forecast = forecast.set_index(pd.DatetimeIndex(forecast['Date']))
        forecast = forecast.drop(["Date"], axis = 1)
        latest = forecast.head(1).index.date[0]
        diff = today - latest
        diff = diff.days
        st.write("Days past from latest Requirement.", str(diff), "days")
        fig = go.Figure(data=[go.Bar(x= forecast.index, y=forecast.iloc[:,0],
            hovertext=["Requirements"]*52)])
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
        #return fig
        return st.plotly_chart(fig, use_container_width=True)

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
    def add_logo():
        st.markdown(
        """
        <style>
    header.css-18ni7ap {
        background: #fff url('http://dev.talentarbor.com/mail-images/TalentArbor.png');
        background-repeat: no-repeat;
        background-position: 28rem 10px;
        height: 5rem;
        box-shadow: 0px 1px 10px 0px rgb(204 204 204 / 50%);
    }
    .css-fg4pbf {
        background: #f6f6f6;
    }
    /* Class name changed - 09-23-2022 */ .stTabs {
        background: #fff;
        padding: 15px;
        gap: 0rem;
        border-radius: 8px;
        box-shadow: 1px 1px 4px 4px rgba(204,204,204,0.2);
    }
    .st-cf {
        font-size: 16px;
    }
    .st-ch {
        padding: 2px 10px;
    }
    .st-c4 {
        font-weight: 600;
    }
    .st-dg {
        font-weight: 600;
    }
    .st-d0 {
        border-width: 2px;
    }
    .css-15tx938 {
        font-size: 16px;
    }
    /* 09-23-2022 CSS changes starts */
    header.css-18ni7ap {
        background-position: 10px 10px;
        width: 704px;
        margin: 0 auto;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        left: -6px;
    }
    @media (max-width: 741px) {
        header.css-18ni7ap {
            width: 100%;
            border-bottom-left-radius: 0px;
            border-bottom-right-radius: 0px;
            left: 0px;
        }
    }
    /* 09-23-2022 CSS changes ends */
    </style>
        """,unsafe_allow_html=True)
    add_logo()
    tab1, tab2, tab3, tab4  = st.tabs(["Home","Cities", "Job Titles", "Clients"])
    #tab = st.sidebar.selectbox("What do you want to Search ?" ,("Home","Cities", "Job Titles", "Rangam Clients"))
    try:
        df = fetch_data()
    except:
        st.warning("Unable to Fetch Data, Please Contact Rangam")
    else:
        with tab1:     
        #if tab == "Home":
            with st.expander("Locations", expanded = True):
                with st.container():
                    st.map(get_top_cities(df), zoom = 1)
                    
            with st.expander("Status"):
                with st.container():
                    st.plotly_chart(get_metric_category(df), use_container_width=True)
                        
            with st.expander("Clients"):
            #with st.container():
                range_clients = st.slider('Select a range of Top Clients',0, 25, (0,10))
                st.plotly_chart(get_top_clients(df,int(range_clients[0]), int(range_clients[1])), use_container_width=True)
                st.info("Use the Clients bar to search for the Requirement Forecasting")

            with st.expander("Jobs"):
            #with st.container():
                values = st.slider('Select a range of Top Jobs',0, 25, (0,10))
                st.plotly_chart(get_top_job_titles(df, int(str(values[0])), int(str(values[1]))), use_container_width=True)
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
                
        with tab2:
        #if tab == "Cities":
            city_name = str()
            st.title("Job Forecasting Based on Location")
            city_names = list(get_citynames())
            city_names.append("Select")
            city_name = st.selectbox('Select Location ', city_names, index = city_names.index("Select"))
            st.write("Selected Option:",city_name)
            create_city_plot(df, city_name)
            st.title(("Check Top Job Requirements for these Cities"))
            with st.container():
                values = st.slider('Select a range of Top Jobs in the City',0, 25, (0,10))
                st.plotly_chart(get_titles_cities(df, city_name,  int(str(values[0])), int(str(values[1]))), use_container_width=True)
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
            with st.expander("See explanation for Job Forecasting For Cities"):
                create_city_plot(df, city_name, use ="evaluate")
        
        with tab3:
        #if tab == "Job Titles":
            job_title = str()
            st.title("Job Forecasting Based on Job Titles: ")  
            job_titles = list(get_jobtitles())
            job_titles.append("Select")         
            job_title = st.selectbox('Select Job Title: ', job_titles, index = job_titles.index("Select"))
            st.write("Selected Option:",job_title)
            create_job_title_plot(df, job_title)
            st.title("Check Top job requirements for these Cities")
            with st.container():
                 values = st.slider('Select a range of Top Clients for the Jobs',0, 25, (0,10))
                 st.plotly_chart(get_clients_titles(df, job_title,  int(str(values[0])), int(str(values[1]))), use_container_width=True)
                 
                 st.info("Use the Clients bar to search for the Requirement Forecasting")
                 
            with st.expander("Analyze the Requirements based on Rangam Category"):    
                 st.plotly_chart(get_category_by_title(df, job_title), use_container_width=True)
            with st.expander("See explanation for Job Forecasting For Job Titles"):
                 create_job_title_plot(df, job_title, use = "evaluate")
            
        with tab4:
        #if tab == "Rangam Clients":
        
            client_name = str()
            st.title("Job Forecasting Based on Rangam Clients: ")
            client_names = list(get_clientnames())
            client_names.append("Select")
            client_name = st.selectbox('Select Rangam Clients', client_names, index = client_names.index("Select"))
            st.write("Selected Option:",client_name)
            create_client_plot(df, client_name)
            st.title("Check Job Requirements for these Clients")
            with st.container():
                values = st.slider('Select a range of Top Jobs for the Clients',0, 25, (0,10))
                st.plotly_chart(get_titles_clients(df, client_name, int(str(values[0])), int(str(values[1]))), use_container_width=True)
                st.info("Use the JobTitles bar to search for the Requirement Forecasting")
            with st.container():
                st.plotly_chart(get_category_clients(df, client_name), use_container_width=True)
            
            with st.expander("See explanation for Job Forecasting For Rangam Clients"):
                 create_client_plot(df, client_name, use = "evaluate")
                 
        

            
                     

    
    



