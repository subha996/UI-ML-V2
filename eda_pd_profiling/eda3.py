import numpy as np
import pandas as pd
# from pydantic.errors import NoneIsAllowedError
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



def app():

    # function for getting wider space on web page
    def _max_width_():
                max_width_str = f"max-width: 950px;"
                st.markdown(
                    f"""
                <style>
                .reportview-container .main .block-container{{
                    {max_width_str}
                }}
                </style>    
                """,
                    unsafe_allow_html=True,
                )
    # calling the function for full page
    _max_width_()

    # writng some on header part
    st.write("<h2 style='text-align: center;'>📈📊📉 EDA With Pandas Profiling 📉📊📈</h2>", unsafe_allow_html=True)
    # Web App Title
    st.markdown('''
    This is the **EDA App** created using the **pandas-profiling** library [Knoe More About It](https://github.com/pandas-profiling/pandas-profiling).
    ''')






    # asking for file
    file_upload = st.sidebar.file_uploader('Upload your Data Her', type=['csv'], help="Only `csv` Please")
    name = st.sidebar.selectbox('Selcect Sample Data', options=['None','Forbes Richest Atheletes', 'IT Salary Survey EU  2020'], help='Select Data From Here')

    # smple file getting function
    def get_dataset(name, sample=True, custome=False ):
        try:
            if sample:
                if name=='Forbes Richest Atheletes': # matchin user choose file 
                    df = pd.read_csv("ed_data\Forbes Richest Atheletes.csv")
                    return  df # retruning the data frame
                elif name == 'IT Salary Survey EU  2020': # 
                    df = pd.read_csv('ed_data\IT Salary Survey EU  2020.csv')
                    return df
            if custome:
                df = pd.read_csv(file_upload)
                return df
        except Exception:
            print('error n load_dataset')

    # giving result by choosing dataset, custome or sample
    if file_upload is None:
        df = get_dataset(name=name) 
    else:
        df = get_dataset(name=name,custome=True, sample=False)

    if df is not None:
        if st.sidebar.checkbox('Show Data', value=True):
            st.write('Your Uploaded Data')
            
            st.dataframe(df)
    else:
        st.warning('Upload a data first')


    if df is not None:
        if st.sidebar.button('Create Report', help='Click this to create the report.  \nThis can take some time depending upon the size of file.'):
            pr = ProfileReport(df, explorative=True)
            st.markdown("<h2 style='text-align: center;'>⛩ Report ⛩</h2>", unsafe_allow_html=True)
            st_profile_report(pr)
    else:
        st.warning('Please Upload a dataset or choose form sample data.')
