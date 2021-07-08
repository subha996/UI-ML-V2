# importing my apps
from multiapp import MultiApp
from Classification import clf1
from Regression import mlv2_V2
from eda_pd_profiling import eda3
import home, about
import streamlit as st


st.set_page_config(page_title='🤖UI ML🤖',page_icon="icon_rob.png" ,layout = 'wide', initial_sidebar_state = 'auto')

# creating multi app class instance
app = MultiApp()

# adding all apps
app.add_app('Home', home.app)
app.add_app('Data Profiling', eda3.app)
app.add_app('Regression', mlv2_V2.app)
app.add_app('Classification', clf1.app)
app.add_app('About & Detailed Project Report', about.app)

# running the current app which will run other apps.
app.run()
