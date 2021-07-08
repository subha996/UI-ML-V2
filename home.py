import streamlit as st


def app():

    # function that will change the width of the web page
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
    st.markdown("<h1 style='text-align: center;'>🤖 UI-ML 🤖™</h1>", unsafe_allow_html=True)
    
    st.write("<h5 style='text-align: center;'>🎈Create Machine Learning Model by just `Clicking`🎈</h5>", unsafe_allow_html=True)
    # JUST TAKING FOR SOME SPACE
    st.markdown("<h1 style='text-align: center;'></h1>", unsafe_allow_html=True)


    rep1 = """Hi, here you have to upload the dataset also can choose from the sample dataset, and then you can build an ml model by 
    just clicking without writing a single line of code. You will be able to tune the hyperparameter for the algorithms, 
    there are 4 algorithms for both `Regression` and `Classification`. """
   
    st.write(rep1)
    rep2 = """ 🛠 Experiment through algorithms that gives you a better result.

    For the complete and full reading go to the `About & Detailed Project Report` 
    where I briefly describe the project and how to use it. also, you can find `Video Explanation` of the project there."""
    st.write(rep2)
    
    # writng about data profiling
    df_pr = """ Here you can perform certain things like exploring the dataset, and get a 
    complete understanding of your data through visualizing the dataset in every aspect got to the 
    `Data Profiling` page for more."""
    st.write("<h3 style='text-align: center;'>📊 Data Profiling</h3>", unsafe_allow_html=True)
    st.write(df_pr)

    # writing about regression
    reg = """ After becoming with your data you can now build a machine learning model. 
    also you can perform the data preprocessing steps here. like categorical label encoding and scaling the numerical values.
    got to the `Regression` page if you want to create a regression model."""
    st.write("<h3 style='text-align: center;'>🏂 Regression </h3>", unsafe_allow_html=True)
    st.write(reg)

    # writing about classification
    clf = """ Now if you are up to the classification problem today go to `Classificatin` where you can build a classification model."""
    st.write("<h3 style='text-align: center;'>⛹️‍♂️ Classification</h3>", unsafe_allow_html=True)
    st.write(clf)


    # JUST TAKING FOR SOME SPACE
    st.markdown("<h1 style='text-align: center;'>--------------------------------------------------------------</h1>", unsafe_allow_html=True)
    # bye
    st.write("If you want to read more about the project and get complete knowledge about it got to `About & Detailed Project Report`.")
    st.write("<h4 style='text-align: center;'>subhanath91@gmail.com</h4>", unsafe_allow_html=True)

