import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from Regression.split import Split
from sklearn.model_selection import train_test_split
from Regression.reg_algos import Models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import base64
import pickle
from warnings import filterwarnings
filterwarnings(action='ignore')




st.cache(suppress_st_warning=True)
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

    # seeting up title
    # st.title(' 🤖 UI ML 🤖')
    st.markdown("<h1 style='text-align: center;'>🤖 UI ML - Regression 🤖</h1>", unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center; color: white;'>🤖 UI ML - Regression 🤖</h1>", unsafe_allow_html=True)
    # user uploaded file
    st.sidebar.info('Please Upload Dataset or Use Sample Dataset')

    # asiking for custome file from local machine
    file_upload = st.sidebar.file_uploader('Upload your file', type=['csv'], help='Only `csv` Please 🥺')
    # getting the name of sample data set name
    name = st.sidebar.selectbox('Sample Datasets', ['None','california_housing', 'boston', 'diabetes'], help='Please select a sample dataset from here !!')

    # smple file getting function
    def get_dataset(sample=True, custome=False ):
        try:
            if sample:
                if name=='boston': # boston dataset
                    bos = datasets.load_boston()
                    df = pd.DataFrame(data = bos['data'], columns= bos['feature_names'])
                    df['Target'] = bos['target']
                    return  df.sample(frac=0.5) # returning 50% of total data for faster preprocessing
                elif name == 'california_housing': # carlifornia housing dataset
                    df, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)
                    df['Target'] = y
                    return df.sample(frac=0.5)
                elif name=='diabetes': # diabetes dataset
                    df, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
                    df['Target'] = y # adding target column
                    return df.sample(frac=0.5)
            if custome:
                df = pd.read_csv(file_upload)
                return df

        except Exception as ex:
            print(str(ex))

    # giving result by choosing dataset, custome or sample
    if file_upload is None:
        df = get_dataset() 
    else:
        df = get_dataset(custome=True, sample=False)

    # showing the data.
    show_data = st.sidebar.checkbox('Show Data', value=True)
    if show_data:
        st.write('Your Current Data: - ')
        st.write(df)

    # creating function to run algorithm
    def run_model(algo_c, preprocessor):
        try:
            algo = Models(X_train, y_train, X_test, y_test, preprocessor)
            if algo_c == 'Linear Regression':
                st.markdown('You Select `Linear Regression`. [Read More](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)')
                fit_intercept = col1.selectbox('fit_intercept', [True, False], help='Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).')
                normalize=col2.selectbox('normalize', [False, True], help="Want to use `normalize` or not")
                n_jobs = col1.selectbox('n_jobs', [1,4,-1], help="No of core will use to run the model, `-1` means all the core(much faster)")
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.lin_reg(fit_intercept, normalize, n_jobs)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning("Selcet Relevant Hyperparameter you are seeing this message because bad hyperparameters are used, try to read about it from above link     \nMaybe your data is in bad shape!! you need yo apply `Preprocessing checkbox`")
        
            elif algo_c == 'RandomForest Regressor':
                st.markdown("You Select `RandomForest Regressor`. [Read More](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)")
                n_estimators = col1.slider('n_estimator',10,500,100, help='The number of trees in the forest. e.g. Numbers of Decision Trees, Higher number leads high variance model(overfit)')
                criterion = col2.selectbox('criterion', ['mse', 'mae'], help='The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.')
                max_depth =  col1.slider('max_depth', 1,50, help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. higher value leads to high variance model(overfit).')
                min_sample_split = col2.slider('min_sample_split', 2,10,3, help='The minimum number of samples required to split an internal node or Decision trees')
                min_sample_leaf = col1.slider('min_sample_leaf', 1,10,1, help='The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.')
                max_features = col2.selectbox('max_features', ['auto', 'sqrt', 'log2'], help='The number of features to consider when looking for the best split')
                bootstrap = col1.selectbox('bootstrap', [True, False], help='Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.')
                max_samples = col2.slider('max_sample', 1,10,3, help='If bootstrap is True, the number of samples to draw from X to train each base estimator.')
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.rf_reg(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_sample_split=min_sample_split, min_sample_leaf=min_sample_leaf, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning('Selcet Relevant Hyperparameter (you are seeing this message because bad hyperparameters are used, try to read about it from above link)   \nMaybe your data is in bad shape!! you need yo apply `Preprocessing checkbox`')


            elif algo_c == 'Support Vector Regression':
                st.markdown('You Select `Support Vector Regression`. [Read More](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)')
                # asking for hyperparameter
                kernel = col1.selectbox('kernal', ['rbf', 'linear', 'poly', 'sigmoid'], help='Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’')
                degree = col2.slider('degree', 1,10,3, help= 'Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.')
                gamma = col1.selectbox('gamma', ['scale', 'auto'], help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.')
                coef0 = col2.number_input('coef0', help='Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.')
                C = col1.number_input('C', help = 'Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.')
                epsilon = col2.number_input('epsilon', help='Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.')
                shrinking = col1.selectbox('shrinking', [True, False], help='Read from [here](https://scikit-learn.org/stable/modules/svm.html#shrinking-svm)') 
                max_iter = col2.number_input('max_iter', help='Hard limit on iterations within solver, or -1 for no limit. set to -1 for initial')
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.svr_reg(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, shrinking=shrinking, max_iter=max_iter)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning('Selcet Relevant Hyperparameter. (you are seeing this message because bad hyperparameters are used, try to read about it from above link)    \nMaybe your data is in bad shape!! you need yo apply `Preprocessing checkbox`') # handeling bad choose of hyperparameter

            elif algo_c == 'KNeighbors Regressor':
                st.markdown('You Select `KNeighbors Regressor`. [Read More](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)')
                # asking for hyperparameter
                n_neighbors = col1.slider('n_neighbors', 1,10,5,help='Number of neighbors to use by default for kneighbors queries. more are [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor.kneighbors)')
                algorithm = col2.selectbox('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], help="Algorithm used to compute the nearest neighbors,'auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.'")
                leaf_size = col1.slider('leaf_size',1,100,30, help = 'Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.')
                p = col2.slider('p',2,10,2, help='Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.')
                try:
                    score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model = algo.knn(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p)
                    return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
                except Exception:
                    st.warning('Select Relevant Hyperparameter (you are seeing this message because bad hyperparameters are used, try to read about it from above link)     \nMaybe your data is in bad shape!! you need yo apply `Preprocessing checkbox`')
        except Exception as ex:
            print('run_model:' + str(ex))



    # starting data traning.
    try:
        if df is not None:
            df.dropna(axis=0, inplace=True) # droping missing value if present
            col = list(df)
            col.insert(0, 'None')
            split = Split(df)
            # creating two column layout
            col1, col2 = st.beta_columns(2)
            # creating alogorithm list
            algo_lst = ['Linear Regression', 'RandomForest Regressor', 'Support Vector Regression', 'KNeighbors Regressor']
            # taking target column to drop
            target = st.sidebar.selectbox('Select Target Column', options=col, help='This list contain columns names form your data.    \nSelect the target column from here.!!')
            # algo slider
            algo_chose = st.sidebar.selectbox('Select Algorithms', options=algo_lst, help='Select Algorithm that you want to use !!')
            # taking X and y from Split class
            try:
                X, y = split.X_and_y(target)
            except Exception:
                st.warning('👈👈👈 Please Chose Target Column from given list.')
            cat_col = X.select_dtypes(include=['object', 'category']).columns.tolist()
            num_col = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # numeric preprocess pipeline
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            # categorical feature transformer pipeline
            categorical_transformer = OneHotEncoder(drop='first', sparse=False)
            # now defining preprocessor
            if st.sidebar.checkbox('Apply Preproessing', value=False, help="If your data have some `object` type or `category` type columns then you have to `check` this box, because the machine learning model can not work with categorical values.   \nAfter clicking this box, there `OneHotEncoding` and `StanderdScaling` pipeline will create.  \n`NOTE:- ` Remember when you check this button that means you are downloading the full pipeline.   \nIf your data is clean and preprocessed then leave this checkbox `uncheck`"):
                preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_col),('cat', categorical_transformer, cat_col)])
            else:
                preprocessor = None # if user not check then no preprocessing steps will apply.
                
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            button = st.button('Train and Give Score', help='Click to train the model and get scores🛠')
            # data training process
            with st.spinner('Hold on while the model is training'):
                        score_train, mae_train, score_test,  mae_test, y_pred_train, y_pred_test, model = run_model(algo_chose,preprocessor)
    



            



            if button:
                import seaborn as sns
                sns.set()
                col1, col2 = st.beta_columns(2)
                col1.info('Traning Score: ' + str(round(score_train,3)))
                col2.info('Testing Score: '+ str(round(score_test,3)))
                col1.info('Traning error(RMSE): '+ str(round(mae_train,3)))
                col2.info('Testing mae(RMSE): ' + str(round(mae_test,3)))
                # plotting traning and testing points
                st.write("<h3 style='text-align: center;'>Training and Testing Prediction PLot</h3>", unsafe_allow_html=True)
                plt.figure(figsize=(15,15))
                plt.subplot(2,2,1)
                plt.scatter(y_train, y_pred_train, c='b')  # plotting traning predction curve
                plt.xlabel('Traning Label')
                plt.ylabel('Predicted on train data')
                plt.title('Traning PLot')
                plt.subplot(2,2,2)
                plt.scatter(y_test, y_pred_test, c='g') # ploting testing prediction curve
                plt.xlabel('Test Data label')
                plt.ylabel('Predicted Data on test data')
                plt.title('Testing Plot')
                # plotting residuals
                residual_train = (y_train - y_pred_train)
                residual_test = (y_test - y_pred_test)
                # plt.figure(figsize=(14,6))
                plt.subplot(2,2,3)
                sns.distplot(residual_train)
                plt.title('Training Residual')
                plt.subplot(2,2,4)
                sns.distplot(residual_test)
                plt.title('Testing Residual')
    
                st.set_option('deprecation.showPyplotGlobalUse', False) # disableing plotting error
                st.pyplot()
                st.balloons()
                # saving the model and generate model download link
                output_model = pickle.dumps(model)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="model.pkl">Download Trained Model .pkl File</a>'
                st.text('You can download the trained model from below link 🧲')
                st.markdown(href, unsafe_allow_html=True)

        else:
            st.info('Please Upload Dataset or Choose from Sample Data')
    except Exception as ex:
        print('model_train ' + str(ex))

