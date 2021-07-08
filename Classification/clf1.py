from streamlit.proto.Button_pb2 import Button
from Classification.split import Split
from sklearn.pipeline import Pipeline
from Classification.clasi_algos import Models
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
# imporing preprocessing modules
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# importing pipeline and columntransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from Classification.my_plot_confusion_matrix import my_confusion_matrix
from Classification.classification_to_df import classification_report_to_dataframe
import pickle, base64
from warnings import filterwarnings
filterwarnings(action='ignore')


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




    # writing title of the app
    st.markdown("<h1 style='text-align: center;'>🤖 UI ML - Classification 🤖</h1>", unsafe_allow_html=True)

    # asking for file 
    st.sidebar.info('Please Upload Your Dataset or Use Sample Dataset.')

    # file uploader
    file_upload = st.sidebar.file_uploader(label='please upload Your file', type=['csv'], help='Please Upload `csv` file only')

    # slection of sample dataset
    sample_datasets_name = st.sidebar.selectbox('Select Dataset', options=('None','Iris Flowers', 'Wine', 'Breast Cancer'), help='Selcet Sample Data From Here.')



    # function for getting the dataset
    def get_dataset(sample=True, custome=False):
        try:
            if sample:
                if sample_datasets_name == 'Iris Flowers':
                    df, y = datasets.load_iris(return_X_y=True, as_frame=True)
                    df['Flowers Name (Target Column)'] = y # adding targeget column to dataframe
                    return df
                elif sample_datasets_name == 'Wine':
                    df, y = datasets.load_wine(return_X_y=True, as_frame=True)
                    df['Profile (Target Column)'] = y
                    return df
                elif sample_datasets_name == 'Breast Cancer':
                    df, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
                    df['Class (Target Column)'] = y
                    return df

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
    if st.sidebar.checkbox('Show Data', True):
        st.write('Your Current Data: - ')
        st.write(df)

    col1, col2 = st.beta_columns(2)

    def run_model(algo_c, preprocessor):
        try:
            algo = Models(X_train, y_train, X_test, y_test, preprocessor)
            if algo_c == 'Logistic Regression':
                st.write('You Select `Logistic Regression`.[Read More](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)')
                penalty = col1.selectbox('penalty', options=['l2', 'l1', 'none', 'elasticnet'],help='Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.')
                dual = col2.selectbox('dual', options=[False, True], help='Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.')
                C = col1.slider('C', 1.0,5.0,0.25, help='Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.')
                solver = col2.selectbox('solver', options=['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], help ='Algorithm to use in the optimization problem. for more read, visit documentation from below link')
                multi_class = col1.selectbox('multi_class', options=['auto', 'ovr', 'multinomial'], help="If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.")
                f1, classi_rep,y_pred, model = algo.log_reg(penalty=penalty, dual=dual, C=C, solver=solver, multi_class=multi_class) # model instance
                return f1, classi_rep,y_pred, model

            elif algo_c == "RandomForest Classifier":
                    st.write('You Select `RandomForest Classifier`. [Ream More](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)')
                    n_estimators = col1.slider('n_estimators', 50,500,100, help="The Number of tree in the forest. in africa's forest!!, 😁😁.  Just kidding !!")
                    max_depth = col2.slider('max_depth', 1,10,3, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
                    criterion = col1.selectbox('criterion', options=['gini', 'entropy'], help="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.")
                    min_samples_split = col2.slider('min_sample_split', 2,10,3,help="The minimum number of samples required to split an internal node")
                    min_samples_leaf = col1.slider('min_samples_leaf', 1,10,1, help="The minimum number of samples required to be at a leaf node")
                    max_features = col2.selectbox('max_features', options=['auto', 'sqrt', 'log2'], help="The number of features to consider when looking for the best split")
                    bootstrap = col1.selectbox('bootstrap',options=[True, False], help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
                    max_samples = col2.slider('max_samples', 1,10,3,help="If bootstrap is True, the number of samples to draw from X to train each base estimator.")
                    f1, classi_rep,y_pred, model = algo.rnd_frst(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
                    return f1, classi_rep,y_pred, model

            elif algo_c == 'Support Vector Classifier':
                st.write('You Select `Support Vector Classifier`. [Read More](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)')
                C = col1.slider('C', 1.0,10.0,0.25,help='Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.')
                kernel = col2.selectbox('kernel', options=['rbf', 'linear', 'ploy', 'sigmoid'], help="Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used., Missing `precomputed`")
                degree = col1.slider('degree', 1,10,1,help='Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.')
                coefo = col2.slider('coed0', 0.0,5.0,0.25, help='Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.')
                f1, classi_rep, y_pred,model = algo.svc(C=C, kernel=kernel, degree=degree, coef0=coefo)
                return f1, classi_rep,y_pred, model
        
            elif algo_c == 'KNeighbors Classifier':
                st.write('You Select `KNeighbors Classifier`.[Read More](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)')
                n_neighbours = col1.slider('n_neighbours', 1,10,1,help='Number of neighbors to use by default for kneighbors queries.')
                weights = col2.selectbox('weights', ['uniform', 'distance'], help="missing `callable`")
                algorithm = col1.selectbox('algorithm', options=['auto', 'ball_tree', 'kd_tree', 'brute'], help="Algorithm used to compute the nearest neighbors")
                leaf_size = col2.slider('leaf_size', 10,100,10,help='Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. default: 30')
                p = col1.selectbox('p', options=[2,1], help="Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.")
                f1, classi_rep,y_pred, model = algo.knn(n_neighbors=n_neighbours, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
                return f1, classi_rep,y_pred, model
        except Exception as ex:
            print(str(ex))


    if df is not None:
        try:
            col = list(df)
            # col = col.insert(0, 'None')
            split = Split(df)
            algo_lst = ['Logistic Regression', 'RandomForest Classifier', 'Support Vector Classifier', 'KNeighbors Classifier']
            target = st.sidebar.selectbox('select target col', options=col)
            algo_chose = st.sidebar.selectbox('Select Algorithms', options=algo_lst, help="Selcet the Algorithm from here !!")
            X, y = split.X_and_y(target)
            # getting categorical columns and numerical columns list
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
            with st.spinner('Hold on model is training 🛠🛠🔧'):
                f1, classi_rep,y_pred, model = run_model(algo_chose, preprocessor)
        except Exception:
            st.warning('Please Chose Target columns')
        button = st.button("Train and Give Score", help="Click this button to train the model and get score")

        if button:
            st.text('f1 score: ' + str(round(f1, 3)))
            classi_df = classification_report_to_dataframe(classi_rep)
            st.text("Classification Report")
            st.dataframe(classi_df)
            with st.spinner('Creating Confusion matrix 🛠🔧🛠⚒'):
                 my_confusion_matrix(y_test, y_pred, figsize=(5,5), text_size=7)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            # download the model
            output_model = pickle.dumps(model)
            b64 = base64.b64encode(output_model).decode()
            href = f'<a href="data:file/output_model;base64,{b64}" download="model.pkl">Download Trained Model .pkl File</a>'
            st.text('You can download the trained model from below link. 🎈')
            st.markdown(href, unsafe_allow_html=True)
            st.balloons()

    else:
        st.warning('Please Upload Dataset or Choose from Sample Data')  


