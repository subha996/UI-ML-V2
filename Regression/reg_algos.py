from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.pipeline import Pipeline



# creating main class for regression algorithms
class Models():
    def __init__(self, X_train, y_train, X_test, y_test, preprocessor):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor


    def lin_reg(self, fit_intercept, normlize, n_jobs):
        """Linear Regression
        Do: fit the data with given hyperparameters
        Return: r2_score, rmse, and model"""
        try:
            lin_reg = LinearRegression(fit_intercept=fit_intercept, normalize=normlize, n_jobs=n_jobs)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', lin_reg)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train) 
            score_test = r2_score(self.y_test, y_pred_test)
            mae_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            score_train = r2_score(self.y_train, y_pred_train)
            mae_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
            
        except Exception as ex:
            print('lin_reg_methode problem '+ str(ex))

    def rf_reg(self, n_estimators, criterion, max_depth, min_sample_split, min_sample_leaf, max_features, bootstrap, max_samples):
            """Random Forest Regression
        Do: fit the data with given hyperparameters
        Return: r2_score, rmse, and model"""
            try:
                rf_reg = RandomForestRegressor(n_estimators=n_estimators,criterion=criterion, max_depth=max_depth,
                                                min_samples_split=min_sample_split, min_samples_leaf=min_sample_leaf,
                                                max_features=max_features, bootstrap=bootstrap, max_samples=max_samples, n_jobs=-1)
                pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', rf_reg)])
                model = pipe.fit(self.X_train, self.y_train)
                y_pred_test = model.predict(self.X_test)
                y_pred_train = model.predict(self.X_train) 
                score_test = r2_score(self.y_test, y_pred_test)
                mae_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                score_train = r2_score(self.y_train, y_pred_train)
                mae_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
            except Exception as ex:
                print('random forest_methode problem '+ str(ex))

    def svr_reg(self,kernel, degree, gamma, coef0, C, epsilon, shrinking, max_iter):
        
        try:
            svr = SVR(kernel=kernel, degree=degree, gamma=gamma,coef0=coef0, C=C, epsilon=epsilon, shrinking=shrinking, max_iter=max_iter)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', svr)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train) 
            score_test = r2_score(self.y_test, y_pred_test)
            mae_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            score_train = r2_score(self.y_train, y_pred_train)
            mae_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
        except Exception as ex:
            print('svr_methode problem '+ str(ex))


    def knn(self, n_neighbors, algorithm, leaf_size, p ):
        try:
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=-1)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', knn)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train) 
            score_test = r2_score(self.y_test, y_pred_test)
            mae_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            score_train = r2_score(self.y_train, y_pred_train)
            mae_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            return score_train, mae_train, score_test, mae_test, y_pred_train, y_pred_test, model
        except Exception as ex:
            print('knn_methode problem '+ str(ex))   


    



