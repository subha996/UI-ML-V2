from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score


class Models():
    def __init__(self, X_train, y_train, X_test, y_test, preprocessor):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor


    def log_reg(self, penalty, dual, C, solver, multi_class):
        """Algorithm : Logistics Regression
        Do: will fit the data
        Return: f1 scorem classififcation report, y_prediction, model"""
        try:
            log_reg = LogisticRegression(penalty=penalty, dual=dual, C=C, solver=solver, multi_class=multi_class)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('log_reg', log_reg)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            classi_rep = classification_report(self.y_test, y_pred)
            return f1, classi_rep,y_pred, model

        except Exception as ex:
            print('log_reg class problem' + str(ex))


    def rnd_frst(self, n_estimators, max_depth, criterion, min_samples_split, min_samples_leaf, max_features, bootstrap, max_samples):
        """Algorithm : Random Forest
        Do: will fit the data
        Return: f1 scorem classififcation report, y_prediction, model"""
        try:
            rnd_frst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('log_reg', rnd_frst)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            classi_rep = classification_report(self.y_test, y_pred)
            return f1, classi_rep,y_pred, model
        except Exception as ex:
            print('random forest problem ' + str(ex))

    def svc(self, C, kernel, degree, coef0):
        """Algorithm : Support Vector Classifire
        Do: will fit the data
        Return: f1 scorem classififcation report, y_prediction, model"""
        try:
            svc = SVC(C=C, kernel=kernel, degree=degree, coef0=coef0)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('log_reg', svc)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            classi_rep = classification_report(self.y_test, y_pred)
            return f1, classi_rep,y_pred, model
        except Exception as ex:
            print('svc problem ' + str(ex))


    def knn(self,n_neighbors, weights, algorithm, leaf_size, p):
        """Algorithm : K-Nearest Neighbors Classififire
        Do: will fit the data
        Return: f1 scorem classififcation report, y_prediction, model"""
        try:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('log_reg', knn)])
            model = pipe.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            classi_rep = classification_report(self.y_test, y_pred)
            return f1, classi_rep,y_pred, model
        except Exception as ex:
            print('knn problem ' + str(ex))
        




