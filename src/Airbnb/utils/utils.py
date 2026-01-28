import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Airbnb.logger import logging
from src.Airbnb.exception import customexception
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise customexception(e, sys)
    

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate_model(X_train,y_train,X_test,y_test,models,params=None):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]] if params else {}

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train,y_train)
                model.set_params(**gs.best_params_)
                logging.info(f"Best Params for {list(models.keys())[i]}: {gs.best_params_}")

            model.fit(X_train,y_train)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean_score = np.mean(cv_scores)

            y_test_pred =model.predict(X_test)
            
            test_model_score = r2_score(y_test,y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            logging.info(f"Model: {list(models.keys())[i]}")
            logging.info(f"   - CV Score (R2): {cv_mean_score:.4f}")
            logging.info(f"   - Test Score (R2): {test_model_score:.4f}")
            logging.info(f"   - RMSE: {rmse:.4f}")
            
            report[list(models.keys())[i]] = cv_mean_score

        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)