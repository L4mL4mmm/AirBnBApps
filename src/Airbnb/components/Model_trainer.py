import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import save_object
from src.Airbnb.exception import customexception
from src.Airbnb.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Artifacts','Model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,val_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train, val and test data')
            X_train, y_train, X_val, y_val, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                val_array[:,:-1],
                val_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'RandomForestRegressor':RandomForestRegressor(),
            'GradientBoostingRegressor':GradientBoostingRegressor()
        }
            
            params={
                "LinearRegression":{},
                "Lasso":{
                    'alpha': [0.1, 1.0, 10.0]
                },
                "Ridge":{
                    'alpha': [0.1, 1.0, 10.0]
                },
                "Elasticnet":{
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                },
                "RandomForestRegressor":{
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                "GradientBoostingRegressor":{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 8]
                }
            }
            
            logging.info(f"Validation Set Shape: {X_val.shape}")

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
          
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)