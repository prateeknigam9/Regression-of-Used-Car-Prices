import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor

import optuna

class data_handling:
    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
    def load_the_data(self):
        train_data = pd.read_csv(self.train_data_path)
        holdout_data = pd.read_csv(self.test_data_path)
        
        return train_data, holdout_data
    
    def apply_encoding(self, data_df, categorical_features, target):
        one_hot_feat = pd.get_dummies(data_df[categorical_features], drop_first=True)
        one_hot_feat['milage'] = data_df['milage']    
        one_hot_feat['train_test_flag'] = data_df['train_test_flag']    
        try:
            one_hot_feat[target] = data_df[target] 
        except:
            pass   
        return one_hot_feat
    
    def data_processing(self, train_data, holdout_data, categorical_features, numerical_features, target, outlier_threshold):
        train_data['train_test_flag'] = 'train'
        holdout_data['train_test_flag'] = 'test'
        
        data = pd.concat([train_data, holdout_data])
        
        
        print(f"Train Data Shape: {train_data.shape}")
        print(f"Holdout Data Shape: {holdout_data.shape}")
        print(f"master Data Shape: {data.shape}")
        
        features = categorical_features + numerical_features
        
        encoded_data = self.apply_encoding(data, categorical_features, target)
                
        encoded_train = encoded_data[encoded_data['train_test_flag'] == 'train']
        encoded_train = encoded_train.drop('train_test_flag', axis=1)
        encoded_train = encoded_train[encoded_train[target] < outlier_threshold]
        
        X = encoded_train.drop(target, axis=1)
        y = encoded_train[target]

        print(f"X - shape {X.shape}")
        print(f"y - shape {y.shape}")
        
        return data, encoded_data, X, y
    
    def split_data_train_test_set(self, X, y, model_metadata):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_metadata['test_split_size'], random_state=model_metadata['seed'])
        return X_train, X_test, y_train, y_test
    
    def process_holdout_data(self, encoded_data):
        holdout_data = encoded_data[encoded_data['train_test_flag'] == 'test']
        print(f"holdout_data shape: {holdout_data.shape}")
        holdout_data = holdout_data.drop(['train_test_flag', 'price'], axis=1)
        return holdout_data

        
    
class modelling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train, 
        self.X_test = X_test, 
        self.y_train = y_train, 
        self.y_test = y_test
    
    def training(self):
        model = XGBRegressor(n_estimators= 171, 
                            max_depth = 4, 
                            learning_rate = 0.11938341542726924, 
                            min_child_weight = 4, 
                            objective='reg:squarederror')

        model.fit(self.X_train[0], self.y_train[0])
        
        train_pred = model.predict(self.X_train[0])
        test_pred = model.predict(self.X_test[0])
        
        return model, train_pred, test_pred
    
    def model_metrics(self, train_pred, test_pred):

        train_R_sq = r2_score(self.y_train[0], train_pred)
        train_mse = mean_squared_error(self.y_train[0], train_pred)
        train_rmse = root_mean_squared_error(self.y_train[0], train_pred)

        test_R_sq = r2_score(self.y_test, test_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        test_rmse = root_mean_squared_error(self.y_test, test_pred)

        eval_metrics = {
            'train':{
                'r_sq':train_R_sq, 
                'mse':train_mse, 
                'rmse':train_rmse
            }, 
            'test':{
                'r_sq':test_R_sq, 
                'mse':test_mse, 
                'rmse':test_rmse
            }}
        
        results =  pd.DataFrame(eval_metrics).reset_index()
        
        return results
    
    def predict_price(self, model, master_data, holdout_data):
        submission_data = pd.DataFrame()
        submission_data['id'] = master_data[master_data['train_test_flag'] == 'test']['id']
        submission_data['price'] = model.predict(holdout_data)
        
        return submission_data

class utility_functions:
    def __init__(self,model_metadata):
        self.output_folder = model_metadata['output_folder']
    def export_dataframe_to_excel(self, list_of_dfs, sheet_names, filename, keep_index):
        writer = pd.ExcelWriter(os.path.join(self.output_folder,f"{filename}.xlsx"), engine='xlsxwriter')   
        for dataframe, sheet in zip(list_of_dfs, sheet_names):
            dataframe.to_excel(writer, sheet_name=sheet, index=keep_index)   
        writer.close()
            
        
        

        
        
        