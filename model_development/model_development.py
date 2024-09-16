import pandas as pd
import numpy as np

class data_handling:
    def __init__(self,train_data_path,test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
    def load_the_data(self):
        train_data = pd.read_csv(self.train_data_path)
        holdout_data = pd.read_csv(self.test_data_path)
        
        return train_data, holdout_data
    
    def apply_encoding(self,data_df,categorical_features):
        one_hot_feat = pd.get_dummies(data_df[categorical_features],drop_first=True)
        one_hot_feat['milage'] = data_df['milage']    
        one_hot_feat['train_test_flag'] = data_df['train_test_flag']    
        try:
            one_hot_feat[target] = data_df[target] 
        except:
            pass   
        return one_hot_feat
    
    def data_processing(self, train_data, holdout_data, categorical_features, numerical_features,target, outlier_threshold):
        train_data['train_test_flag'] = 'train'
        holdout_data['train_test_flag'] = 'test'
        
        data = pd.concat([train_data,holdout_data])
        
        
        print(f"Train Data Shape: {train_data.shape}")
        print(f"Holdout Data Shape: {holdout_data.shape}")
        print(f"master Data Shape: {data.shape}")
        
        features = categorical_features + numerical_features
        
        encoded_data = self.apply_encoding(data,categorical_features,target)
                
        encoded_train = encoded_data[encoded_data['train_test_flag'] == 'train']
        encoded_train = encoded_train.drop('train_test_flag',axis=1)
        encoded_train = encoded_train[encoded_train[target] < outlier_threshold]
        
        X = encoded_train.drop(target,axis=1)
        y = encoded_train[target]

        print(f"X - shape {X.shape}")
        print(f"y - shape {y.shape}")
        
        return X,y
    
    def split_data_train_test_set(self,X,y,model_metadata):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=model_metadata['test_split_size'],random_state=model_metadata['seed'])
        return X_train,X_test,y_train,y_test
    
class modelling:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train,
        self.X_test = X_test,
        self.y_train = y_train,
        self.y_test = y_test
    
    def training(self):
        model = XGBRegressor(n_estimators= 763,
                            max_depth = 3,
                            learning_rate = 0.04236490789285951,
                            min_child_weight = 3,
                            objective='reg:squarederror')
        
        model.fit(self.X_train,self.y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        return model, train_pred, test_pred
    
    def model_metrics(self):
        train_R_sq = r2_score(self.y_train,self.train_pred)
        train_mse = mean_squared_error(self.y_train,self.train_pred)
        train_rmse = root_mean_squared_error(self.y_train,self.train_pred)

        test_R_sq = r2_score(self.y_test,self.test_pred)
        test_mse = mean_squared_error(self.y_test,self.test_pred)
        test_rmse = root_mean_squared_error(self.y_test,self.test_pred)

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
        
        results =  pd.DataFrame(eval_metrics)
        
        return results
        

        
        
        