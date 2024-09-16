import pandas as pd
import numpy as np
from config import model_metadata

if __name__ == "__main__":
    
    preprocessing = data_handling(model_metadata['train_data'],model_metadata['test_data'])
    model_development = modelling(X_train, X_test, y_train, y_test)
    
    train_data, holdout_data = preprocessing.load_the_data()
    X,y = preprocessing.data_processing(train_data, holdout_data, model_metadata['categorical_features'], model_metadata['numerical_features'],model_metadata['target'], model_metadata['outlier_price_threshold'])
    X_train, X_test, y_train, y_test = preprocessing.split_data_train_test_set(X,y,model_metadata)
    
    model, train_pred, test_pred = model_development.training()
    model_results = model_development.model_metrics()
   
   