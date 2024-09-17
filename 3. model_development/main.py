import pandas as pd
import numpy as np
from config import model_metadata
from model_development import *

def runner():
    print("Handling the data...")
    preprocessing = data_handling(model_metadata['train_data'], model_metadata['test_data'])
    
    train_data, holdout_data = preprocessing.load_the_data()
    print("Data loaded")
    master_data, encoded_data, X, y = preprocessing.data_processing(train_data,
                                                        holdout_data,
                                                        model_metadata['categorical_features'],
                                                        model_metadata['numerical_features'],
                                                        model_metadata['target'],
                                                        model_metadata['outlier_price_threshold'])
    
    X_train, X_test, y_train, y_test = preprocessing.split_data_train_test_set(X, y, model_metadata)
    print("Data Split done")
    
    model_development = modelling(X_train, X_test, y_train, y_test)
    
    print("training the model ....")
    model, train_pred, test_pred = model_development.training()
    model_results = model_development.model_metrics(train_pred, test_pred)
    
    holdout_data = preprocessing.process_holdout_data(encoded_data)
    
    print("model training completed")
    print("prediciting the holdout data")    
    submission_data = model_development.predict_price(model, master_data, holdout_data)
    
    utility = utility_functions(model_metadata)
    print(f"storing the data at : {model_metadata['output_folder']}")
    actualvspred = pd.DataFrame()
    actualvspred['actual_values'] = y_train.tolist() + y_test.tolist()
    actualvspred['predicted_values'] = train_pred.tolist() + test_pred.tolist()
    
    model_config = pd.DataFrame([model_metadata]).T.reset_index()
    
    list_of_dfs = [model_config, model_results, actualvspred, submission_data]
    sheet_names = ['model_config', 'model_results', 'actual vs pred', 'submission_data']
    
    utility.export_dataframe_to_excel(list_of_dfs, sheet_names, model_metadata['itr_name'], keep_index=False)
    
    print("Completed")
    
if __name__ == "__main__":
    runner()
    

    
    
    
    
    
   
   
   