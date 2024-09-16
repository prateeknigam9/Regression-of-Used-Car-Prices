model_configuration = {
    'seed':18,
    'target': 'price',
    'train_data': r'C:\Users\nigam\OneDrive\Documents\self\Used Car Prices Prediction\train.csv',
    'test_data':r'C:\Users\nigam\OneDrive\Documents\self\Used Car Prices Prediction\test.csv',
    'test_split_size':20;
    'output_folder':r'C:\Users\nigam\OneDrive\Documents\self\Used Car Prices Prediction\results'
    'categorical_features' : ['brand', 'model', 'model_year', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    'numerical_features' : ['milage'],
    
    'outlier_price_threshold': 100000,
}