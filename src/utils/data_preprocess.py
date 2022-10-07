import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

class Data_preprocess:
    def __init__(self):
        self.numerical_column_selector = make_column_selector(dtype_exclude=object)
        self.categorical_column_selector = make_column_selector(dtype_include=object)

    def fill_nan_values(self, method, data, *args):
        #num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        #cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        if method == "train":
            for arg in args:
                columns = arg[0]
                strategy = arg[1]
                if strategy == 'mean':
                    #print(f'col:{columns}')
                    num_imputer = ColumnTransformer([('num', SimpleImputer(missing_values=np.nan, strategy='mean'), columns)], remainder='passthrough')
                    num_imputer_tr = num_imputer.fit_transform(data)
                    new_col = [col for col in data.columns if col not in columns]
                    columns1 = columns.copy()
                    columns1.extend(new_col)
                    #print(f'new_col:{columns}')
                    data[columns1] = num_imputer_tr
                    joblib.dump(num_imputer, './num_imputer.pkl')

                elif strategy == 'most_frequent':
                    #print(f'col:{columns}')
                    cat_imputer = ColumnTransformer([('num', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), columns)], remainder='passthrough')
                    cat_imputer_tr = cat_imputer.fit_transform(data)
                    new_col = [col for col in data.columns if col not in columns]
                    columns1 = columns.copy()
                    columns1.extend(new_col)
                    #print(f'new_col:{columns1}')
                    data[columns1] = cat_imputer_tr
                    joblib.dump(cat_imputer, './cat_imputer.pkl')

        elif method == "test":
            for arg in args:
                columns = arg[0]
                strategy = arg[1]
                if strategy == 'mean':
                    num_imputer = joblib.load('./num_imputer.pkl')
                    num_imputer_ts = num_imputer.transform(data)
                    new_col = [col for col in data.columns if col not in columns]
                    columns1 = columns.copy()
                    columns1.extend(new_col)
                    data[columns1] = num_imputer_ts

                if strategy == 'most_frequent':
                    cat_imputer = joblib.load('./cat_imputer.pkl')
                    cat_imputer_ts = cat_imputer.transform(data)
                    new_col = [col for col in data.columns if col not in columns]
                    columns1 = columns.copy()
                    columns1.extend(new_col)
                    data[columns1] = cat_imputer_ts

            """for col in columns:
                print(f'col:{col}')
                if strategy == 'mean':
                    temp_data = num_imputer.fit_transform(np.array(data[col]).reshape(-1,1))
                elif strategy == 'most_frequent':
                    temp_data = cat_imputer.fit_transform(np.array(data[col]).reshape(-1,1))

                data[col] = temp_data"""
        #print(f'data:{data.iloc[1]}')
        return data

    def encode_categorical_values(self, data, columns, method):
        if method == 'train':
            encoder = ColumnTransformer([('enc', OneHotEncoder(drop='first', sparse=False), columns)], remainder='passthrough')
            encoder_tr = encoder.fit_transform(data)
            joblib.dump(encoder, './encoder.pkl')
            return encoder_tr

        elif method == 'test':
            encoder = joblib.load('./encoder.pkl')
            encoder_ts = encoder.transform(data)
            return encoder_ts
        """ for col in columns:
            unique_val = data[col].unique()
            temp_data = encoder.fit_transform(np.array(data[col]).reshape(-1,1))
            temp_data_df = pd.DataFrame(temp_data.toarray(), columns=unique_val[1:])
            data.drop([col], axis=1, inplace=True)
            data = pd.concat([data, temp_data_df], axis=1) """


    def normalize_numerical_values(self, data, columns, method):
        if method == 'train':
            print(f'columns:{columns}')
            scalar = ColumnTransformer([('sc', StandardScaler(), columns)], remainder='passthrough')
            scalar_tr = scalar.fit_transform(data)
            new_col = [col for col in data.columns if col not in columns]
            columns.extend(new_col)
            data[columns] = scalar_tr
            joblib.dump(scalar, './scalar.pkl')
        elif method == 'test':
            scalar = joblib.load('./scalar.pkl')
            scalar_ts = scalar.transform(data)
            new_col = [col for col in data.columns if col not in columns]
            columns.extend(new_col)
            data[columns] = scalar_ts

        return data

    
    def preprocess_train_data(self, data):
        
        numerical_columns = self.numerical_column_selector(data)
        categorical_columns = self.categorical_column_selector(data)
        joblib.dump(categorical_columns, './categorical_col.pkl')
        joblib.dump(numerical_columns, './numerical_col.pkl')

        data = self.fill_nan_values('train', data, [numerical_columns, 'mean'], [categorical_columns, 'most_frequent'])
        data = self.normalize_numerical_values(data, numerical_columns, 'train')
        data = self.encode_categorical_values(data, categorical_columns, 'train')
        return np.array(data)

    def preprocess_test_data(self, data):

        numerical_columns = joblib.load('./numerical_col.pkl')
        categorical_columns = joblib.load('./categorical_col.pkl')

        data = self.fill_nan_values('test', data, [numerical_columns, 'mean'], [categorical_columns, 'most_frequent'])
        data = self.normalize_numerical_values(data, numerical_columns, 'test')
        data = self.encode_categorical_values(data, categorical_columns, 'test')
        return np.array(data)




    
        
   


    

