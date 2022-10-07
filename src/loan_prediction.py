import pandas as pd
import numpy as np
import json
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, app, jsonify,url_for, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils.data_preprocess import Data_preprocess
from src.utils.common import *

app = Flask(__name__)
class Loan_prediction:
    def __init__(self):
        """ data_df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
        data_df.drop(['Loan_ID', 'Loan_Amount_Term', 'Loan_Status'], axis=1, inplace=True)
        self.preprocessor = Data_preprocess()
        pr = self.preprocessor.preprocess_train_data(data_df) 
        print(pr)"""
        self.preprocessor = Data_preprocess()
        self.model = joblib.load('./regression_model.pkl')
        


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train', methods=['POST'])
def train():
    data_df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
    data_df.drop(['Loan_ID', 'Loan_Amount_Term'], axis=1, inplace=True)
    X, y = data_df.drop(['Loan_Status'], axis=1), pd.DataFrame(data_df['Loan_Status'])
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = loan_pred.preprocessor.preprocess_train_data(X_train)
    log_model = LogisticRegression(solver='liblinear', max_iter=500, penalty='l1')
    log_model.fit(X_train, Y_train)
    x_test = loan_pred.preprocessor.preprocess_test_data(x_test)
    y_pred = log_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)*100
    cm = confusion_matrix(y_test, y_pred)
    #print(f'pcm:{pcm}')
    ax = sns.heatmap(cm/np.sum(cm), annot=True, cmap='Blues')
    ax.set_title('Seaorn Confusion matrix\n\n')
    ax.set_xlabel('\nPredicted values')
    ax.set_ylabel('\nActual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()
    joblib.dump(log_model, './regression_model.pkl')
    return render_template('home.html', prediction_text=f'accuracy is {acc}, y_pred: {y_pred}')

@app.route('/predict', methods=['POST'])
def predict():
    #loan_pred = Loan_prediction()
    input_data = [x for x in request.form.values()]
    input_data = make_df(np.array(input_data).reshape(1,-1))
    input_data = loan_pred.preprocessor.preprocess_test_data(input_data)
    output = loan_pred.model.predict(input_data)[0]
    return render_template('home.html', prediction_text=f'Loan Prediction is {output}')


if __name__ == '__main__':
    loan_pred = Loan_prediction()
    #port = 5000
    #app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)

    
        