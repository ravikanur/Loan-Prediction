import pandas as pd


def make_df(data):
    columns = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Credit_History', 'Property_Area']

    input_data = pd.DataFrame(data, columns=columns)
    return input_data
