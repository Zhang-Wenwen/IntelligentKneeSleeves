import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch, random
from sklearn.preprocessing import MinMaxScaler
import os

def get_filename(file_path):
    filename = pd.read_csv(file_path)

    # for _, file in filename.iterrows():
        # if not os.path.exists('dataset_2//PINCheck//'+str(file['Date']) + '/' + file['ID']):
        #     os.makedirs('dataset_2//PINCheck//'+file['Date'] + '/' + file['ID'])

    filename = filename['Date'] + "/" + filename['ID'] + "/" + filename['Filename']

    return filename
        
def plot(actual, predicted, number_of_output, output, type):
    if type == 'scatter':
        for i in range(number_of_output):
            fig = plt.figure(figsize=(20, 10))
            p = sns.relplot(x=actual[:, i] * 40 + 40, y=predicted[:, i] * 40 + 40)
            plt.title(output[i])
            p.set(xlabel='Actual',
                  ylabel='Predicted')
            plt.show()
    if type == 'time':
        for i in range(number_of_output):
            fig = plt.figure(figsize=(20, 10))
            sns.lineplot(data=actual[:, i])
            sns.lineplot(data=predicted[:, i])
            fig.legend(labels=['Actual', 'Predicted'])
            plt.title(output[i])
            plt.show()
