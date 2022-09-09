'''
Date: 9-Sep-2022
Programmer: Hyunwook Kang
Description: This program uses Multilayer Perceptron to train the model, and recognize
            human emotions given a video. The emotions consist one of (neutral, calm, happy,
            sad, angry, fearful, disgust, surprised). The model learns human emotions from 
            facial action units detected in the video. The dataset is RAVDESS. 
            The dataset link: https://zenodo.org/record/1188976#.YxsQ33ZBx9M
            Only video-speech considered. The facial action units are extracted using OpenFace.
            The OpenFace can be installed by following the instruction in below link:
              -https://github.com/TadasBaltrusaitis/OpenFace
'''

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit

import os

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(files_list):
    
    total_au_list=[]
    labels=[]

    for fn in files_list:
        total_au=get_total_au(fn)
        total_au_list.append(total_au)
        label=int(fn.split('\\')[-1].split('-')[2])-1
        labels.append(label)
        
    total_au_list=np.array(total_au_list)
    
    labels=np.array(labels)
    labels=labels.reshape(-1,1)
    
    return total_au_list, labels
    
def get_files(root):
    file_paths=[]
    
    files=os.listdir(root)
    
    for fn in files:
        full_path=os.path.join(root, fn)
        if(os.path.isfile(full_path)):
            
            file_paths.append(full_path)
        else:

            file_paths.extend(get_files(full_path))
            
    return file_paths
        

def get_total_au(data_path):
    data = pd.read_csv(data_path, sep=';', header=0)
    
    data=data.values
    
    total_au=data[0]
    for i in range(1, data.shape[0]): 
        total_au+=data[i]
    total_au/=(data.shape[0]*1.0)
    
    return total_au

def evaluate(y_trues, y_preds):
    
    assert len(y_trues)==len(y_preds)

    n_corr=0
    for i in range(len(y_preds)):
        if(y_trues[i]==y_preds[i]):
            n_corr+=1

    return n_corr/(len(y_trues)*1.0)

if __name__ == '__main__':
    scaler = MinMaxScaler(feature_range=(0, 1))

    classifier = MLPClassifier(random_state=7, hidden_layer_sizes=eval('(80)'))
    
    data_files=get_files('E:/test_ravdess_data')
    data_files=shuffle(data_files)

    au_data, labels = preprocess_data(data_files)

    xxx = StratifiedShuffleSplit(1, test_size=0.3, random_state=12)

    for train_index, test_index in xxx.split(au_data, labels):
        train_X, test_X = au_data[train_index], au_data[test_index]
        train_y, test_y = labels[train_index], labels[test_index]
    
    print(train_X.shape, test_X.shape)
    print(train_y.shape, test_y.shape)
    
    train_X = scaler.fit_transform(train_X)
    test_X=scaler.transform(test_X)

    classifier.fit(train_X, train_y)        
    
    train_acc=evaluate(train_y, classifier.predict(train_X))
    test_acc=evaluate(test_y, classifier.predict(test_X))

    print('train accuracy: %.3f'%(train_acc))
    print('test accuracy: %.3f'%(test_acc))
