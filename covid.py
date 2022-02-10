# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:22:18 2022

@author: theo-
"""
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
import pathlib
import os
import pip
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
data = pd.read_csv('dataset.csv')

data.head()

df = data.copy()

#target variablle = test resul sars covid

#shape 5644 ligne 111 colonnes
print(df.shape)

df.dtypes.value_counts().plot.pie()

plt.figure(figsize=(20,10))    
sns.heatmap(df.isna(), cbar=False)


print((df.isna().sum()/df.shape[0]).sort_values(ascending=False))


df = df[df.columns[df.isna().sum()/df.shape[0]<0.9]]
#on drop le lid du patient
df = df.drop('Patient ID', axis=1)

print(df.shape)
sns.heatmap(df.isna(), cbar=False)



print(df['SARS-Cov-2 exam result'].value_counts())



#Il y a 10 % de cas positif et 90 % de cas negatif
'''
for col in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[col])
'''
plt.figure()
sns.distplot(df['Patient age quantile'])

df['SARS-Cov-2 exam result'].unique()

'''
for col in df.select_dtypes('object'):
    print( f'{col  :-<50} {df[col].unique()}')
    '''
    
df = df.drop('Parainfluenza 2', axis=1)

'''
for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts().plot.pie()
    print( f'{col  :-<50} {df[col].unique()}')
    '''
    
positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']

missing_rate = df.isna().sum()/df.shape[0]

blood_columns = df.columns [(missing_rate < 0.9) & (missing_rate > 0.88)]


viral_columns = df.columns [(missing_rate < 0.80) & (missing_rate > 0.75)]
'''
for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='positive')
    plt.legend()
'''    
    
    
#platelet monocyte leukocyte on l'air detredifferent en cas de covid


sns.countplot(x = 'Patient age quantile', hue='SARS-Cov-2 exam result', data = df)

'''
for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')
    
    '''
    
sns.heatmap(df[blood_columns].corr())
sns.clustermap(df[blood_columns].corr())

'''
for col in blood_columns:
    plt.figure()
    sns.lmplot(x = 'Patient age quantile', y=col,hue='SARS-Cov-2 exam result', data = df)
'''
def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] ==1:
        return 'surveillance'
    if df['Patient addmited to semi-intensive unit (1=yes, 0=no)']==1:
        return 'soins semi-intensives'
    if df['Patient addmited to intensive care unit (1=yes, 0=no)']==1:
        return 'soins intensifs'
    else:
        return 'inconnu'
df['statut'] = df.apply(hospitalisation, axis=1)

'''
# test relation entre le type de soins intensif et les autre categories
for col in blood_columns :
    plt.figure()
    for cat in df['statut'].unique():
        sns.distplot(df[df['statut']==cat][col])
        plt.legend()
'''
        

#on voit bien que si on croise les lignes qui ont toutes les valeurs on a plus que 99 colonnes

# Sinon on peut travailler avec viral colummns -> 1300 colonne ou blood colonne  500 colonne

#test hypothèse

#On emet une hypothes les taux moyens sont egaux chez les individus positifs ou negatif.

# on calcule notre valeur

df['est malade'] = np.sum(df[viral_columns[:-2]] == 'detected', axis=1) >=1

print(positive_df.shape)
print(negative_df.shape)


#on pose un set negatif egal au nombre de positif :
#    On rearde la proba si elle est rejeté
    
balanced_neg = negative_df.sample(positive_df.shape[0])

print(negative_df.shape)


def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 Rejetée'
    else:
        return 0




for col in blood_columns:
    print(f'{col :-<50} {t_test(col)}')
    
# on remarque que pour les platelets, pour les leukocytes pour les eosinophils et les monocyte la theorie est rejjeté
# l'analyse des données est terminé on va faire le preprocessing des données

#Selection, creation train set, test set, encodage, on fait un modele simple et on essaye d'optimiser.

df1 = df.copy()
df1.head()


missing_rate = df1.isna().sum() / df1.shape[0]

blood_columns = list(df1.columns [(missing_rate < 0.9) & (missing_rate > 0.88)])


viral_columns = list(df1.columns [(missing_rate < 0.80) & (missing_rate > 0.75)])

key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result'] 

df1 = df1[key_columns + blood_columns + viral_columns]

print(df1.head())

trainset, testset = train_test_split(df1, test_size = 0.2, random_state=0)

print(trainset['SARS-Cov-2 exam result'].value_counts())

print(testset['SARS-Cov-2 exam result'].value_counts())


# NETOYAGE DU TRAIN SET
def encodage (df1):
    code = {'positive': 1,
            'negative': 0,
            'detected': 1,
            'not_detected': 0}

    for col in df1.select_dtypes('object'):
        df1[col]= df1[col].map(code)
    return df1
    # il n'ya plus de varaible de type objet

def feature_engineering(df1):
    df1['est malade'] = np.sum(df1[viral_columns[:-2]] == 'detected', axis=1) >=1
    df1 = df1.drop(viral_columns, axis=1)
    return df1

def imputation(df1):
    #df1['is na'] = (df1['Parainfluenza 3'].isna()) | (df1['Leukocytes'].isna())
    #df1.fillna(-999)
    df1 = df1.dropna(axis=0)
    return df1

def preprocessing(df1):
    df1 = encodage(df1)
    df1 = feature_engineering(df1)
    df1 = imputation(df1)
    
    X = df1.drop('SARS-Cov-2 exam result', axis=1)
    y = df1['SARS-Cov-2 exam result']
    return X, y



X_train, y_train = preprocessing(df1)

X_test, y_test = preprocessing(testset)

        
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

list_of_models = []
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=(False)),SelectKBest(f_classif, k=5))

RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=(0)))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=(0)))
SVM = make_pipeline(preprocessor,StandardScaler(), SVC(random_state=(0),))
KNN = make_pipeline(preprocessor,StandardScaler(), KNeighborsClassifier())
list_of_models = [RandomForest,AdaBoost,SVM,KNN]

#Procedure d'évaluation

def evaluation(model):
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))

    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, scoring='f1', train_sizes=np.linspace(0.1, 1,10))

    plt.figure(figsize=(12, 8))
    
    plt.plot(N, train_score.mean(axis=1),label='trainscore')
    plt.plot(N, val_score.mean(axis=1),label='valscore')
    plt.legend()



dict_of_models = {'randomForest': RandomForest,
                  'AdaBoost': AdaBoost,
                  'SVM': SVM,
                  'KNN': KNN}
from sklearn.model_selection import GridSearchCV

hyper_params = { 'svc__gamma':[1e-3, 1e-4],
                'svc__C': [1, 10, 100, 1000]}



grid = GridSearchCV(SVM, hyper_params, scoring='recall', cv=4)
grid.fit(X_train, y_train)
print(grid.best_params_)


y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))

    


#On va regarder les variables les plus importante

plt.figure()
#plt.plot(pd.DataFrame(model.feature_importances_, index=X_train.columns))

evaluation(grid.best_estimator_)