#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:46:19 2020

@author: Aline Barbosa

Modelo para classificar os places da Nohs Somos em 6 labels:
    Bares e baladas
    Restaurantes
    Cafés
    Shows
    Eventos
    Hotéis
    Saúde e bem-estar
"""

# Import packages
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, \
                            classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import numpy as np

"""Import data"""
# Filepath 
path = '/home/aline/Documentos/NohsSomos/ClusteringTypes/'

# Read the data and store data in DataFrame
tabela = pd.read_csv(path + 'tabela_places.csv')

listJson = tabela['metadata'].apply(json.loads).to_list()
metadata = pd.DataFrame.from_records(listJson)

"""Getting to know the data"""
# Print a summary of the data
metadata.describe()

# Columns
metadata.columns

# Types
metadata.dtypes

# First 50 data points
metadata.head(50)

# Shape
metadata.shape

# Missing data
metadata.isnull().sum()

# Drop row with all nan values
metadata = metadata.dropna(how='all')

# Create a new dataframe with only address components
addressComponents = pd.DataFrame.from_records(metadata['address_components'])

# Create a new dataframe with only url, name, place id, types and formatted
# address
types = metadata[{'url', 'name','place_id','types', 'formatted_address'}]

# Basic information about types dataframe
types.isnull().sum()
types.dtypes
types.head(5)

# Dummifying types list
dummies = pd.get_dummies(types['types'].apply(pd.Series).stack()).sum(level=0)

# Concatenate columns from dummies and types dataframes
typesDummified = pd.concat([types, dummies], axis=1)

# Save dataframe on a csv
typesDummified.to_csv(path + 'tabelaDummificada.csv')

# Check numbers of each type
sumTypes = typesDummified.iloc[:,5:4704].sum()

# Chech numbers of each type without establishment (because it is the most
# frequenty type)
sumWithoutEstablishment = typesDummified.loc[typesDummified['establishment'] == 0].iloc[:,5:4704].sum()


"""Manually labeled data"""
# Read the manually labeled data
tableDummies = pd.read_csv(path + 'tableDummies.csv')

# Drop row with all columns as nan values
tableDummies = tableDummies.dropna(how='all')

# Create a new dataframe with only labeled data
labeled = tableDummies[tableDummies.label.notnull()]

"""Model - Out of the box - Labeled data"""
# Split data into train and validation 75/25 %
train_X, val_X, train_y, val_y = train_test_split(labeled.iloc[:,6:106], 
                                                  labeled['label'],
                                                  random_state = 3)

# Random Forest
rf = RandomForestClassifier(n_estimators=400)
rf.fit(train_X, train_y)
rf_predictions = rf.predict(val_X)
accuracy_score(val_y, rf_predictions)
confusion_matrix(val_y, rf_predictions)
print(classification_report(val_y, rf_predictions))
cohen_kappa_score(val_y, rf_predictions)

#SVM 
svc = SVC()
svc.fit(train_X, train_y)
svc_predictions = svc.predict(val_X)
accuracy_score(val_y, svc_predictions)
confusion_matrix(val_y, svc_predictions)
print(classification_report(val_y, svc_predictions))
cohen_kappa_score(val_y, svc_predictions)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_X, train_y)
knn_predictions = svc.predict(val_X)
accuracy_score(val_y, knn_predictions)
confusion_matrix(val_y, knn_predictions)
print(classification_report(val_y, knn_predictions))
cohen_kappa_score(val_y, knn_predictions)

"""Model - Clustering data"""
# Split data into train and validation 75/25 %
trainComplete_X, valComplete_X, trainComplete_y, valComplete_y = \
                                train_test_split(tableDummies.iloc[:,6:106], 
                                                 tableDummies['label'],
                                                 random_state = 3)
#-------------------------------
k=5
kmeans = KMeans(n_clusters=k)
X_dist = kmeans.fit_transform(trainComplete_X) 
representative_idx = np.argmin(X_dist, axis=0)
X_representative = trainComplete_X.values[representative_idx]

y_representative = [list(trainComplete_y)[x] for x in representative_idx]
#------------------------------

# Modelo para usar dados com label para predizer todo o conjunto de dados e 
# salvar csv

#Fit all labeled data
rf.fit(labeled.iloc[:,6:106], labeled['label'])
rf_predictions = rf.predict(tableDummies.iloc[:,6:106])

pd.DataFrame(rf_predictions).to_csv(path + 'outOfTheBoxPredictions.csv')



#--------------------------------
accuracy_score(tableDummies['label'], rf_predictions)
confusion_matrix(valComplete_y, rf_predictions)
print(classification_report(valComplete_y, rf_predictions))
cohen_kappa_score(valComplete_y, rf_predictions)
