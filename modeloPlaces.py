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
"""

# Import packages
import pandas as pd
import json

"""Import data"""
# Filepath 
path = '/home/aline/Documentos/NohsSomos/ClusteringTypes/'
#metadata_path = '/home/aline/Documentos/NohsSomos/metadata_local.json'

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
sumTypes = typesDummified.iloc[:,3:4704].sum()

# Chech numbers of each type without establishment (because it is the most
# frequenty type)
sumWithoutEstablishment = typesDummified.loc[typesDummified['establishment'] == 0].iloc[:,3:4704].sum()