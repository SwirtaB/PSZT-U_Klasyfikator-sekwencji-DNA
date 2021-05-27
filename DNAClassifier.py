from pandas.core.frame import DataFrame
from pandas.core.series import Series
import numpy as np
import sys
import pprint


# Wczytuje plik z danymi i zwraca obiekt z nimi.
def readSpliceFile(filePath, atrNumber) -> DataFrame:
    try:
        file = open(filePath, 'r')
    except IOError:
        print(f"path to file: {filePath} is not correct")
        return

    labels = []
    i = 1
    while i < atrNumber + 1:
        labels.append('x' + str(i))
        i += 1
    labels.append('class')

    data = DataFrame(columns=labels)
    file.readline()
    lines = file.read().splitlines()

    i = 1
    while i < len(lines):
        record = lines[i] + lines[i - 1]
        row = list(record)
        data.loc[len(data)] = row
        i += 2

    file.close()
    return data


def get_entropy(divideAttribute, collection):
    uniqueValueCounter = collection[divideAttribute].value_counts()
    entropy = 0
    for value in uniqueValueCounter:
        frequency = value / len(collection[divideAttribute])
        entropy += -frequency*np.log(frequency)
    
    return entropy


def get_inf_gain(divideAttribute, collection):
    uniqueAttributes = collection[divideAttribute].unique()
    collectionGrouped = collection.groupby([divideAttribute])
    information = 0
    for attribute in uniqueAttributes:
        group = collectionGrouped.get_group(attribute)
        information += len(group)/len(collection) * get_entropy('class', group)
    
    return get_entropy('class', collection) - information
    

def choose_best_attribute(collection, classLabel):
    attributes = []
    gains = []
    for attribute in collection.keys():
        if attribute != classLabel:
            attributes.append(attribute)
            infGain = get_inf_gain(attribute, collection)
            gains.append(infGain)
    
    maxInfGain = gains[0]
    bestAttribute = attributes[0]

    for i in range(1, len(attributes)):
        if gains[i] > maxInfGain:
            maxInfGain = gains[i]
            bestAttribute = attributes[i]

    return bestAttribute
        

# Buduje i zwraca drzewo ID3 wytrenowane na podanych danych.
def build_ID3(collection : DataFrame, classLabel, ID3Tree = None):
    if ID3Tree is None:
        ID3Tree = {}

    uniqueClasses = collection[classLabel].value_counts()
    if len(uniqueClasses) == 1:
        return uniqueClasses.index[0]

    if len(collection.keys()) == 1:
        return uniqueClasses.index[0]

    bestAttribute = choose_best_attribute(collection, classLabel)
    uniqueValues = collection[bestAttribute].unique()
    ID3Tree[bestAttribute] = {}
    for value in uniqueValues:
        subCollection = collection[collection[bestAttribute] == value].reset_index(drop=True) #dzieli DataFrame zgodnie z wartością wybranego atrybutu
        subCollection = subCollection.drop(bestAttribute, axis=1) #pozbywamy się wyciętej kolumny
        ID3Tree[bestAttribute][value] = build_ID3(subCollection, classLabel)

    return ID3Tree


# Klasyfikuje za pomocą podanego drzewa ID3 podany zestaw atrybutów.
def classify(ID3Tree, row, lastKey=None, checkKey=False):
    if type(ID3Tree) is not dict:
        return ID3Tree
    for key, value in ID3Tree.items():
        if checkKey is False:
            checkKey = True
            lastKey = key
            return classify(value, row, lastKey, checkKey)
        if checkKey is True:
            if(row[lastKey][0] == key):
                checkKey = False
                return classify(value, row, lastKey, checkKey)


# Testuje drzewo ID3 za pomoca danych testowych
# Zwraca celnosc drzewa dla podanych danych
def testID3(tree, test_data: DataFrame, class_label: str) -> float:

    test_rows = test_data.shape[0]
    test_success = 0
    for i in range(test_rows):
        obj = test_data.iloc[i].drop(class_label)
        c = test_data.iloc[i].at[class_label]
        result = classify(tree, obj)
        if result == c:
            test_success += 1
    
    return test_success / test_rows
