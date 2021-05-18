import pandas as pd
import numpy as np
import sklearn
import sys
import pprint

def readSpliceFile(filePath, atrNumber):
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

    data = pd.DataFrame(columns=labels)
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
    # print(uniqueAttributes)
    collectionGrouped = collection.groupby([divideAttribute])
    information = 0
    for attribute in uniqueAttributes:
        group = collectionGrouped.get_group(attribute)
        information += len(group)/len(collection) * get_entropy('class', group)
    
    return get_entropy('class', collection) - information

def choose_best_attribute(collection, classLabel):
    maxInfGain = 0
    bestAttribute = None
    for attribute in collection.keys():
        if attribute != classLabel:
            infGain = get_inf_gain(attribute, collection)
            if infGain > maxInfGain:
                bestAttribute = attribute
                maxInfGain = infGain
            elif infGain == maxInfGain:
                bestAttribute = attribute
                maxInfGain = infGain

    return bestAttribute
        

def build_ID3(collection : pd.DataFrame, classLabel, ID3Tree = None):
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
        
data = readSpliceFile("Data/spliceDTrainKIS.txt", 15)
       
# data = pd.DataFrame({'x1' : ['A', 'B', 'B', 'B', 'B'], 'x2' : [1, 1, 2, 2, 3], 'class' : [0, 1, 1, 0, 1]} )
# print(get_inf_gain('x1', data))
# print(get_inf_gain('x2', data))
# pprint.pprint(build_ID3(data, "class"))

tree = build_ID3(data, "class")
# print(tree['x1']['B']['x2'][3])
pprint.pprint(tree)