import pandas as pd
import numpy as np
import sklearn
import sys

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
    print(uniqueAttributes)
    collectionGrouped = collection.groupby([divideAttribute])
    information = 0
    for attribute in uniqueAttributes:
        group = collectionGrouped.get_group(attribute)
        information += len(group)/len(collection) * get_entropy('class', group)
    
    return get_entropy('class', collection) - information

data = pd.DataFrame({'x1' : ['A', 'B', 'B', 'B', 'B'], 'x2' : [1, 1, 2, 2, 3], 'class' : [0, 1, 1, 0, 1]} )
print(get_inf_gain('x1', data))
print(get_inf_gain('x2', data))