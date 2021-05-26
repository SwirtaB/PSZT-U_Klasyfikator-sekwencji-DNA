from pandas.core.frame import DataFrame
from pandas.core.series import Series
import numpy as np
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
    maxInfGain = 0.0
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


if __name__ == "__main__":
    data = readSpliceFile("Data/spliceDTrainKIS.txt", 15)      
    # data = DataFrame({'x1' : ['A', 'B', 'B', 'B', 'B'], 'x2' : [1, 1, 2, 2, 3], 'class' : [0, 1, 1, 0, 1]} )

    tree = build_ID3(data, "class")

    # obj = DataFrame({'x1' : ['B'], 'x2' : [3]})
    # print(data) 
    obj = data.iloc[0].drop(labels='class') #z danych tzreba się pozbyć klas <- wyrzucam klucz 'class' z series 
    print(f"class for data in row 0 is : " + classify(tree, obj))
    obj = data.iloc[5255].drop(labels='class')
    print(f"class for data in row 5255 is : " + classify(tree, obj))

    # pprint.pprint(tree)