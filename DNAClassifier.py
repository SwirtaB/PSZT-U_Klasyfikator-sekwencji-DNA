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
        i = i + 1
    labels.append('class')

    data = pd.DataFrame(columns=labels)
    file.readline()
    lines = file.read().splitlines()
    i = 1
    while i < len(lines):
        record = lines[i] + lines[i - 1]
        row = list(record)
        data.loc[len(data)] = row
        i = i + 2

    file.close()
    return data