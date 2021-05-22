from typing import Tuple
from pandas.core.frame import DataFrame
from random import randint
from DNAClassifier import classify


# Dzieli dane na treningowe (0) i testowe (1).
# test_ratio wyznacza ile otrzymanych danych zostanie użyte do testowania
def splitData(data: DataFrame, test_ratio: float) -> Tuple[DataFrame, DataFrame]:

    rows = data.shape[0]
    pool = [x[0] for x in data.iterrows()]
    chosen = []
    for _ in range(int(rows * (1.0 - test_ratio))):
        i = randint(0, len(pool) - 1)
        chosen.append(pool[i])
        pool.pop(i)

    data1 = data.drop(pool)
    data2 = data.drop(chosen)

    return (data1, data2)


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
