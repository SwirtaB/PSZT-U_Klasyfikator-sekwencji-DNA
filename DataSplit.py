from typing import Tuple, List
from pandas.core.frame import DataFrame
from random import randint


# Dzieli dane na treningowe (0) i testowe (1).
# split_ratio wyznacza ile % danych zostanie umieszczone w danych treningowych (0).
def splitDataRandom(data: DataFrame, split_ratio: float, drop_second: bool = False) -> Tuple[DataFrame, DataFrame]:

    rows = data.shape[0]
    pool = [x[0] for x in data.iterrows()]
    chosen = []
    for _ in range(int(rows * split_ratio)):
        i = randint(0, len(pool) - 1)
        chosen.append(pool[i])
        pool.pop(i)

    data1 = data.drop(pool)
    if drop_second:
        data2 = None
    else:
        data2 = data.drop(chosen)

    return (data1, data2)


# Dzieli dane na k równych paczek.
# Nie zmienia kolejności danych, są one dzielone w takiej kolejności w jakiej występują w podanych danych.
def splitDataToPacksSequencial(data: DataFrame, packs: int) -> List[DataFrame]:

    pcks = []
    rows = data.shape[0]
    for k in range(packs):
        r = [x for x in range(int(k / packs * rows), int((k+1) / packs * rows))]
        pcks.append(data.take(r))
    
    return pcks


# Dzieli dane na k równych paczek.
# Zmienia losowo kolejność danych, ich kolejność w podanych danych nie ma znaczenia.
def splitDataToPacksRandom(data: DataFrame, packs: int) -> List[DataFrame]:

    pcks = []
    data = data
    for k in range(packs):
        (pack, rest) = splitDataRandom(data, 1.0 / (packs - k))
        pcks.append(pack)
        data = rest
    
    return pcks


# Wybiera losowo n atrybutów w podanych danych i odrzuca resztę.
# Zwrace podane dane bez kolumn z atrybutami, które nie zostały wybrane.
def chooseNAttributes(data: DataFrame, n: int, class_label: str) -> DataFrame:

    attributes = [x for x in data.columns]
    attributes.remove(class_label)

    for i in range(n):
        attributes.pop(randint(0, len(attributes) - 1))
    
    return data.drop(attributes, axis=1)

