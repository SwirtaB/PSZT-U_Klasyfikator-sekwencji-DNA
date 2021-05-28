from typing import List
from CrossValidation import CrossValidator
from DNAClassifier import readSpliceFile
from pandas import DataFrame
from math import sqrt as msqrt
from multiprocessing import Process

def identity(x):
    return x

def div_2(x):
    return int(x/2)

def div_3(x):
    return int(x/3)

def sqrt(x):
    return int(msqrt(x))

# Ilość walidatorów krzyżowych użytych do testowania.
cvsn = 1
# Ilość paczek na które walidator krzyżowy dzieli dane.
cvpacks = 5
# Przypadki ilości drzew w lesie.
treesn = [1, 3, 6, 12, 25, 50, 100, 200, 400, 800]
# Przypadki ilości danych przeznaczonych na trenowanie jednego drzewa.
ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
# Przypadki funkcji zależności ilości atrybutów w jednym drzewie od wszystkich atrybutów.
attr_fns = [
    ("n", identity),
    ("nDIV2", div_2),
    ("nDIV3", div_3),
    ("sqrt(n)", sqrt),
]


# Przeprowadzenie jednego testu na podanym walidatorze krzyżowym, z podanymi parametrami.
def test(cvs: List[CrossValidator], file_prefix: str, ratio: float, trees: int, attr_fn):

    name = "%s ratio=%f, trees=%d, attr_fn=%s.txt"%(file_prefix, ratio, trees, attr_fn[0])
    print("Started test: %s"%(name))

    sum = 0.0
    for cv in cvs:
        tests = cv.makeForests(trees, "class", ratio, attr_fn[1])[1]
        sumt = 0.0
        for test in tests:
            sumt += test
        sum += sumt / cvpacks
    score = sum / cvsn

    file = open("Test/%s"%(name), "w")
    file.write("%s ratio: %f trees: %d attr_fn: %s score: %f"%(file_prefix, ratio, trees, attr_fn[0], score))
    file.close()

    print("Finished test: %s"%(name))


# Przeprowadzenie testów dla spliceD.
def testSpliceD(data: DataFrame):

    file_prefix = "spliceD"

    processes: List[Process] = []

    cvs: List[CrossValidator] = []
    for _ in range(cvsn):
        cvs.append(CrossValidator(data, cvpacks, True))

    # Przypadek bazowy.
    p = Process(target=test, args=(cvs, file_prefix, ratios[2], treesn[7], attr_fns[3]))
    processes.append(p)

    # Testy dla zmiennej ilości drzew.
    for trees in treesn:
        p = Process(target=test, args=(cvs, file_prefix, ratios[2], trees, attr_fns[3]))
        processes.append(p)

    # Testy dla różnych funkcji ilości atrybutów w drzewie.
    for attr_fn in attr_fns:
        p = Process(target=test, args=(cvs, file_prefix, ratios[2], treesn[1], attr_fn))
        processes.append(p)

    # Testy dla różnych ilości danych przeznaczanych na trenowanie jednego drzewa.
    for ratio in ratios:
        p = Process(target=test, args=(cvs, file_prefix, ratio, treesn[1], attr_fns[0]))
        processes.append(p)

    # Uruchomienie testów.
    for p in processes:
        p.start()
        
    # Czekanie na zakończenie testów.
    for p in processes:
        p.join()


# Przeprowadzenie testów dla spliceA.
def testSpliceA(data: DataFrame):

    file_prefix = "spliceA"

    processes: List[Process] = []

    cvs: List[CrossValidator] = []
    for _ in range(cvsn):
        cvs.append(CrossValidator(data, cvpacks, True))

    # Przypadek bazowy.
    p = Process(target=test, args=(cvs, file_prefix, ratios[2], treesn[7], attr_fns[3]))
    processes.append(p)

    # Testy dla zmiennej ilości drzew.
    for trees in treesn:
        p = Process(target=test, args=(cvs, file_prefix, ratios[2], trees, attr_fns[3]))
        processes.append(p)

    # Testy dla różnych funkcji ilości atrybutów w drzewie.
    for attr_fn in attr_fns:
        p = Process(target=test, args=(cvs, file_prefix, ratios[2], treesn[3], attr_fn))
        processes.append(p)

    # Testy dla różnych ilości danych przeznaczanych na trenowanie jednego drzewa.
    for ratio in ratios:
        p = Process(target=test, args=(cvs, file_prefix, ratio, treesn[3], attr_fns[0]))
        processes.append(p)

    # Uruchomienie testów.
    for p in processes:
        p.start()
        
    # Czekanie na zakończenie testów.
    for p in processes:
        p.join()


def main():

    spliceDdata = readSpliceFile("Data/spliceDTrainKIS.txt", 15)
    testSpliceD(spliceDdata)
    spliceAdata = readSpliceFile("Data/spliceATrainKIS.txt", 90)
    testSpliceA(spliceAdata)



if __name__ == "__main__":
    main()