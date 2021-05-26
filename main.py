from typing import List
from CrossValidation import CrossValidator
from DNAClassifier import readSpliceFile
from pandas import DataFrame
from math import sqrt as msqrt
from multiprocessing import Process, Pool
from time import sleep

def identity(x):
    return x

def div_2(x):
    return int(x/2)

def div_3(x):
    return int(x/3)

def sqrt(x):
    return int(msqrt(x))

cvsn = 1
cvpacks = 5
ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
treesn = [50, 100, 200, 400, 800]
attr_fns = [
    ("n", identity),
    ("n/2", div_2),
    ("n/3", div_3),
    ("sqrt(n)", sqrt),
]


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


def tests(data: DataFrame, file_prefix: str):

    processes: List[Process] = []

    cvs: List[CrossValidator] = []
    for i in range(cvsn):
        cvs.append(CrossValidator(data, cvpacks, True))

    # base test
    p = Process(target=test, args=(cvs, file_prefix, ratios[2], treesn[2], attr_fns[3]))
    processes.append(p)

    # trees tests
    for trees in treesn:
        p = Process(target=test, args=(cvs, file_prefix, ratios[2], trees, attr_fns[3]))
        processes.append(p)

    # attr tests
    for attr_fn in attr_fns:
        p = Process(target=test, args=(cvs, file_prefix, ratios[2], treesn[2], attr_fn))
        processes.append(p)

    # train set tests
    for ratio in ratios:
        p = Process(target=test, args=(cvs, file_prefix, ratio, treesn[2], attr_fns[3]))
        processes.append(p)

    for p in processes:
        p.start()
        
    for p in processes:
        p.join()


def main():

    spliceDdata = readSpliceFile("Data/spliceDTrainKIS.txt", 15)
    tests(spliceDdata, "spliceD")
    spliceAdata = readSpliceFile("Data/spliceATrainKIS.txt", 90)
    tests(spliceAdata, "spliceA")


if __name__ == "__main__":
    main()