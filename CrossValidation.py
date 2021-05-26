from typing import Tuple, List, Dict
from pandas.core.frame import DataFrame
from random import randint
from DataSplit import *
import DNAClassifier
from RandomForest import RandomForest


# Zestaw danych do walidacji krzyżowej
class CrossValidator(object):

    def __init__(self, data: DataFrame, k: int, random: bool = False):

        self.k = k
        self.packs = []
        test_packs = []
        if random:
            test_packs = splitDataToPacksRandom(data, k)
        else:
            test_packs = splitDataToPacksSequencial(data, k)
        for test_pack in test_packs:
            drop_rows = [x[0] for x in test_pack.iterrows()]
            train_pack = data.drop(drop_rows)
            self.packs.append((train_pack, test_pack))
    

    def makeTrees(self, class_label: str) -> Tuple[List[Dict], List[float]]:

        trees = []
        tests = []
        for (train_pack, test_pack) in self.packs:
            tree = DNAClassifier.build_ID3(train_pack, class_label)
            trees.append(tree)
            tests.append(DNAClassifier.testID3(tree, test_pack, class_label))

        return (trees, tests)
    

    def makeForests(self, size: int, class_label: str, test_ratio: float) -> Tuple[List[RandomForest], List[float]]:
        
        forests = []
        tests = []
        for (train_pack, test_pack) in self.packs:
            forest = RandomForest(size, train_pack, class_label, test_ratio)
            forests.append(forest)
            tests.append(forest.test(test_pack))

        return (forests, tests)


if __name__ == "__main__":
    cv = CrossValidator(DNAClassifier.readSpliceFile("Data/spliceDTrainKIS_small.txt", 15), 9, True)

    print("trees")
    (trees, ttests) = cv.makeTrees("class")
    for test in ttests:
        print(test)

    print("forests")
    (forests, ftests) = cv.makeForests(9, "class", 0.5)
    for test in ftests:
        print(test)