from typing import Tuple, List, Dict
from pandas.core.frame import DataFrame
from random import randint
from DataSplit import *
import DNAClassifier
from RandomForest import RandomForest
from math import sqrt


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
    

    # def makeTrees(self, class_label: str) -> Tuple[List[Dict], List[float]]:

    #     trees = []
    #     tests = []
    #     for (train_pack, test_pack) in self.packs:
    #         tree = DNAClassifier.build_ID3(train_pack, class_label)
    #         trees.append(tree)
    #         tests.append(DNAClassifier.testID3(tree, test_pack, class_label))

    #     return (trees, tests)
    
    # Tworzy i testuje lasy losowe za pomocą swoich danych
    def makeForests(self, size: int, class_label: str, split_ratio: float, attribute_choice_fn) -> Tuple[List[RandomForest], List[float]]:
        
        forests = []
        tests = []
        for (train_pack, test_pack) in self.packs:
            forest = RandomForest(size, train_pack, class_label, split_ratio, attribute_choice_fn)
            forests.append(forest)
            tests.append(forest.test(test_pack))

        return (forests, tests)
