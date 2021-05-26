import DNAClassifier
from pandas.core.frame import DataFrame
from DataSplit import *
from math import sqrt


class RandomForest(object):

    def __init__(self, size: int, data: DataFrame, class_label: str, split_ratio: float, attribute_choice_fn, test: bool = False):

        n = attribute_choice_fn(len([x for x in data.columns]) - 1)

        self.trees = []
        self.stats = []
        self.class_label = class_label
        for i in range(size):
            (train_data, test_data) = splitDataRandom(data, split_ratio, not test)
            train_data = chooseNAttributes(train_data, n, class_label)
            tree = DNAClassifier.build_ID3(train_data, class_label)
            self.trees.append(tree)
            if test:
                stat = DNAClassifier.testID3(tree, test_data, class_label)
                self.stats.append(stat)


    # Klasyfikuje podany zestaw atrybutów
    def classify(self, row):

        results = {}
        for tree in self.trees:
            result = DNAClassifier.classify(tree, row)
            if result in results:
                results[result] += 1
            else:
                results[result] = 1
        
        max = 0
        max_result = None
        for key in results:
            value = results[key]
            if value > max:
                max = value
                max_result = key
        
        return max_result


    # Testuje las losowy za pomocą podanych danych
    # Zwraca celność lasu na podanych danych
    def test(self, test_data: DataFrame) -> float:

        test_rows = test_data.shape[0]
        test_success = 0
        for i in range(test_rows):
            obj = test_data.iloc[i].drop(self.class_label)
            c = test_data.iloc[i].at[self.class_label]
            result = self.classify(obj)
            if result == c:
                test_success += 1
        
        return test_success / test_rows
