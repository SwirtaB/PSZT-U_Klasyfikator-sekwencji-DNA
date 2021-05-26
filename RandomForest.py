import DNAClassifier
from pandas.core.frame import DataFrame
from DataSplit import *


class RandomForest(object):

    def __init__(self, size: int, data: DataFrame, class_label: str, test_ratio: float):

        self.trees = []
        self.stats = []
        self.class_label = class_label
        for i in range(size):
            (train_data, test_data) = splitDataRandom(data, test_ratio)
            tree = DNAClassifier.build_ID3(train_data, class_label)
            stat = DNAClassifier.testID3(tree, test_data, class_label)
            self.trees.append(tree)
            self.stats.append(stat)


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


if __name__ == "__main__":
    (rf_data, test_rf_data) = splitDataRandom(DNAClassifier.readSpliceFile("Data/spliceDTrainKIS.txt", 15), 0.25)
    rf = RandomForest(9, rf_data, "class", 0.33)
    print(rf.stats)
    print(rf.test(test_rf_data))