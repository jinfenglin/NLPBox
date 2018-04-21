from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from config import *
import random
from data_prepare import DataPrepare


class RandomForest:
    def __init__(self):
        pass

    def predict(self, num):
        res = []
        for i in range(num):
            number = random.randint(1, 100)
            if number < 50:
                res.append('yes')
            else:
                res.append('no')
        return res

    def ten_fold(self, data_set: DataPrepare):
        result_file = RESULT_DIR + os.sep + "RandomGuess_result{}.txt".format(len(os.listdir(RESULT_DIR)))
        result_csv = RESULT_DIR + os.sep + "csv" + os.sep + "RandomGuess_result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        a_acc = a_recall = a_pre = a_f1 = 0
        with open(result_file, "w", encoding='utf8') as fout, open(result_csv, "w", encoding='utf8') as csv_fout:
            exp_data = data_set.ten_fold()
            for index, (train_set, test_set) in enumerate(exp_data):
                train_X, train_y, train_word_pair = train_set.all()
                test_X, test_y, test_word_pair = test_set.all()
                train_y = [data_set.encoder.one_hot_decode(x) for x in train_y]
                test_y = [data_set.encoder.one_hot_decode(x) for x in test_y]
                pre_labels = self.predict(len(test_X))
                res = []
                for i, pre_label in enumerate(pre_labels):
                    true_label = test_y[i]
                    correctness = "InCorrect"
                    if pre_label == true_label:
                        correctness = "Correct"
                    res.append((pre_label, correctness, test_word_pair[i]))
                re, pre, f1, accuracy = self.eval(res)
                write_csv([re, pre, f1, accuracy], csv_fout)
                self.write_res(res, fout)
                a_recall += re
                a_pre += pre
                a_f1 += f1
                a_acc += accuracy

    def write_res(self, res, writer):
        writer.write("label, correctness, w1, w2\n")
        for label, correctness, word_pairs in res:
            res_str = "{}\t{}\t{}\t\t{}".format(label, correctness, word_pairs[0], word_pairs[1])
            writer.write(res_str + "\n")

    def eval(self, results):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for label, correctness, test_word_pair in results:
            if label == 'yes':  # positive
                if correctness == 'Correct':
                    tp += 1
                else:
                    fp += 1
            else:
                if correctness == 'Correct':
                    tn += 1
                else:
                    fn += 1

        if tp + fn == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)

        accuracy = (tp + tn) / (tp + tn + fn + fp)

        if tp + fp == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        print("True Negative:{}, True Positive:{}, False Negative:{}, False Positive:{}".format(tn, tp, fn, fp))
        print("recall: {}".format(recall))
        print("precision: {}".format(precision))
        print("f1:{}".format(f1))
        print("accuracy:{}".format(accuracy))
        return recall, precision, f1, accuracy


if __name__ == "__main__":
    data = DataPrepare("dataset_origin.pickle", feature_pipe=None, raw_materials=None,
                       rebuild=False)
    random_forest = RandomForest()
    random_forest.ten_fold(data)
