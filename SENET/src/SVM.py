from nltk import PorterStemmer
from sklearn import svm
from config import *

from data_prepare import DataPrepare


class SVM:
    def __init__(self):
        self.clf = svm.SVC(probability=True)

    def __stem_Tokens(self, words):
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(x) for x in words.split(" ")]

    def is_heuristic(self, entry):
        readable_info = entry[2]
        w1 = readable_info[0]
        w2 = readable_info[1]
        w1_stems = self.__stem_Tokens(w1)
        w2_stems = self.__stem_Tokens(w2)
        if w1_stems[0] == w2_stems[0] or w1_stems[-1] == w2_stems[-1]:
            return True
        return False

    def reduce_data_set(self, data_set):
        data_set.data = [x for x in data_set.data if not self.is_heuristic(x)]
        return data_set

    def special_ten_fold(self, data_set: DataPrepare):
        result_file1 = RESULT_DIR + os.sep + "SVM_train_with_heu_result{}.txt".format(len(os.listdir(RESULT_DIR)))
        result_csv1 = RESULT_DIR + os.sep + "csv" + os.sep + "SVM_train_with_heu_result_result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        result_file2 = RESULT_DIR + os.sep + "SVM_train_without_with_heu_result{}.txt".format(
            len(os.listdir(RESULT_DIR)))
        result_csv2 = RESULT_DIR + os.sep + "csv" + os.sep + "SVM_train_without_heu_result_result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        with open(result_file1, "w", encoding='utf8') as fout1, open(result_csv1, "w", encoding='utf8') as csv_fout1, \
                open(result_file2, "w", encoding='utf8') as fout2, open(result_csv2, "w", encoding='utf8') as csv_fout2:
            exp_data = data_set.ten_fold()
            for index, (train_set, test_set) in enumerate(exp_data):

                reduced_test_set = self.reduce_data_set(test_set)
                train_X, train_y, train_word_pair = train_set.all()
                train_y = [data_set.encoder.one_hot_decode(x) for x in train_y]

                reduced_test_X, reduced_test_y, reduced_test_word_pair = reduced_test_set.all()
                reduced_test_y = [data_set.encoder.one_hot_decode(x) for x in reduced_test_y]

                clf1 = svm.SVC(probability=True)
                clf1.fit(train_X, train_y)
                pre_labels1 = clf1.predict(reduced_test_X)
                pre_scores1 = clf1.predict_proba(reduced_test_X)
                res1 = []
                for i, pre_label in enumerate(pre_labels1):
                    true_label = reduced_test_y[i]
                    pre_score = pre_scores1[i]
                    correctness = "InCorrect"
                    if pre_label == true_label:
                        correctness = "Correct"
                    res1.append((pre_label, correctness, pre_score, reduced_test_word_pair[i]))


                reduced_train_X, reduced_train_y, reduced_train_word_pair = self.reduce_data_set(train_set).all()
                reduced_train_y =  [data_set.encoder.one_hot_decode(x) for x in reduced_train_y]
                clf2 = svm.SVC(probability=True)
                clf2.fit(reduced_train_X, reduced_train_y)
                pre_labels2 = clf2.predict(reduced_test_X)
                pre_scores2 = clf2.predict_proba(reduced_test_X)
                re, pre, f1, accuracy = self.eval(res1)
                write_csv([re, pre, f1, accuracy], csv_fout1)
                self.write_res(res1, fout1)
                res2 = []
                for i, pre_label in enumerate(pre_labels2):
                    true_label = reduced_test_y[i]
                    pre_score = pre_scores2[i]
                    correctness = "InCorrect"
                    if pre_label == true_label:
                        correctness = "Correct"
                    res2.append((pre_label, correctness, pre_score, reduced_test_word_pair[i]))
                re, pre, f1, accuracy = self.eval(res2)
                write_csv([re, pre, f1, accuracy], csv_fout2)
                self.write_res(res2, fout2)

    def ten_fold(self, data_set: DataPrepare):
        result_file = RESULT_DIR + os.sep + "SVM_result{}.txt".format(len(os.listdir(RESULT_DIR)))
        result_csv = RESULT_DIR + os.sep + "csv" + os.sep + "SVM_result_result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        a_acc = a_recall = a_pre = a_f1 = 0
        with open(result_file, "w", encoding='utf8') as fout, open(result_csv, "w", encoding='utf8') as csv_fout:
            exp_data = data_set.ten_fold()
            for index, (train_set, test_set) in enumerate(exp_data):
                train_X, train_y, train_word_pair = train_set.all()
                train_y = [data_set.encoder.one_hot_decode(x) for x in train_y]
                test_X, test_y, test_word_pair = test_set.all()
                test_y = [data_set.encoder.one_hot_decode(x) for x in test_y]
                self.clf.fit(train_X, train_y)
                pre_labels = self.clf.predict(test_X)
                pre_scores = self.clf.predict_proba(test_X)
                res = []
                for i, pre_label in enumerate(pre_labels):
                    true_label = test_y[i]
                    pre_score = pre_scores[i]
                    correctness = "InCorrect"
                    if pre_label == true_label:
                        correctness = "Correct"
                    res.append((pre_label, correctness, pre_score, test_word_pair[i]))
                re, pre, f1, accuracy = self.eval(res)
                write_csv([re, pre, f1, accuracy], csv_fout)
                self.write_res(res, fout)
                a_recall += re
                a_pre += pre
                a_f1 += f1
                a_acc += accuracy

    def write_res(self, res, writer):
        writer.write("label, correctness, w1, w2\n")
        for label, correctness, pre_score, word_pairs in res:
            res_str = "{}\t{}\t{}\t{}\t\t{}".format(label, correctness, pre_score, word_pairs[0], word_pairs[1])
            writer.write(res_str + "\n")

    def eval(self, results):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for label, correctness, pre_score, test_word_pair in results:
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
    #     for i in range(0,10):
    #         data = DataPrepare("dataset_filter.pickle", feature_pipe=None, raw_materials=None,
    #                            rebuild=False)
    #         svm_model = SVM()
    #         svm_model.ten_fold(data)
    for i in range(10):
        data = DataPrepare("dataset_origin.pickle", feature_pipe=None, raw_materials=None,
                           rebuild=False)
        svm_classifier = SVM()
        svm_classifier.special_ten_fold(data)
        print("Round {} finished".format(i))
