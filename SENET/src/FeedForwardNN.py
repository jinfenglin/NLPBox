# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import pickle

from nltk import PorterStemmer

from config import *
import tensorflow as tf

from data_entry_structures import DataSet
from data_prepare import DataPrepare, SENETRawDataBuilder, Encoder
from config import *


class FNN:
    def __init__(self, vec_len, model_path, label_encoder_pickle):
        self.x_size = vec_len
        self.h_size = 15
        self.y_size = 2
        self.epoch = 200
        self.batch = 100
        self.graph = tf.Graph()
        self.model_path = model_path
        self.label_encoder_pickle = label_encoder_pickle
        with self.graph.as_default():
            self.X = tf.placeholder("float", shape=[None, self.x_size])
            self.y = tf.placeholder("float", shape=[None, self.y_size])
            w_1 = self.init_weights((self.x_size, self.h_size))
            w_2 = self.init_weights((self.h_size, self.y_size))
            self.yhat = self.forwardprop(self.X, w_1, w_2)
            self.predict = tf.argmax(self.yhat, axis=1)
            self.correct_pred = tf.equal(self.predict, tf.argmax(self.y, 1))
            self.confidence = tf.nn.softmax(self.forwardprop(self.X, w_1, w_2))

    def train(self, train_set: DataSet):
        with tf.Session(graph=self.graph) as sess:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yhat, labels=self.y))
            updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            train_X, train_y, train_word_pair = train_set.all()
            for epoch in range(self.epoch):
                # Train with each example
                #print("Running epoch {}".format(epoch))
                for i in range(int(len(train_X) / self.batch)):
                    sess.run(updates, feed_dict={self.X: train_X[i * self.batch: (i + 1) * self.batch],
                                                 self.y: train_y[i * self.batch: (i + 1) * self.batch]})
            saver.save(sess, self.model_path)
            with open(self.label_encoder_pickle, "wb") as encoder_fout:
                pickle.dump(train_set.label_encoder, encoder_fout)

    def classify(self, feature_vecs: DataSet):
        """
        Classify a given entry
        :param feature_vec:
        :param classify_res_file The file to store classification result
        :return: One type of classes
        """
        res = []
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            classify_X, classify_y, classify__word_pair = feature_vecs.all()
            confidence_score = sess.run(self.confidence, feed_dict={self.X: classify_X})
            res.append(classify__word_pair)
            res.append(confidence_score)

        with open(self.label_encoder_pickle, 'rb') as pickle_in:
            # classifier use its own label encoder and don't trust the one provided by dataset
            label_encoder = pickle.load(pickle_in)
        return res, label_encoder

    def test(self, test_set: DataSet):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            test_X, test_y, test_word_pair = test_set.all()
            is_correct = sess.run(self.correct_pred, feed_dict={self.X: test_X, self.y: test_y})
            confidence_score = sess.run(self.confidence, feed_dict={self.X: test_X})
            res = (is_correct, test_word_pair, confidence_score)
        return res

    def eval(self, results, encoder: Encoder):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for correctness, word_pairs, confidence in zip(results[0], results[1], results[2]):
            label = encoder.one_hot_confidence_decode(confidence)
            if label == "yes":  # positive
                if correctness:
                    tp += 1
                else:
                    fp += 1
            else:
                if correctness:
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
        print(
            "True Negative:{}, True Positive:{}, False Negative:{}, False Positive:{}".format(tn, tp, fn, fp))
        print("recall: {}".format(recall))
        print("precision: {}".format(precision))
        print("f1:{}".format(f1))
        print("accuracy:{}".format(accuracy))
        return recall, precision, f1, accuracy

    def write_res(self, res, writer, encoder):
        writer.write("label, correctness, w1, w2\n")
        for correctness, word_pairs, confidence_score in zip(res[0], res[1], res[2]):
            label = encoder.one_hot_confidence_decode(confidence_score)
            correct_output = 'Incorrect'
            if correctness:
                correct_output = 'Correct'
            res_str = "{}\t{}\t{}\t{}\t\t{}".format(label, correct_output, confidence_score, word_pairs[0],
                                                    word_pairs[1])
            writer.write(res_str + "\n")

    def init_weights(self, shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)

    def forwardprop(self, X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    def ten_fold_test(self, data_set: DataPrepare):
        result_file = RESULT_DIR + os.sep + "FeedForward_Result{}.txt".format(len(os.listdir(RESULT_DIR)))
        result_csv = RESULT_DIR + os.sep + "csv" + os.sep + "FeedForward_result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        with open(result_file, "w", encoding='utf8') as fout, open(result_csv, "w", encoding='utf8') as csv_fout:
            fout.write("Label encoding:")
            for l_type in data_set.encoder.all_types:
                fout.write("{}={}\n".format(l_type, data_set.encoder.one_hot_encode(l_type)))
            fout.write("label,correctness,w1,w2,confidence\n")
            a_acc = a_recall = a_pre = a_f1 = 0
            for index, (train_set, test_set) in enumerate(data_set.ten_fold()):
                print("Start fold {}".format(index))
                self.train(train_set)
                tf.reset_default_graph()
                res = self.test(test_set)
                re, pre, f1, acc = self.eval(res, data_set.encoder)
                write_csv([re, pre, f1, acc], csv_fout)
                self.write_res(res, fout, data_set.encoder)
                a_recall += re
                a_pre += pre
                a_f1 += f1
                a_acc += acc
            print("Average Recall={}\nAverage Precision={}\nAverage F1={}\nAverage Accuracy={}\n".
                  format(a_recall / 10, a_pre / 10, a_f1 / 10, a_acc / 10))

    def run(self):
        data_set = DataPrepare()
        self.train_test(data_set)

    def train_test(self, data_set):
        RANDOM_SEED = 42
        tf.set_random_seed(RANDOM_SEED)
        # Layer's sizes
        x_size = data_set.get_vec_length()  # Length of the feature vector
        h_size = self.h_size  # Number of hidden nodes
        y_size = self.y_size  # classify as yes/no

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        w_1 = self.init_weights((x_size, h_size))
        w_2 = self.init_weights((h_size, y_size))

        # Forward propagation
        yhat = self.forwardprop(X, w_1, w_2)
        predict = tf.argmax(yhat, axis=1)
        correct_pred = tf.equal(predict, tf.argmax(y, 1))

        # Backward propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        # Run SGD
        with tf.Session() as sess:
            result_file = RESULT_DIR + os.sep + "FeedForward_Result{}.txt".format(len(os.listdir(RESULT_DIR)))
            result_csv = RESULT_DIR + os.sep + "csv" + os.sep + "FeedForward_result{}.csv".format(
                len(os.listdir(RESULT_DIR)))
            a_acc = a_recall = a_pre = a_f1 = 0
            with open(result_file, "w", encoding='utf8') as fout, open(result_csv, "w", encoding='utf8') as csv_fout:
                exp_data = data_set.ten_fold()
                for index, (train_set, test_set) in enumerate(exp_data):
                    train_X, train_y, train_word_pair = train_set.all()
                    test_X, test_y, test_word_pair = test_set.all()
                    self.confidence = tf.nn.softmax(self.forwardprop(X, w_1, w_2))
                    print("Start fold {}".format(index))
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    res = []
                    for epoch in range(self.epoch):
                        # Train with each example
                        print("Running epoch {}".format(epoch))
                        for i in range(len(train_X)):
                            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
                    is_correct = sess.run(correct_pred, feed_dict={X: test_X, y: test_y})
                    confidence_score = sess.run(self.confidence, feed_dict={X: test_X})
                    res = (is_correct, test_word_pair, confidence_score)
                    re, pre, f1, acc = self.eval(res, data.encoder)
                    write_csv([re, pre, f1, acc], csv_fout)
                    self.write_res(res, fout, data_set.encoder)
                    a_recall += re
                    a_pre += pre
                    a_f1 += f1
                    a_acc += acc

                avg_str = "Average recall:{}, precision:{}, f1:{}, accuracy:{}".format(a_recall / 10, a_pre / 10,
                                                                                       a_f1 / 10,
                                                                                       a_acc / 10)
                fout.write(avg_str)
                print(avg_str)
        tf.reset_default_graph()

    def write_classify_res(self, file_path, res, label_encoder: Encoder):
        # TODO inlcude encoder information in result
        with open(file_path, "w", encoding="utf8") as fout:
            fout.write("Label Encoding:\n")
            for l_type in label_encoder.all_types:
                fout.write("{}={}\n".format(l_type, label_encoder.one_hot_encode(l_type)))
            for (test_word_pairs, confidence_score) in zip(res[0], res[1]):
                fout.write("{},{}\n".format(str(test_word_pairs), str(confidence_score)))

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

    def special_ten_fold(self, data_set):
        result_file1 = RESULT_DIR + os.sep + "FNN_train_heu_test_non_heu_Result{}.txt".format(
            len(os.listdir(RESULT_DIR)))
        result_csv1 = RESULT_DIR + os.sep + "csv" + os.sep + "FNN_train_heu_test_non_heu_Result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        result_file2 = RESULT_DIR + os.sep + "FNN_train_non_heu_test_non_heu_Result{}.txt".format(
            len(os.listdir(RESULT_DIR)))
        result_csv2 = RESULT_DIR + os.sep + "csv" + os.sep + "FNN_train_non_heu_test_non_heu_Result{}.csv".format(
            len(os.listdir(RESULT_DIR)))
        with open(result_file1, "w", encoding='utf8') as fout1, open(result_csv1, "w", encoding='utf8') as csv_fout1, \
                open(result_file2, "w", encoding='utf8') as fout2, open(result_csv2, "w", encoding='utf8') as csv_fout2:
            fout1.write("label,correctness,w1,w2,confidence\n")
            fout2.write("label,correctness,w1,w2,confidence\n")
            for index, (train_set, test_set) in enumerate(data_set.ten_fold()):
                print("Start fold {}".format(index))
                # self.train(train_set)
                # tf.reset_default_graph()
                reduced_test_set = self.reduce_data_set(test_set)
                # res = self.test(reduced_test_set)
                # re, pre, f1, acc = self.eval(res, data_set.encoder)
                # write_csv([re, pre, f1, acc], csv_fout1)
                # self.write_res(res, fout1, data_set.encoder)

                reduced_train_set = self.reduce_data_set(train_set)
                tf.reset_default_graph()
                self.train(reduced_train_set)
                res = self.test(reduced_test_set)
                re, pre, f1, acc = self.eval(res, data_set.encoder)
                write_csv([re, pre, f1, acc], csv_fout2)
                self.write_res(res, fout2, data_set.encoder)


# if __name__ == '__main__':
#     for i in range(10):
#         data = DataPrepare("dataset_origin.pickle", feature_pipe=None, raw_materials=None,
#                            rebuild=False)
#         fnn = FNN(data.get_vec_length(), FNN_MODEL_DIR, FNN_ENCODER_PATH)
#         fnn.train_test(data)
#         print("Round {} finished".format(i))

if __name__ == '__main__':
    for i in range(10):
        data = DataPrepare("dataset_origin.pickle", feature_pipe=None, raw_materials=None,
                           rebuild=False)
        fnn = FNN(data.get_vec_length(), FNN_MODEL_DIR, FNN_ENCODER_PATH)
        fnn.special_ten_fold(data)
        print("Round {} finished".format(i))