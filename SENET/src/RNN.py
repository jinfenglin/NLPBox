import tensorflow as tf

from data_entry_structures import DataSet
from data_prepare import DataPrepare, Encoder, SENETRawDataBuilder
import logging
from config import *
from feature_extractors import SENETFeaturePipe
import sys, pickle


class RNN:
    def __init__(self, vec_len, model_path, label_encoder_pickle):
        self.lr = 0.001  # learning rate
        self.training_iters = 4000  # 100000  # train step upper bound
        self.batch_size = 1
        self.n_inputs = vec_len  # MNIST data input (img shape: 28*28)
        self.n_steps = 1  # time steps
        self.n_hidden_units = 128  # neurons in hidden layer
        self.n_classes = 2  # classes (0/1 digits)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start building network...")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])
            self.weights = {
                'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units]), name='w_in'),
                'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]), name='w_out')
            }
            self.biases = {
                'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ]), name='b_in'),
                'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]), name='b_out')
            }
            self.pred = self.__classify(self.x, self.weights, self.biases)
            self.confidence = tf.nn.softmax(self.pred)
        self.model_path = model_path
        self.label_encoder_pickle = label_encoder_pickle

    def train(self, train_set: DataSet):
        with tf.Session(graph=self.graph) as sess:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            step = 0
            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys, train_word_pairs = train_set.next_batch(self.batch_size)
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                sess.run([train_op], feed_dict={
                    self.x: batch_xs,
                    self.y: batch_ys,
                })
                step += 1
            saver.save(sess, self.model_path)
            with open(self.label_encoder_pickle, "wb") as encoder_fout:
                pickle.dump(train_set.label_encoder, encoder_fout)

    def test(self, test_set: DataSet):
        """
        Evaluate the quality of  trained model. The label in test are all encoded thus don't need a label encoder. If
        the trian/test set have different encoding format or order, then the result is not valid.
        :return:
        """
        res = []
        pred_label_index = tf.argmax(self.pred, 1)  # Since we use one-hot represent the predicted label, index = label
        correct_pred = tf.equal(pred_label_index, tf.argmax(self.y, 1))
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            for i in range(len(test_set.data)):
                batch_xs, batch_ys, test_word_pairs = test_set.next_batch(self.batch_size)
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                is_correct = sess.run(correct_pred, feed_dict={self.x: batch_xs, self.y: batch_ys})
                confidence_score = sess.run(self.confidence, feed_dict={self.x: batch_xs})
                res.append((batch_ys, is_correct, test_word_pairs, batch_xs, confidence_score))
        return res

    def ten_fold_test(self, data_set: DataPrepare, result_file):
        with open(result_file, "w", encoding="utf8") as fout:
            fout.write("Label encoding:")
            for l_type in data_set.encoder.all_types:
                fout.write("{}={}\n".format(l_type, data_set.encoder.one_hot_encode(l_type)))
            fout.write("label,correctness,w1,w2,confidence,features\n")
            a_acc = a_recall = a_pre = a_f1 = 0
            for index, (train_set, test_set) in enumerate(data_set.ten_fold()):
                self.logger.info("Start fold {}".format(index))
                self.train(train_set)
                tf.reset_default_graph()
                res = self.test(test_set)
                re, pre, f1, accuracy = self.write_fold_result(res, fout, data_set.encoder)
                a_recall += re
                a_pre += pre
                a_f1 += f1
                a_acc += accuracy
            fout.write("Average Recall={}\nAverage Precision={}\nAverage F1={}\nAverage Accuracy={}\n".
                       format(a_recall / 10, a_pre / 10, a_f1 / 10, a_acc / 10))

    def __classify(self, X, weights, biases):
        X = tf.reshape(X, [-1, self.n_inputs])
        X_in = tf.matmul(X, weights['in']) + biases['in']
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return results

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
            for i in range(len(feature_vecs.data)):
                batch_xs, batch_ys, test_word_pairs = feature_vecs.next_batch(self.batch_size)
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                confidence_score = sess.run(self.confidence, feed_dict={self.x: batch_xs})
                res.append((test_word_pairs, confidence_score))
        with open(self.label_encoder_pickle, 'rb') as pickle_in:
            label_encoder = pickle.load(pickle_in)
        return res, label_encoder

    def eval(self, results):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for label, correctness, word_pairs, feature, confidence in results:
            if label[0][0] == 1:  # positive
                if correctness[0]:
                    tp += 1
                else:
                    fp += 1
            else:
                if correctness[0]:
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
        self.logger.info(
            "True Negative:{}, True Positive:{}, False Negative:{}, False Positive:{}".format(tn, tp, fn, fp))
        self.logger.info("recall: {}".format(recall))
        self.logger.info("precision: {}".format(precision))
        self.logger.info("f1:{}".format(f1))
        self.logger.info("accuracy:{}".format(accuracy))
        return recall, precision, f1, accuracy

    def write_fold_result(self, res, writer, label_encoder: Encoder):
        for label, correctness, word_pairs, features, confidence in res:
            label = label_encoder.one_hot_decode(label[0])
            if correctness[0] == True:
                correct_output = 'Correct'
            else:
                correct_output = "Incorrect"

            res_str = "{}\t{}\t{}\t\t{}\t{}\t{}\n".format(label, correct_output, word_pairs[0][0], word_pairs[0][1],
                                                          confidence, features)
            writer.write(res_str)
        re, pre, f1, accuracy = self.eval(res)
        stat_str = "recall:{}, precision:{}, f1:{}, accuracy:{} \n".format(re, pre, f1, accuracy)
        writer.write(stat_str)
        return re, pre, f1, accuracy

    def write_classify_res(self, file_path, res, label_encoder: Encoder):
        # TODO inlcude encoder information in result
        with open(file_path, "w", encoding="utf8") as fout:
            fout.write("Label Encoding:\n")
            for l_type in label_encoder.all_types:
                fout.write("{}={}\n".format(l_type, label_encoder.one_hot_encode(l_type)))
            for (test_word_pairs, confidence_score) in res:
                fout.write("{},{}\n".format(str(test_word_pairs), str(confidence_score)))


if __name__ == "__main__":
    golden_pairs = ["debug_gold.test"]
    vocab = "debug_vocab.test"
    rb = SENETRawDataBuilder("test.db", golden_pair_files=golden_pairs, vocab_file_name=vocab, golden_list_files=[])
    pipeline = SENETFeaturePipe()
    data = DataPrepare("test_dataset.pickle", feature_pipe=pipeline, raw_materials=rb.raws, rebuild=False)
    rnn = RNN(data.get_vec_length(), RNN_MODEL_PATH)
    rnn.ten_fold_test(data_set=data, result_file="rnn_test.res")
