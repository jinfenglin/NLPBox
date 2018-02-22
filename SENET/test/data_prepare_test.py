import unittest

from RNN import RNN, RNN_MODEL_DIR
from data_prepare import SENETRawDataBuilder, DataPrepare, Encoder
from feature_extractors import SENETFeaturePipe


class TestDataPrepare(unittest.TestCase):
    def test_raw_prepare(self):
        rb = SENETRawDataBuilder("test.db")

    def test_encoder(self):
        labels = ["yes", "no", "unknow"]
        print("labels:", labels)
        encoder = Encoder(labels)
        test_label = "yes"
        encoded_label = encoder.one_hot_encode(test_label)
        print(encoded_label)
        assert encoded_label == [1, 0, 0]
        assert test_label == encoder.one_hot_decode(encoded_label)

    def test_feature_building(self):
        golden_pairs = ["debug_gold.test"]
        vocab = "debug_vocab.test"
        rb = SENETRawDataBuilder("test.db", golden_pair_files=golden_pairs, vocab_file_name=vocab, golden_list_files=[])
        pipeline = SENETFeaturePipe()
        data = DataPrepare("test_dataset.pickle", feature_pipe=pipeline, raw_materials=rb.raws, rebuild=True)
        rnn = RNN(data.get_vec_length(), RNN_MODEL_DIR)
        rnn.ten_fold_test(data_set=data, result_file="rnn_test.res")
