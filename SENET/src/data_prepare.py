import random
from nltk.stem.porter import PorterStemmer
import pickle
from data_entry_structures import DataSet, SENETWordPairRaw
import logging
from dict_utils import collection_to_index_dict, invert_dict
from Cleaner.cleaner import clean_phrase
from sql_db_manager import Sqlite3Manger
from config import *


class Encoder:
    """
    Encode the label to a given representation
    """

    def __init__(self, all_date_type):
        self.all_type = set(all_date_type)
        self.obj_to_num = None
        self.num_to_obj = None

    def one_hot_encode(self, data_entry):
        encoded = [0] * len(self.all_type + 1)
        if self.obj_to_num == None:
            self.obj_to_num = collection_to_index_dict(self.all_type)
            self.num_to_obj = invert_dict(self.obj_to_num)
        encoded[self.obj_to_num[data_entry]] = 1
        return encoded

    def one_bot_decode(self, encoded_entry):
        if self.num_to_obj == None:
            raise Exception("No Decoding book availabel")
        hot_spot_index = list(encoded_entry).index(1)
        origin = self.num_to_obj[hot_spot_index]
        return origin


class SENETRawDataBuilder:
    """
    Prepare a raw material for feature vector building.
    """

    def __init__(self, sql_file, golden_pair_files=["synonym.txt", "contrast.txt", "related.txt"],
                 golden_list_files=["hyper.txt"]):
        """

        :param sql_file: The sqlite file store the scapred document
        :param golden_pair_files: The list of file who contains golden pairs. File should have format <p1>,<p2>. File name
        will be searched in vocab file whose path is defined in config.py
        :param golden_list_files:  The list of file who contains golden pairs. File should have format <root>:<p1>,<p2>....
        """
        sql_manger = Sqlite3Manger(sql_file)
        self.raws = []
        self.keyword_path = VOCAB_DIR + os.sep + "vocabulary.txt"
        self.keys = []
        with open(self.keyword_path, 'r', encoding='utf-8') as kwin:
            for line in kwin:
                self.keys.append(line.strip(" \n\r\t"))

        golden_pairs = self.build_golden(golden_pair_files, golden_list_files)
        neg_pairs = self.build_neg_with_random_pair(golden_pairs)
        labels = ["yes", "no"]  # [0,1] is negative and [1,0] is positive
        print("Candidate neg pairs:{}, Golden pairs:{}".format(len(neg_pairs), len(golden_pairs)))
        cnt_n = cnt_p = 0
        for i, plist in enumerate([golden_pairs, neg_pairs]):
            label = labels[i]
            for pair in plist:
                try:
                    w1_docs = sql_manger.get_content_for_query(pair[0])
                    w2_docs = sql_manger.get_content_for_query(pair[1])
                    material = SENETWordPairRaw(label, pair, (w1_docs, w2_docs))
                    self.raws.append(material)  # This will be parsed by next_batch() in dataset object
                    if i == 0:
                        cnt_n += 1
                    else:
                        cnt_p += 1
                except Exception as e:
                    print(e)
        print("Negative pairs:{} Golden Pairs:{}".format(cnt_n, cnt_p))
        random.shuffle(self.raws)

    def build_neg_with_random_pair(self, golden_pairs):
        def get_random_word(num):
            res = []
            cnt = 0
            key_size = len(self.keys)
            while cnt < num:
                neg_pair = (self.keys[random.randint(0, key_size - 1)], self.keys[random.randint(0, key_size - 1)])
                neg_verse = (neg_pair[1], neg_pair[0])
                if neg_pair not in golden_pairs and neg_verse not in golden_pairs and neg_pair[0] != neg_pair[1]:
                    res.append(neg_pair)
                    cnt += 1
            return res

        neg_pairs = []
        for pair in golden_pairs:
            try:
                g1_negs = get_random_word(1)
                neg_pairs.extend(g1_negs)
            except Exception as e:
                pass
        return neg_pairs

    def build_golden(self, golden_pair_files, golden_list_files):
        pair_set = set()
        for g_pair_name in golden_pair_files:
            path = VOCAB_DIR + os.sep + g_pair_name
            with open(path, encoding='utf8') as fin:
                for line in fin.readlines():
                    words1, words2 = line.strip(" \n").split(",")
                    words1 = clean_phrase(words1)
                    words2 = clean_phrase(words2)
                    if (words2, words1) not in pair_set:
                        pair_set.add((words1, words2))

        for g_list_name in golden_list_files:
            with open(VOCAB_DIR + os.sep + g_list_name) as fin:
                for line in fin.readlines():
                    words1, rest = line.strip(" \n").split(":")
                    words1 = clean_phrase(words1)
                    if rest == "":
                        continue
                    for word in rest.strip(" ").split(","):
                        word = clean_phrase(word)
                        wp = (words1, word)
                        wp_r = (wp[1], wp[0])
                        if wp_r not in pair_set:
                            pair_set.add(wp)

        print("Golden pair number:{}".format(len(pair_set)))
        return pair_set


class DataPrepare:
    """
    Prepare a data set for machine learning tasks. Provide functionality like ten fold validation.
    The dataset can be pickled and imported. The raw_material should match the requirement of feature_pipe
    """

    def __init__(self, pickle_path, feature_pipe, raw_materials, rebuild=True):
        """

        :param pickle_path: The path to store or load data
        :param feature_pipe: A list containing methods to consume raw_material
        :param raw_materials: The data applied to feature pipe. This object should be iterable.
        :param rebuild: If the pickle file exists already, determine whether read it in or just
        rebuild and override existing one
        """
        self.data_set = []  # follow the format of (feature_vec, label, readable_info)
        self.logger = logging.getLogger(__name__)
        self.pick_path = pickle_path
        self.label_encoder = None
        if os.path.isfile(pickle_path) and not rebuild:
            self.__load_file()
        else:
            self.__build_data_set(feature_pipe, raw_materials)

    def __build_data_set(self, feature_pipe, raw_materials):
        """
        Drive the production of data by applying the raw_material to feature_pipe. The
        :param feature_pipe: A list of function process RawMaterial Object
        :param raw_materials: A list of RawMaterial object
        :return:
        """
        self.encoder = Encoder([r.label() for r in raw_materials])
        for entry in raw_materials:
            feature_vec = feature_pipe.get_feature(entry)
            label = self.encoder.one_hot_encode(entry.label())
            readable_info = entry.info()
            data_entry = (feature_vec, label, readable_info)
            self.data_set.append(data_entry)
        self.write_file()

    def write_file(self):
        '''
        entry = (vector, label, (words1, words2)))
        :return:
        '''
        with open(self.pick_path, 'wb') as fout:
            dump_obj = (self.data_set, self.encoder)
            pickle.dump(dump_obj, fout)

    def __load_file(self):
        with open(self.pick_path, 'rb') as fin:
            dump_obj = pickle.load(fin)
            self.data_set = dump_obj[0]
            self.encoder = dump_obj[1]

    def remove_pair_with_same_pre_post(self, pair_set):
        def __stem_Tokens(words):
            porter_stemmer = PorterStemmer()
            return [porter_stemmer.stem(x) for x in words.split(" ")]

        cnt = 0
        filtered = []
        for p in pair_set:
            w1 = __stem_Tokens(p[0])
            w2 = __stem_Tokens(p[1])
            flag = False
            for tk in w1:
                if tk in w2:
                    flag = True
            if flag:
                cnt += 1
                continue
            filtered.append(p)
        print("Totally {} pairs have been removed".format(cnt))
        return filtered

    def get_vec_length(self):
        first = self.data_set[0][0]
        return len(first)

    def ten_fold(self):
        train_test_pair = []
        folds = []
        slice_size = int(len(self.data_set) / 10)
        if slice_size == 0:
            raise Exception("Not enough data to do 10 fold")
        start_cut_index = 0;
        for i in range(0, 10):
            end_cut_index = min(start_cut_index + slice_size, len(self.data_set))
            folds.append(self.data_set[start_cut_index: end_cut_index])
            start_cut_index = end_cut_index

        for i in range(0, 10):
            test_entries = folds[i]
            train_entries = []
            for fd in folds[:i]:
                train_entries.extend(fd)
            for fd in folds[i + 1:]:
                train_entries.extend(fd)

            positive_test_entries = []
            negative_test_entries = []
            for test_entry in test_entries:
                if test_entry[1] == [0., 1.]:
                    negative_test_entries.append(test_entry)
                else:
                    positive_test_entries.append(test_entry)
            pos_test_entries_num = int(
                (0.1 * len(negative_test_entries) / len(positive_test_entries) + 0.1) * len(positive_test_entries))
            positive_test_entries = positive_test_entries[:pos_test_entries_num]
            positive_test_entries.extend(negative_test_entries)
            test_entries = positive_test_entries

            train_set = DataSet(train_entries, self.label_encoder)
            test_set = DataSet(test_entries, self.label_encoder)
            train_test_pair.append((train_set, test_set))
        return train_test_pair
