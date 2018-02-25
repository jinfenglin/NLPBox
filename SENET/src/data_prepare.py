import sys
sys.path.append("../../Utils")
sys.path.append("../../Cleaner")
import random
from nltk.stem.porter import PorterStemmer
import pickle
from data_entry_structures import DataSet, SENETWordPairRaw
from dict_utils import collection_to_index_dict, invert_dict
from Cleaner.cleaner import clean_phrase
from sql_db_manager import Sqlite3Manger
from config import *
import json
import logging
import threading, math


class PairBuilder:
    def __read_words(self, file_path):
        res = set()
        with open(file_path, encoding='utf8') as fin:
            for line in fin.readlines():
                phrase = line.strip("\n\t\r ")
                res.add(phrase)
        return list(res)

    def __get_all_relationships(self):
        constras = os.path.join(VOCAB_DIR, "contrast.txt")
        hyper = os.path.join(VOCAB_DIR, "hyper.txt")
        related = os.path.join(VOCAB_DIR, "related.txt")
        synonym = os.path.join(VOCAB_DIR, "synonym.txt")
        one_pair_in_line = [constras, related, synonym]
        multi_pair_in_line = [hyper]
        rel = set()
        for f in one_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    word_pair = line.split(",")
                    relation = (word_pair[0], word_pair[1])
                    rel.add(relation)
        for f in multi_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    hyper, rest = line.split(":")
                    for w_r in rest.split(","):
                        rel.add((hyper[0], w_r))
        return rel

    def __init__(self, expension_list_txt):
        self.exp_list = self.__read_words(expension_list_txt)
        self.relations = self.__get_all_relationships()

    def get_pairs(self):
        pairs = []
        vocab = self.__read_words(os.path.join(VOCAB_DIR, "small_vocabulary.txt"))
        for w_v in vocab:
            for w_e in self.exp_list:
                if (w_v, w_e) not in self.relations and (w_e, w_v) not in self.relations:
                    pairs.append((w_v, w_e))
        return pairs


class Encoder:
    """
    Encode the label to a given representation
    """

    def __str__(self):
        return str(self.obj_to_num)

    def __init__(self, all_types):
        assert len(set(all_types)) == len(all_types)
        self.all_types = all_types
        self.obj_to_num = None
        self.num_to_obj = None

    def one_hot_encode(self, data_entry):
        encoded = [0] * len(self.all_types)
        if self.obj_to_num == None:
            self.obj_to_num = collection_to_index_dict(self.all_types)
            self.num_to_obj = invert_dict(self.obj_to_num)
        encoded[self.obj_to_num[data_entry]] = 1
        return encoded

    def one_hot_decode(self, encoded_entry):
        if self.num_to_obj == None:
            raise Exception("No Decoding book availabel")
        hot_spot_index = list(encoded_entry).index(1)
        origin = self.num_to_obj[hot_spot_index]
        return origin


class SENETRawDataBuilder:
    """
    Prepare a raw material for feature vector building.
    """

    @staticmethod
    def find_document(label, w1, w2, sql_manger):
        documents = {}
        try:
            w1_docs = sql_manger.get_content_for_query(w1)
            w2_docs = sql_manger.get_content_for_query(w2)

            for key in w1_docs:
                if len(w1_docs[key]) > 0:
                    w1_docs[key] = json.loads(w1_docs[key])
                else:
                    w1_docs[key] = []

            for key in w2_docs:
                if len(w2_docs[key]) > 0:
                    w2_docs[key] = json.loads(w2_docs[key])
                else:
                    w2_docs[key] = []
            material = SENETWordPairRaw(label, (w1, w2))
            documents[w1] = w1_docs
            documents[w2] = w2_docs
            return material, documents
        except Exception as e:
            print(e)

    def __init__(self, sql_file, pair_builder: PairBuilder = None,
                 golden_pair_files=["synonym.txt", "contrast.txt", "related.txt"],
                 golden_list_files=["hyper.txt"], vocab_file_name="vocabulary.txt"):
        """
        :param pair_builder: Build pairs, if this is not None other pararmeters will be ignored
        :param sql_file: The sqlite file store the scapred document
        :param golden_pair_files: The list of file who contains golden pairs. File should have format <p1>,<p2>. File name
        will be searched in vocab file whose path is defined in config.py
        :param golden_list_files:  The list of file who contains golden pairs. File should have format <root>:<p1>,<p2>....
        """
        self.logger = logging.getLogger(__name__)
        sql_manger = Sqlite3Manger(sql_file)
        self.documents = {}
        self.raws = []
        if pair_builder is None:
            self.keyword_path = os.path.join(VOCAB_DIR, vocab_file_name)
            self.keys = []
            with open(self.keyword_path, 'r', encoding='utf-8') as kwin:
                for line in kwin:
                    self.keys.append(line.strip(" \n\r\t"))

            golden_pairs = self.build_golden(golden_pair_files, golden_list_files)
            neg_pairs = self.build_neg_with_random_pair(golden_pairs)
            labels = ["yes", "no"]  # [0,1] is negative and [1,0] is positive
            pair_groups = [golden_pairs, neg_pairs]
            self.logger.info("Candidate neg pairs:{}, Golden pairs:{}".format(len(neg_pairs), len(golden_pairs)))
        else:
            pairs = pair_builder.get_pairs()
            pair_groups = [pairs]
            labels = [""]

        for i, plist in enumerate(pair_groups):
            label = labels[i]
            for i, pair in enumerate(plist):
                self.logger.info("Document Attaching {}/{}".format(i, len(plist)))
                material, docs_for_pair = SENETRawDataBuilder.find_document(label, pair[0], pair[1], sql_manger)
                for doc in docs_for_pair:
                    self.documents[doc] = docs_for_pair[doc]
                self.raws.append(material)
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
        for i in len(golden_pairs):
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
                lines = fin.readlines()
                for i, line in enumerate(lines):
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

        self.logger.info("Golden pair number:{}".format(len(pair_set)))
        return pair_set


class DataPrepare:
    """
    Prepare a data set for machine learning tasks. Provide functionality like ten fold validation.
    The dataset can be pickled and imported. The raw_material should match the requirement of feature_pipe
    """

    def __init__(self, pickle_path, feature_pipe, raw_materials, rebuild=True, thread_num=4):
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
        self.encoder = None
        if os.path.isfile(pickle_path) and not rebuild:
            self.logger.info("Loading data from path {}".format(pickle_path))
            self.__load_file()
        else:
            self.logger.info("Build dat set with {} threads".format(thread_num))
            chunk_size = math.ceil(len(raw_materials) / thread_num)
            labels = list(set([r.label() for r in raw_materials]))
            labels.sort()
            self.encoder = Encoder(labels)
            workers = []
            for i in range(thread_num):
                raw_parts = raw_materials[i * chunk_size: (i + 1) * chunk_size]
                t = threading.Thread(target=self.__build_data_set, args=(feature_pipe, raw_parts, i))
                workers.append(t)
                t.start()
            for t in workers:
                t.join()
            self.write_file()

    def __build_data_set(self, feature_pipe, raw_materials, thread_id):
        """
        Drive the production of data by applying the raw_material to feature_pipe. The
        :param feature_pipe: A list of function process RawMaterial Object
        :param raw_materials: A list of RawMaterial object
        :return:
        """
        for i, entry in enumerate(raw_materials):
            self.logger.info("Data Prepare - Thread-{}: {}/{}".format(thread_id, i, len(raw_materials)))
            feature_vec = feature_pipe.get_feature(entry)
            label = self.encoder.one_hot_encode(entry.label())
            readable_info = entry.info()
            data_entry = (feature_vec, label, readable_info)
            self.data_set.append(data_entry)

    def write_file(self):
        '''
        entry = (vector, label, (words1, words2)))
        :return:
        '''
        with open(self.pick_path, 'wb') as fout:
            dump_obj = (self.data_set, self.encoder)
            pickle.dump(dump_obj, fout)
        self.logger.info("Feature Vectors are pickled as {}".format(self.pick_path))

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
        self.logger.info("Totally {} pairs have been removed".format(cnt))
        return filtered

    def get_vec_length(self):
        first = self.data_set[0][0]
        return len(first)

    def all(self):
        return DataSet(self.data_set, self.encoder)

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

            # positive_test_entries = []
            # negative_test_entries = []
            # for test_entry in test_entries:
            #     if test_entry[1] == [0., 1.]:
            #         negative_test_entries.append(test_entry)
            #     else:
            #         positive_test_entries.append(test_entry)
            # pos_test_entries_num = int(
            #     (0.1 * len(negative_test_entries) / len(positive_test_entries) + 0.1) * len(positive_test_entries))
            # positive_test_entries = positive_test_entries[:pos_test_entries_num]
            # positive_test_entries.extend(negative_test_entries)
            # test_entries = positive_test_entries

            train_set = DataSet(train_entries, self.encoder)
            test_set = DataSet(test_entries, self.encoder)
            train_test_pair.append((train_set, test_set))
        return train_test_pair


class PairBuilder:
    def __read_words(self, file_path):
        res = set()
        with open(file_path, encoding='utf8') as fin:
            for line in fin.readlines():
                phrase = line.strip("\n\t\r ")
                res.add(phrase)
        return list(res)

    def __get_all_relationships(self):
        constras = os.path.join(VOCAB_DIR, "contrast.txt")
        hyper = os.path.join(VOCAB_DIR, "hyper.txt")
        related = os.path.join(VOCAB_DIR, "related.txt")
        synonym = os.path.join(VOCAB_DIR, "synonym.txt")
        one_pair_in_line = [constras, related, synonym]
        multi_pair_in_line = [hyper]
        rel = set()
        for f in one_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    word_pair = line.split(",")
                    relation = (word_pair[0], word_pair[1])
                    rel.add(relation)
        for f in multi_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    hyper, rest = line.split(":")
                    for w_r in rest.split(","):
                        rel.add((hyper[0], w_r))
        return rel

    def __init__(self, expension_list_txt):
        """

        :param expension_list_txt: A list of file path which will be used to match with all vocabulary to construct pairs
        """
        self.exp_list = self.__read_words(expension_list_txt)
        self.relations = self.__get_all_relationships()

    def get_pairs(self):
        pairs = []
        vocab = self.__read_words(os.path.join(VOCAB_DIR, "small_vocabulary_backup.txt"))
        for w_v in vocab:
            for w_e in self.exp_list:
                if (w_v, w_e) not in self.relations and (w_e, w_v) not in self.relations:
                    pairs.append((w_v, w_e))
        return pairs
