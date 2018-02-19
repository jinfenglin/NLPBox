import random
import re
from common import *
from nltk.stem.porter import PorterStemmer
import pickle

from data_entry_structures import DataSet
from feature_extractors import FeaturePipe
import logging
from dict_operations import collection_to_index_dict, invert_dict


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
        encoder = Encoder([r.label() for r in raw_materials])
        for entry in raw_materials:
            feature_vec = self.build_feature_vector(feature_pipe, entry)
            label = encoder.one_hot_encode(entry.label())
            readable_info = entry.info()
            data_entry = (feature_vec, label, readable_info)
            self.data_set.append(data_entry)
        # self.keyword_path = VOCAB_DIR + os.sep + "vocabulary.txt"
        # self.keys = []
        # with open(self.keyword_path, 'r', encoding='utf-8') as kwin:
        #     for line in kwin:
        #         self.keys.append(line.strip(" \n\r\t"))

        self.golden_pair_files = ["synonym.txt", "contrast.txt", "related.txt"]
        golden_pairs = self.build_golden()
        neg_pairs = self.build_neg_with_random_pair(golden_pairs)
        labels = [[0., 1.], [1., 0.]]  # [0,1] is negative and [1,0] is positive
        print("Candidate neg pairs:{}, Golden pairs:{}".format(len(neg_pairs), len(golden_pairs)))
        cnt_n = cnt_p = 0
        for i, plist in enumerate([neg_pairs, golden_pairs]):
            label = labels[i]
            for pair in plist:
                try:
                    words1 = pair[0].strip(" \n")
                    words2 = pair[1].strip(" \n")
                    vector = []
                    vector.extend(self.build_feature_vector(words1, words2))
                    self.data_set.append(
                        (vector, label, (words1, words2)))  # This will be parsed by next_batch() in dataset object
                    if i == 0:
                        cnt_n += 1
                    else:
                        cnt_p += 1
                except Exception as e:
                    print(e)
        print("Negative pairs:{} Golden Pairs:{}".format(cnt_n, cnt_p))
        random.shuffle(self.data_set)
        self.write_file()

    def write_file(self):
        '''
        entry = (vector, label, (words1, words2)))
        :return:
        '''
        with open(self.pick_path, 'wb') as fout:
            pickle.dump(self.data_set, fout)

    def __load_file(self):
        with open(self.pick_path, 'rb') as fin:
            self.data_set = pickle.load(fin)

    def build_neg_with_random_pair(self, golden_pairs):
        def get_random_word(gold, num):
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
                g1_negs = get_random_word(pair[0], 1)
                # g2_negs = get_random_word(pair[1], 3)
                neg_pairs.extend(g1_negs)
                # neg_pairs.extend(g2_negs)
            except Exception as e:
                pass
        return neg_pairs

    def build_golden(self):
        pair_set = set()
        for g_pair_name in self.golden_pair_files:
            path = VOCAB_DIR + os.sep + g_pair_name
            with open(path, encoding='utf8') as fin:
                for line in fin.readlines():
                    words1, words2 = line.strip(" \n").split(",")
                    if (words2, words1) not in pair_set:
                        pair_set.add((words1, words2))

        with open(VOCAB_DIR + os.sep + "hyper.txt") as fin:
            for line in fin.readlines():
                words1, rest = line.strip(" \n").split(":")
                if rest == "":
                    continue
                for word in rest.strip(" ").split(","):
                    wp = (words1.strip(" \n"), word.strip("\n"))
                    wp_r = (wp[1], wp[0])
                    if wp_r not in pair_set:
                        pair_set.add(wp)

        print("Golden pair number:{}".format(len(pair_set)))
        if self.remove_same_pre_post:
            pair_set = self.remove_pair_with_same_pre_post(pair_set)
        return pair_set

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

    def clean_word(self, word):
        word = re.sub(r'\([^)]*\)', '', word)
        tokens = word.split(" ")
        tokens = [token.lower() for token in tokens if len(token) > 0 and not token.isupper()]
        return " ".join(tokens)

    def build_feature_vector(self, feature_pipe, raw_material):
        """
        :return:
        """
        define1 = ""
        define2 = ""
        for dir in BING_WORD_DIR:
            try:
                with open(dir + os.sep + WordCleaner.to_file_name_format(words1) + ".txt", encoding='utf8') as f1:
                    define1 += f1.read()
            except Exception as e:
                print("word \'{}\' try to access file \'{}\', get error {}".format(words1,
                                                                                   WordCleaner.to_file_name_format(
                                                                                       words1) + ".txt", e))

            for dir in BING_WORD_DIR:
                try:
                    with open(dir + os.sep + WordCleaner.to_file_name_format(words2) + ".txt", encoding='utf8') as f2:
                        define2 += f2.read()
                except Exception as e:
                    print("word \'{}\' try to access file \'{}\', get error {}".format(words2,
                                                                                       WordCleaner.to_file_name_format(
                                                                                           words2) + ".txt", e))

        return FeaturePipe().get_feature(words1, define1, words2, define2)

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

            train_set = DataSet(train_entries)
            test_set = DataSet(test_entries)
            train_test_pair.append((train_set, test_set))
        return train_test_pair
