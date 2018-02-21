import random
import numpy as np

class RawMaterial:
    def __init__(self, raw_label, raw_content):
        self.raw_label = raw_label
        self.raw_content = raw_content

    def label(self):
        return self.raw_label

    def info(self):
        return self.raw_content


class SENETWordPairRaw(RawMaterial):
    """
    One type of raw data that consists of a pair of words. The keys of the doc_pack can be found in the web_page_parser.py
    """

    # TODO A shared text resource file for this 2 class should be created
    def __init__(self, raw_label, str_pair, doc_packs):
        """
        :param raw_label: An obejct stands for label
        :param str_pair: A pair of string
        :param A large, complex json string, include all info scrapped for the terms in the str_pair
        """
        assert type(str_pair) is tuple
        assert type(doc_packs) is tuple
        super().__init__(raw_label, str_pair)
        self.wd_doc_dict = {self.get_w1_str(): doc_packs[0], self.get_w2_str(): doc_packs[1]}

    def get_w1_str(self):
        return self.raw_content[0]

    def get_w2_str(self):
        return self.raw_content[1]

    def get_stackoverflow_questions(self, word):
        """
        Get stackoverflow questions for the given word
        :param word:
        :return: A list of questions related with the word
        """
        res = []
        doc_dict = self.wd_doc_dict[word]
        page_contents = doc_dict["stackoverflow"]
        nested = [x["questions"] for x in page_contents]
        if len(nested) == 0:
            return []
        else:
            return np.concatenate(nested).ravel().tolist()

    def get_stackoverflow_answers(self, word):
        """
        Get stackoverflow answers for the given word
        :param word:
        :return: A list of answers related with the word
        """
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["stackoverflow"]
        nested = [x["answers"] for x in page_content]
        if len(nested) == 0:
            return []
        else:
            return np.concatenate(nested).ravel().tolist()

    def get_stackoverflow_related_links(self, word):
        """
        Get stackoverflow answers for the given word
        :param word:
        :return: A list of related question links. Each list contains pairs: (questions, link)
        """
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["stackoverflow"]
        nested = [x["related_questions"] for x in page_content]
        flat = []
        for sublist in nested:
            for item in sublist:
                flat.append(item[0])
        return flat

    def get_quora_questions(self, word):
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["quora"]
        nested = [x["questions"] for x in page_content]
        if len(nested) == 0:
            return []
        else:
            return np.concatenate(nested).ravel().tolist()

    def get_quora_answers(self, word):
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["quora"]
        nested = [x["answers"] for x in page_content]
        if len(nested) == 0:
            return []
        else:
            return np.concatenate(nested).ravel().tolist()

    def get_quora_related_links(self, word):
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["quora"]
        nested = [x["related_questions"] for x in page_content]
        flat = []
        for sublist in nested:
            for item in sublist:
                flat.append(item[0])
        return flat

    def get_pcMag_definition(self, word):
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["pcMag"]
        if len(page_content) == 0:
            return ""
        return page_content[0]["definition"]# Definition only parse 1 link

    def get_regular_doc_content(self, word):
        doc_dict = self.wd_doc_dict[word]
        page_content = doc_dict["regular"]
        nested = [x["content"] for x in page_content]
        if len(nested) == 0:
            return []
        else:
            return np.concatenate(nested).ravel().tolist()


class DataSet:
    def __init__(self, entry_list, label_encoder):
        self.label_encoder = label_encoder
        self.cur_batch_start = 0
        self.data = entry_list

    def next_batch(self, batch_size):
        """
        Get next batch of the data. If the times of requesting new batch larger than the dataset,
        shuffle the dataset and do it again
        :param batch_size:
        :return:
        """
        start = self.cur_batch_start
        self.cur_batch_start += batch_size
        if self.cur_batch_start > len(self.data):
            random.shuffle(self.data)
            start = 0
            self.cur_batch_start = batch_size
            assert batch_size <= len(self.data)
        end = self.cur_batch_start
        batch_data = self.data[start:end]
        # Provide the vector, label and the readable words
        return np.array([x[0] for x in batch_data]), np.array([x[1] for x in batch_data]), [x[2] for x in batch_data]

    def all(self):
        return np.array([x[0] for x in self.data]), np.array([x[1] for x in self.data]), [x[2] for x in self.data]
