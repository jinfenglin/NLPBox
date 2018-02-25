import sys

sys.path.append("../../Utils")
sys.path.append("../../Cleaner")
import nltk
import spacy
from spacy.tokens import Doc

from Cleaner.cleaner import *
from nltk.corpus import wordnet

from data_entry_structures import SENETWordPairRaw
import logging


class SENETFeaturePipe:
    """
    A feature pipe line process WordPairRawMaterial and serve features for SENET work tasks.
    """

    def __init__(self, documents, model="en_core_web_lg"):
        logging.getLogger(__name__).info("Loaded language model:{}".format(model))
        self.nlp = spacy.load(model)
        self.documents = documents
        self.processed_docs = {}  # Serve as a cache, store the processed documents as
        self.func_pip = [
            self.stackoverflow_questions_noun_phrase_similarity,
            self.stackoverflow_answer_noun_phrase_similarity,
            self.stackoverflow_related_link_similarity,
            self.quora_answer_noun_phrase_similarity,
            self.quora_questions_noun_phrase_similarity,
            self.quora_related_question_similarity,

            self.definition_similarity,
            self.w1_to_w2Definition_count,
            self.w2_to_w1Definition_count,
            self.w1_to_w2_definition_noun_similarity,
            self.w2_to_w1_definition_noun_similarity,
            self.w1_to_w2_see_also,
            self.w2_to_w1_see_also,

            self.common_token_len,
            self.same_last_token,
            self.same_first_token,
            self.token_num_same,
            self.iterative_levenshtein,
            self.one_side_single_token,
        ]

    def __noun_phrase_similarity(self, doc1: Doc, doc2: Doc):
        w1_chunks = set([remove_stop_words(x.text.lower()) for x in set(doc1.noun_chunks)])
        w2_chunks = set([remove_stop_words(x.text.lower()) for x in set(doc2.noun_chunks)])
        w1_noun_str = "\n".join(w1_chunks)
        w2_noun_str = "\n".join(w2_chunks)
        return self.nlp(w1_noun_str).similarity(self.nlp(w2_noun_str))

    def stackoverflow_questions_noun_phrase_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_stk_clean_questoins"], w2_info["w2_stk_clean_questions"])

    def stackoverflow_answer_noun_phrase_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_stk_clean_answers"], w2_info["w2_stk_clean_answers"])

    def stackoverflow_related_link_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_stk_related_questions"], w2_info["w2_stk_related_questions"])

    def quora_questions_noun_phrase_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_quora_clean_questions"], w2_info["w2_quora_clean_questions"])

    def quora_answer_noun_phrase_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_quora_clean_answers"], w2_info["w2_quora_clean_answers"])

    def quora_related_question_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_quora_related_questions"],
                                             w2_info["w2_quora_related_questions"])

    def definition_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(w1_info["w1_pcMag_clean_doc"], w2_info["w2_pcMag_clean_doc"])

    def w1_to_w2Definition_count(self, w1, w2, w1_info, w2_info):
        """
        Cal the similarity between w1 and w2 definition
        :return:
        """
        matches = re.findall("\s+{}\s".format(w1), w2_info["w2_pcMag_clean_doc"].text, re.IGNORECASE)
        return len(matches)

    def w2_to_w1Definition_count(self, w1, w2, w1_info, w2_info):
        """
        Cal the similarity between w2 and w1 defnition
        :return:
        """
        matches = re.findall("\s+{}\s".format(w2), w1_info["w1_pcMag_clean_doc"].text, re.IGNORECASE)
        return len(matches)

    def w1_to_w2_see_also(self, w1, w2, w1_info, w2_info):
        sent = self.__extract_sent_with_see_also(w2_info["w2_pcMag_clean_doc"].text)
        phrase = re.split(",|and", sent)
        return w1 in phrase

    def w2_to_w1_see_also(self, w1, w2, w1_info, w2_info):
        sent = self.__extract_sent_with_see_also(w1_info["w1_pcMag_clean_doc"].text)
        phrase = re.split(",|and", sent)
        return w2 in phrase

    def w1_to_w2_definition_noun_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(self.nlp(w1), w2_info["w2_pcMag_clean_doc"])

    def w2_to_w1_definition_noun_similarity(self, w1, w2, w1_info, w2_info):
        return self.__noun_phrase_similarity(self.nlp(w2), w1_info["w1_pcMag_clean_doc"])

    def common_token_len(self, w1, w2, w1_info, w2_info):
        """
        Number of common tokens. Split on white space then stem each token
        :return:
        """
        w1_tk = set(w1_info["w1_stem_tokens"])
        w2_tk = set(w2_info["w2_stem_tokens"])
        common_len = len(w1_tk.intersection(w2_tk))
        return common_len

    def same_first_token(self, w1, w2, w1_info, w2_info):
        if w1_info["w1_stem_tokens"][0] == w2_info["w2_stem_tokens"][0]:
            return 1
        else:
            return 0

    def same_last_token(self, w1, w2, w1_info, w2_info):
        if w1_info["w1_stem_tokens"][-1] == w2_info["w2_stem_tokens"][-1]:
            return 1
        else:
            return 0

    def same_post_fix_len(self, w1, w2, w1_info, w2_info):
        w1_last_token = nltk.word_tokenize(w1)[-1]
        w2_last_token = nltk.word_tokenize(w2)[-1]
        steps = max(5, min(len(w1_last_token), len(w2_last_token)))
        cnt = 0
        for i in range(steps):
            if w1_last_token[-i] == w2_last_token[-i]:
                cnt += 1
        return max(0, cnt - 1)

    def token_num_same(self, w1, w2, w1_info, w2_info):
        # Check if two words have same length
        return len(w1_info["w1_stem_tokens"]) == len(w2_info["w2_stem_tokens"])

    def iterative_levenshtein(self, w1, w2, w1_info, w2_info):
        """
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        rows = len(w1) + 1
        cols = len(w2) + 1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for i in range(1, rows):
            dist[i][0] = i
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if w1[row - 1] == w2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[row][col]

    def wordnet_related_tokens_intersection(self, w1, w2, w1_info, w2_info):
        w1_tk = nltk.word_tokenize(w1)
        w2_tk = nltk.word_tokenize(w2)

        w1_related_set = self.__get_wordnet_related_set(w1_tk)
        w2_related_set = self.__get_wordnet_related_set(w2_tk)

        return len(w1_related_set.intersection(w2_related_set))

    def wordnet_last_token_h_similarity(self, w1, w2, w1_info, w2_info):
        w1_tk = nltk.word_tokenize(w1)[-1:]
        w2_tk = nltk.word_tokenize(w2)[-1:]
        score = 0

        w1_syn = self.__get_synsets(w1_tk)
        w2_syn = self.__get_synsets(w2_tk)

        for w1_s in w1_syn:
            for w2_s in w2_syn:
                cur_score = w1_s.wup_similarity(w2_s)
                if cur_score != None:
                    score = max(score, cur_score)
        return score

    def first_phrase_token_num(self, w1, w2, w1_info, w2_info):
        return len(w1_info["w1_stem_tokens"])

    def second_phrase_token_num(self, w1, w2, w1_info, w2_info):
        return len(w2_info["w2_stem_tokens"])

    def one_side_single_token(self, w1, w2, w1_info, w2_info):
        l_tk1 = len(w1_info["w1_stem_tokens"])
        l_tk2 = len(w2_info["w2_stem_tokens"])
        cnt = 0
        if (l_tk1 == 1):
            cnt += 1
        if (l_tk2 == 1):
            cnt += 1
        return cnt

    def get_feature(self, wp_raw_material: SENETWordPairRaw):
        # Prepare shared nlp resources before running pipeline for one raw_material
        feature_vec = []
        w1 = wp_raw_material.get_w1_str()
        w2 = wp_raw_material.get_w2_str()
        w1_info = {}
        w2_info = {}

        if w1 in self.processed_docs:
            w1_info = self.processed_docs[w1]
        else:
            w1_info["w1_stk_clean_questoins"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_stackoverflow_questions(w1, self.documents)))
            w1_info["w1_stem_tokens"] = stem_string(w1, regx_split_chars="[\s-]")
            w1_info["w1_stk_clean_answers"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_stackoverflow_answers(w1, self.documents)))
            w1_info["w1_stk_related_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_stackoverflow_related_links(w1, self.documents)))
            w1_info["w1_quora_clean_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_quora_questions(w1, self.documents)))
            w1_info["w1_quora_clean_answers"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_quora_answers(w1, self.documents)))
            w1_info["w1_quora_related_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_quora_related_links(w1, self.documents)))
            w1_info["w1_pcMag_clean_doc"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_pcMag_definition(w1, self.documents)))

        if w2 in self.processed_docs:
            w2_info = self.processed_docs[w2]
        else:
            w2_info["w2_stem_tokens"] = stem_string(w2, regx_split_chars="[\s-]")
            w2_info["w2_stk_clean_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_stackoverflow_questions(w2, self.documents)))
            w2_info["w2_stk_clean_answers"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_stackoverflow_answers(w2, self.documents)))
            w2_info["w2_stk_related_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_stackoverflow_related_links(w2, self.documents)))
            w2_info["w2_quora_clean_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_quora_questions(w2, self.documents)))
            w2_info["w2_quora_clean_answers"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_quora_answers(w2, self.documents)))
            w2_info["w2_quora_related_questions"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_quora_related_links(w2, self.documents)))
            w2_info["w2_pcMag_clean_doc"] = self.nlp(
                self.__clean_docs(wp_raw_material.get_pcMag_definition(w2, self.documents)))
        for func in self.func_pip:
            feature_vec.append(func(w1, w2, w1_info, w2_info))
        return feature_vec

    def __clean_docs(self, docs):
        if type(docs) is not list:
            docs = [docs]
        clean_doc = []
        for doc in docs:
            doc = keep_only_given_chars(doc)
            doc = merge_white_space(doc)
            clean_doc.append(doc)
        return "\n".join(clean_doc)

    def __get_wordnet_related_set(self, words):
        related_set = set()
        for word in words:
            word_morphy = wordnet.morphy(word)
            if word_morphy == None:
                word_morphy = word
            for syn in wordnet.synsets(word_morphy):
                for l in syn.lemmas():
                    related_set.add(l.name())
                    if l.antonyms():
                        related_set.add(l.antonyms()[0].name())
        return related_set

    def __get_synsets(self, words):
        syn_sets = set()
        for word in words:
            word_morphy = wordnet.morphy(word)
            if word_morphy == None:
                word_morphy = word
            for syn in wordnet.synsets(word_morphy):
                syn_sets.add(syn)
        return syn_sets

    def __get_subject_of_sentence(self, sent):
        sent = self.nlp(sent)
        for word in sent:
            if word.dep_ == "nsubj":
                return word.text
        return None

    def __extract_sent_with_see_also(self, doc):
        pattern = re.compile("^\s*(([sS]ee [also]*)|([Aa]lso know as))+[^\.]+\.")
        res = pattern.match(doc)
        if res:
            sent = res.group(0)
            head_pattren = re.compile("^\s*(([sS]ee [also]*)|([Aa]lso know as))")
            start_index = len(head_pattren.match(sent).group(0))
            return sent[start_index:]
        else:
            return ""


if __name__ == "__main__":
    pass
