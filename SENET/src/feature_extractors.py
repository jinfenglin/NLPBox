import nltk
from spacy.tokens import Span

import common
import spacy
from Cleaner.cleaner import *
from nltk.corpus import wordnet

from data_entry_structures import SENETWordPairRaw


class SENETFeaturePipe:
    """
    A feature pipe line process WordPairRawMaterial and serve features for SENET work tasks.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.func_pip = [
            self.stackoverflow_questions_noun_phrase_similarity,
            self.common_token_len,
            self.same_postfix,
            self.same_prefix,
            self.token_num_same,
            self.iterative_levenshtein,
            self.wordnet_last_token_h_similarity,
            self.one_side_single_token,
        ]

    def stackoverflow_questions_noun_phrase_similarity(self, wp_raw_material):
        w1_chunks = set([remove_stop_words(x.text.lower()) for x in set(self.w1_stk_clean_questoins.noun_chunks)])
        w2_chunks = set([remove_stop_words(x.text.lower()) for x in set(self.w2_stk_clean_questions.noun_chunks)])
        w1_noun_str = "\n".join(w1_chunks)
        w2_noun_str = "\n".join(w2_chunks)
        return self.nlp(w1_noun_str).similarity(self.nlp(w2_noun_str))

    def stackoverflow_answer_noun_phrase_similarity(self, wp_raw_material):
        pass

    def definition_similarity(self, raw_m):
        pass

    def same_subject_of_definition(self, raw_m):
        """
        The subject of the two phrases' definition is same or not
        :return:
        """
        pass

    def common_token_len(self, wp_raw_material):
        """
        Number of common tokens. Split on white space then stem each token
        :return:
        """
        w1_tk = set(self.w1_stem_tokens)
        w2_tk = set(self.w2_stem_tokens)
        common_len = len(w1_tk.intersection(w2_tk))
        return common_len

    def same_prefix(self, wp_raw_material):
        if self.w1_stem_tokens[0] == self.w2_stem_tokens[0]:
            return 1
        else:
            return 0

    def same_postfix(self, wp_raw_material):
        if self.w1_stem_tokens[-1] == self.w2_stem_tokens[-1]:
            return 1
        else:
            return 0

    def token_num_same(self, wp_raw_material):
        # Check if two words have same length
        return len(self.w1_stem_tokens) == len(self.w2_stem_tokens)

    def include_each_other_in_regular_text(self, wp_raw_material):
        # Count how many time each word appear on each other's definition
        # res = 0
        # if d1.lower().count(w2) > 0:
        #     res += 1
        #
        # if d2.lower().count(w1):
        #     res += 1
        # return res
        return -1

    def __find_index_for_phrase(self, tags_list, phrase_tokens):
        res = []
        start_indices = []
        for i, tk in enumerate(tags_list):
            if phrase_tokens[0] == tk[0]:
                start_indices.append(i)

        for i in start_indices:
            flag = True
            for j in range(len(phrase_tokens)):
                if i + j >= len(tags_list) or phrase_tokens[j] != tags_list[i + j][0]:
                    flag = False
                    break
            if flag:
                res = [n for n in range(i, i + len(phrase_tokens))]
                return res
        return res

    def iterative_levenshtein(self, wp_raw_material):
        """
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        rows = len(self.w1) + 1
        cols = len(self.w2) + 1
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
                if self.w1[row - 1] == self.w2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[row][col]

    def wordnet_related_tokens_intersection(self, wp_raw_material):
        w1_tk = nltk.word_tokenize(self.w1)
        w2_tk = nltk.word_tokenize(self.w2)

        w1_related_set = self.__get_wordnet_related_set(w1_tk)
        w2_related_set = self.__get_wordnet_related_set(w2_tk)

        return len(w1_related_set.intersection(w2_related_set))

    def wordnet_last_token_h_similarity(self, wp_raw_material):
        w1_tk = nltk.word_tokenize(self.w1)[-1:]
        w2_tk = nltk.word_tokenize(self.w2)[-1:]
        score = 0

        w1_syn = self.__get_synsets(w1_tk)
        w2_syn = self.__get_synsets(w2_tk)

        for w1_s in w1_syn:
            for w2_s in w2_syn:
                cur_score = w1_s.wup_similarity(w2_s)
                if cur_score != None:
                    score = max(score, cur_score)
        return score

    def first_phrase_token_num(self, wp_raw_material):
        return len(self.w1_stem_tokens)

    def second_phrase_token_num(self, wp_raw_material):
        return len(self.w2_stem_tokens)

    def one_side_single_token(self, wp_raw_material):
        l_tk1 = len(self.w1_stem_tokens)
        l_tk2 = len(self.w2_stem_tokens)
        cnt = 0
        if (l_tk1 == 1):
            cnt += 1
        if (l_tk2 == 1):
            cnt += 1
        return cnt

    def get_feature(self, wp_raw_material: SENETWordPairRaw):
        # Prepare shared nlp resources before running pipeline for one raw_material
        feature_vec = []
        self.w1 = wp_raw_material.get_w1_str()
        self.w2 = wp_raw_material.get_w2_str()
        self.w1_stem_tokens = stem_string(self.w1, regx_split_chars="[\s-]")
        self.w2_stem_tokens = stem_string(self.w2, regx_split_chars="[\s-]")

        ## Stackoverflow ##
        self.w1_stk_clean_questoins = self.nlp(self.__clean_docs(wp_raw_material.get_stackoverflow_questions(self.w1)))
        self.w2_stk_clean_questions = self.nlp(self.__clean_docs(wp_raw_material.get_stackoverflow_questions(self.w2)))
        self.w1_stk_clean_answers = self.nlp(self.__clean_docs(wp_raw_material.get_stackoverflow_answers(self.w1)))
        self.w2_stk_clean_answers = self.nlp(self.__clean_docs(wp_raw_material.get_stackoverflow_answers(self.w2)))
        self.w1_stk_related_questions = self.nlp(
            self.__clean_docs(wp_raw_material.get_stackoverflow_related_links(self.w1)))
        self.w2_stk_related_questions = self.nlp(
            self.__clean_docs(wp_raw_material.get_stackoverflow_related_links(self.w2)))

        ## Quora ##
        self.w1_quora_clean_questions = self.nlp(self.__clean_docs(wp_raw_material.get_quora_questions(self.w1)))
        self.w2_quora_clean_questions = self.nlp(self.__clean_docs(wp_raw_material.get_quora_questions(self.w2)))
        self.w1_quora_clean_answers = self.nlp(self.__clean_docs(wp_raw_material.get_quora_answers(self.w1)))
        self.w2_quora_clean_answers = self.nlp(self.__clean_docs(wp_raw_material.get_quora_answers(self.w2)))
        self.w1_quora_related_questions = self.nlp(self.__clean_docs(wp_raw_material.get_quora_related_links(self.w1)))
        self.w2_quora_related_questions = self.nlp(self.__clean_docs(wp_raw_material.get_quora_related_links(self.w2)))

        ## Regular ##
        self.w1_regular_clean_doc = self.nlp(self.__clean_docs(wp_raw_material.get_regular_doc_content(self.w1)))
        self.w2_regular_clean_doc = self.nlp(self.__clean_docs(wp_raw_material.get_regular_doc_content(self.w2)))

        ## PcMag ##
        self.w1_pcMag_clean_doc = self.nlp(self.__clean_docs(wp_raw_material.get_pcMag_definition(self.w1)))
        self.w2_pcMag_clean_doc = self.nlp(self.__clean_docs(wp_raw_material.get_pcMag_definition(self.w2)))

        for func in self.func_pip:
            feature_vec.append(func(wp_raw_material))
        return feature_vec

    def __clean_docs(self, docs):
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
