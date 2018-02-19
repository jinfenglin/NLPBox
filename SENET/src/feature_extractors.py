from nltk.stem.porter import PorterStemmer
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import common
import re


class FeaturePipe:
    """
    Prototype of feature pipe line, if any resource is requested for computing a feature, add them in the subclass
    __init__ function and access them in the subclass functions. The raw_material should be customized for a pipe line.
    """

    def __init__(self):
        self.func_pip = []

    def get_feature(self, raw_material):
        feature_vec = []
        for func in self.func_pip:
            feature_vec.append(func(raw_material))
        return feature_vec


class SENETFeaturePipe:
    """
    A feature pipe line process WordPairRawMaterial and serve features for SENET work tasks.
    """

    def __init__(self):
        self.func_pip = [
            self.common_token_len,
            self.common_token_len_with_slash,

            self.same_postfix,
            self.same_prefix,
            self.token_num_same,
            self.include_each_other,

            # self.pos_compare, #Very time consuming

            self.doc_similarity,
            self.iterative_levenshtein,

            # self.doc_contain_tokens_w1,
            # self.doc_contain_tokens_w2,
            # self.wordnet_related_tokens_intersection,
            # self.wordnet_related_tokens_highest_similarity,
            # self.wordnet_related_tokens_lowest_similarity

            self.wordnet_last_token_h_similarity,
            self.one_side_single_token,
            self.token_is_substr
        ]

    def __stem_Tokens(self, words):
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(x) for x in words.split(" ")]

    def token_is_substr(self, wp_raw_material):
        w1_tk = re.split("[\s-]", w1)
        w2_tk = re.split("[\s-]", w2)
        count = 0
        for tk1 in w1_tk:
            for tk2 in w2_tk:
                if tk1 in tk2 or tk2 in tk1:
                    count += 1
        return count

    def common_token_len(self, wp_raw_material):
        """
        Number of common tokens. Split on white space then stem each token
        :return:
        """
        w1_tk = set(self.__stem_Tokens(w1))
        w2_tk = set(self.__stem_Tokens(w2))
        common_len = len(w1_tk.intersection(w2_tk))
        return common_len

    def common_token_len_with_slash(self, wp_raw_material):
        """
        If a token in the phrases have '-', split on '-' as well then calculate the interaction
        :param w1:
        :param d1:
        :param w2:
        :param d2:
        :return:
        """
        w1_tk = set(re.split("[\s-]", w1))
        w2_tk = set(re.split("[\s-]", w2))
        common_len = len(w1_tk.intersection(w2_tk))
        return common_len

    def same_prefix(self, wp_raw_material):
        w1_tk = self.__stem_Tokens(w1)
        w2_tk = self.__stem_Tokens(w2)
        if w1_tk[0] == w2_tk[0]:
            return 1
        else:
            return 0

    def same_postfix(self, wp_raw_material):
        w1_tk = self.__stem_Tokens(w1)
        w2_tk = self.__stem_Tokens(w2)
        if w1_tk[-1] == w2_tk[-1]:
            return 1
        else:
            return 0

    def get_feature(self, wp_raw_material):
        feature_vec = []
        for func in self.func_pip:
            feature_vec.append(func(wp_raw_material))
        return feature_vec

    def token_num_same(self, wp_raw_material):
        # Check if two words have same length
        d1_tk = self.__stem_Tokens(w1)
        d2_tk = self.__stem_Tokens(w2)
        return len(d1_tk) == len(d2_tk)

    def include_each_other(self, wp_raw_material):
        # Count how many time each word appear on each other's definition
        res = 0
        if d1.lower().count(w2) > 0:
            res += 1

        if d2.lower().count(w1):
            res += 1
        return res

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

    def pos_compare(self, wp_raw_material):
        score = 0
        d1_sent = d1.split('\n')
        d2_sent = d2.split('\n')

        w1_tk = self.__stem_Tokens(w1)
        w2_tk = self.__stem_Tokens(w2)

        w1_candidate_tags = []
        w2_candidate_tags = []

        for sent in d1_sent:
            if len(w1_candidate_tags) > 4:
                break
            if w1 in sent:
                sent_tokens = self.__stem_Tokens(sent)
                pos_tags = nltk.pos_tag(sent_tokens)
                tokens_indexs = self.__find_index_for_phrase(pos_tags, w1_tk)
                ph_tags = []
                for tk_index in tokens_indexs:
                    ph_tags.append(pos_tags[tk_index][1])
                if ph_tags not in w1_candidate_tags:
                    w1_candidate_tags.append(ph_tags)

        for sent in d2_sent:
            if len(w2_candidate_tags) > 4:
                break
            if w2 in sent:
                sent_tokens = self.__stem_Tokens(sent)
                pos_tags = nltk.pos_tag(sent_tokens)
                tokens_indexs = self.__find_index_for_phrase(pos_tags, w2_tk)
                ph_tags = []
                for tk_index in tokens_indexs:
                    ph_tags.append(pos_tags[tk_index][1])
                if ph_tags not in w2_candidate_tags:
                    w2_candidate_tags.append(ph_tags)

            for pos1 in w1_candidate_tags:
                for pos2 in w2_candidate_tags:
                    if len(pos1) > 0 and len(pos2) > 0 and pos1[-1] == pos2[-1]:
                        score += 1
        return score

    def doc_similarity(self, wp_raw_material):
        '''remove punctuation, lowercase, stem'''
        stemmer = nltk.stem.porter.PorterStemmer()
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

        def stem_tokens(tokens):
            return [stemmer.stem(item) for item in tokens]

        def normalize(text):
            return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

        vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

        def cosine_sim(text1, text2):
            tfidf = vectorizer.fit_transform([text1, text2])
            return ((tfidf * tfidf.T).A)[0, 1]

        if len(d1) == 0 or len(d2) == 0:
            return -1
        return cosine_sim(d1, d2)

    def iterative_levenshtein(self, wp_raw_material):
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
        return dist[row][col] < 5

    def doc_contain_tokens_w1(self, wp_raw_material):
        w1_tk = set(self.__stem_Tokens(w1))
        w1_cnt = 0
        for tk in w1_tk:
            if tk in d2.lower():
                w1_cnt += 1
        return w1_cnt / len(w1_tk)

    def doc_contain_tokens_w2(self, wp_raw_material):
        w2_tk = set(self.__stem_Tokens(w2))
        w2_cnt = 0
        for tk in w2_tk:
            if tk in d1.lower():
                w2_cnt += 1
        return w2_cnt / len(w2_tk)

    def wordnet_related_tokens_intersection(self, wp_raw_material):
        w1_tk = nltk.word_tokenize(w1)
        w2_tk = nltk.word_tokenize(w2)

        w1_related_set = common.get_related_set(w1_tk)
        w2_related_set = common.get_related_set(w2_tk)

        return len(w1_related_set.intersection(w2_related_set))

    def wordnet_related_tokens_highest_similarity(self, wp_raw_material):
        w1_tk = nltk.word_tokenize(w1)
        w2_tk = nltk.word_tokenize(w2)
        score = 0

        w1_syn = common.get_synsets(w1_tk)
        w2_syn = common.get_synsets(w2_tk)

        for w1_s in w1_syn:
            for w2_s in w2_syn:
                cur_score = w1_s.wup_similarity(w2_s)
                if cur_score != None:
                    score = max(score, cur_score)
        return score

    def wordnet_related_tokens_lowest_similarity(self, wp_raw_material):
        w1_tk = nltk.word_tokenize(w1)
        w2_tk = nltk.word_tokenize(w2)
        score = 1

        w1_syn = common.get_synsets(w1_tk)
        w2_syn = common.get_synsets(w2_tk)

        for w1_s in w1_syn:
            for w2_s in w2_syn:
                cur_score = w1_s.wup_similarity(w2_s)
                if cur_score != None:
                    score = min(score, cur_score)
        return score

    def wordnet_last_token_h_similarity(self, wp_raw_material):
        w1_tk = nltk.word_tokenize(w1)[-1:]
        w2_tk = nltk.word_tokenize(w2)[-1:]
        score = 0

        w1_syn = common.get_synsets(w1_tk)
        w2_syn = common.get_synsets(w2_tk)

        for w1_s in w1_syn:
            for w2_s in w2_syn:
                cur_score = w1_s.wup_similarity(w2_s)
                if cur_score != None:
                    score = max(score, cur_score)
        return score

    def first_phrase_token_num(self, wp_raw_material):
        return len(nltk.word_tokenize(w1))

    def second_phrase_token_num(self, wp_raw_material):
        return len(nltk.word_tokenize(w2))

    def one_side_single_token(self, wp_raw_material):
        l_tk1 = len(nltk.word_tokenize(w1))
        l_tk2 = len(nltk.word_tokenize(w2))
        cnt = 0
        if (l_tk1 == 1):
            cnt += 1
        if (l_tk2 == 1):
            cnt += 1
        return cnt
