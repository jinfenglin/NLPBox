import re

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))


def clean_phrase(phrase):
    phrase = phrase.strip("\n\t\r ")
    phrase = keep_only_given_chars(phrase, re_char_white_list="\w\-&\'\s")
    phrase = merge_white_space(phrase)
    return phrase


def merge_white_space(text):
    text = re.sub("[^\S\r\n]+", " ", text)
    return text


def remove_space_around_char(char="-", text=""):
    """
    Remove the space around a char in the text, eg, text = "health - care" will become
    health-care
    :param text:
    :param char:
    :return:
    """
    text = re.sub("\s*{}\s*".format(char), char, text)
    return text


def keep_only_given_chars(text="", re_char_white_list="\w\-&\s\.\,\'\""):
    """
    Remove all characters not in the white list.
    :param chars_re: Regulra expression indicating the legal chars
    :param text:
    :return:
    """
    text = re.sub("[^{}]".format(re_char_white_list), " ", text)
    return merge_white_space(text)


def esapce_sql_variable_quote(sql_variable):
    """
    Sql database usaually encoding single quote by adding an extra single quote.
    :param sql_variable:
    :return:
    """
    return re.sub("\'", "\'\'", sql_variable)


def stem_tokens(tokens):
    """
    Stem tokens
    :param self:
    :param words:
    :return:
    """
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(x) for x in tokens]


def stem_string(str, regx_split_chars="[\s]"):
    merge_white_space(str)
    tokens = re.split(regx_split_chars, str)
    return stem_tokens(tokens)


def remove_stop_words(doc):
    words = word_tokenize(doc)
    words_filtered = []

    for w in words:
        if w.lower() not in stop_words:
            words_filtered.append(w)
    return " ".join(words_filtered)


if __name__ == "__main__":
    print(keep_only_given_chars(text="ldjfajfjla $%^&*()213213\"?<\}\{\}"))
