import re


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


def keep_only_given_chars(re_char_white_list="\w\-&\s\.\,\'\"", text=""):
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


if __name__ == "__main__":
    print(keep_only_given_chars(text="ldjfajfjla $%^&*()213213\"?<\}\{\}"))
