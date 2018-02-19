def pair_in_collection(pair, pair_collection, have_direction=True):
    """
    Check if a pair is in a collection or not.
    :param pair: A tuple
    :param pair_set: A collection of tuple
    :param have_direction if true (a,b) is different from (b,a) otherwise consider them as same one
    :return: boolean
    """
    if have_direction:
        return pair in pair_collection
    else:
        reversed_pair = (pair[0], pair[1])
        return pair in pair_collection or reversed_pair in pair_collection
