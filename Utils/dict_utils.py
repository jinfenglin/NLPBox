def collection_to_index_dict(collection):
    """
    Given a collection, give each type of the data a unique integer id
    :param collection:
    :return: a diction project object to a number
    """
    unique_collection = set(collection)
    res = {}
    for i, entry in enumerate(unique_collection):
        res[entry] = i
    return res


def invert_dict(origin_dict):
    inv_map = {v: k for k, v in origin_dict.items()}
    return inv_map
