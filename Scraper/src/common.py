import os

PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCRAP_TMP = os.path.join(DATA_DIR, "db_dump.tmp")


def split_dict(target_dict, nums):
    """
    Split a dictionary whose value is a list into multiple smaller dictionary with same key but shorter list.
    E.g origin = {k1:[1,2,3,4], k2:[5,6,7]} may be splitted as p1 = {k1:[1,3],ke:[5,7]} and p2{k1:[2,4],k2:[6]}
    :param target_dict:
    :param nums:
    :return:
    """
    if nums <= 1:
        return target_dict
    sub_dicts = []
    for i in range(nums):
        sub_dicts.append(dict())

    for key in target_dict:
        origin_list = target_dict[key]
        for item_index, item in enumerate(origin_list):
            sub_dict_index = item_index % nums
            sub_dict = sub_dicts[sub_dict_index]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(item)
    return sub_dicts
