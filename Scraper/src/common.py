import os

import functools
from threading import Thread

PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def split_dict(target_dict, nums):
    """
    Split a dictionary into multiple parts
    :param target_dict:
    :param nums:
    :return:
    """
    if nums <= 1:
        return target_dict
    sub_dicts = []
    for i in range(nums):
        sub_dicts.append(dict())
    for i, key in enumerate(target_dict):
        parti_num = i % nums
        sub_dict = sub_dicts[parti_num]
        sub_dict[key] = target_dict[key]
    return sub_dicts
