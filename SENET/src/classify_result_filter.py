from config import *
import re


def find_pos_index(yes_line):
    line = re.sub("[^0-9\.\s]", "", yes_line)
    line = line.strip("\n")
    parts = line.split(" ")
    return parts.index("1")


def positive_score(line, pos_index):
    digits = line.split(",")[-1]
    prob = digits.strip("\[\]\n")
    nums = prob.split()
    pos_prob = float(nums[pos_index])
    return pos_prob


threshold = 0.5
result_file_path = os.path.join(RESULT_DIR, "classify.txt")
with open(result_file_path) as fin:
    data_lines = []
    for line in fin.readlines():
        line = line.strip("\n")
        if line.startswith("yes="):
            index = find_pos_index(line)
        elif line.startswith("["):
            data_lines.append(line)

    data_lines = sorted(data_lines, key=lambda x: positive_score(x, index), reverse=True)

    above_threshold = [x for x in data_lines if positive_score(x, index) > threshold]
    print("{} pairs have probabelity above 0.5".format(len(above_threshold)))
    for line in data_lines:
        print(line)
