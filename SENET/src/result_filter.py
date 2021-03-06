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


def is_heuristcs(w1, w2):
    w1_tk = w1.split()
    w2_tk = w2.split()
    if w1_tk[0] == w2_tk[0] or w1_tk[-1] == w2_tk[-1]:
        return True
    return False


def get_words(line):
    p = re.compile("\([\'\"](.+)[\'\"], [\'\"](.+)[\'\"]\)")
    res = p.match(line)
    if res is None:
        print(line)
    w1 = res.group(1)
    w2 = res.group(2)
    return (w1, w2)


thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
result_file_path = os.path.join(RESULT_DIR, "total_result.txt")
analysis_fold = os.path.join(RESULT_DIR, "analysis")
with open(result_file_path) as fin:
    data_lines = []
    visited = set()
    for line in fin.readlines():
        line = line.strip("\n")
        if line.startswith("yes="):
            index = find_pos_index(line)
        elif line.startswith("("):
            word_pair = get_words(line)
            w1 = word_pair[0]
            w2 = word_pair[1]
            if w1 != w2 and word_pair not in visited and (w2, w1) not in visited:
                visited.add(word_pair)
                data_lines.append(line)

    data_lines = sorted(data_lines, key=lambda x: positive_score(x, index), reverse=True)

    for threshold in thresholds:
        with open(os.path.join(analysis_fold, str(threshold) + ".txt"), 'w', encoding="utf8") as fout, open(
                os.path.join(analysis_fold, str(threshold) + "_score.txt"), 'w', encoding="utf8") as f_no_score:
            above_threshold = [x for x in data_lines if positive_score(x, index) > threshold]
            for line in above_threshold:
                fout.write(line + "\n")
                w1, w2 = get_words(line)
                f_no_score.write("{},{},{}\n".format(w1, w2, positive_score(line, index)))
            fout.write("{} pairs have probabelity above {}".format(len(above_threshold), threshold))
    with open(os.path.join(analysis_fold, "sorted.txt"), "w", encoding="utf8") as fout:
        for line in data_lines:
            fout.write(line + "\n")

        with open(os.path.join(analysis_fold, "top_500.txt"), "w", encoding="utf8") as fout:
            cnt = 0
            for line in data_lines[:500]:
                p = re.compile("\([\'\"](.+)[\'\"], [\'\"](.+)[\'\"]\)")
                res = p.match(line)
                if res is None:
                    print(line)
                cnt += 1
                w1 = res.group(1)
                w2 = res.group(2)
                fout.write("{}\t{}\n".format(w1, w2))
