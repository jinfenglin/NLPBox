from nltk.stem.porter import PorterStemmer
import os


def __stem_Tokens(words):
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(x) for x in words.split(" ")]


def same_pre_post(tokens1, tokens2):
    if tokens1[0] == tokens2[0] or tokens1[-1] == tokens2[-1]:
        return True
    return False


def single_token_same_pre_post_fix(tokens1, tokens2):
    if len(tokens1) == 1 and len(tokens2) == 1:
        w1 = tokens1[0]
        w2 = tokens2[0]
        if len(w1) > 3 and len(w2) > 3:
            return w1[:3] == w2[:3] or w1[-3:] == w2[-3:]
    return False


def share_tokens(tokens1, tokens2):
    for tk1 in tokens1:
        for tk2 in tokens2:
            if tk1 == tk2:
                return True
    return False


def is_heuristic_ones(w1, w2):
    w1_stems = __stem_Tokens(w1)
    w2_stems = __stem_Tokens(w2)
    if same_pre_post(w1_stems, w2_stems):
        return True
    return False


for file_name in os.listdir("."):
    if not os.path.isfile(file_name) or (not file_name.endswith("txt") and not file_name.endswith("csv")):
        continue
    # file_name = "FeedForward_Result{}.txt".format(i)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    with open(file_name) as fin, open("../filter_result/{}".format(file_name), "w") as fout: #,open("../filter_result/csv/{}".format(file_name), "w") as csv_fout:
        cnt = 0
        for line in fin:
            cnt += 1
            line = line.strip("\n")
            if "label, correctness, w1, w2" in line:
                if cnt == 1:
                    continue
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
                accuracy = (tp + tn) / (tp + tn + fn + fp)
                #csv_fout.write("{},{},{},{}\n".format(recall, precision, f1, accuracy))
                tn = 0
                tp = 0
                fn = 0
                fp = 0
            else:
                parts = [x for x in line.split("\t") if len(x) > 0]
                if len(parts) < 5:
                    print(parts)
                    continue
                pre_label = parts[0]
                correctness = parts[1]
                score = parts[2]
                w1 = parts[3]
                w2 = parts[4]
                if is_heuristic_ones(w1, w2):
                    continue
                if correctness == "Correct":
                    if pre_label == "yes":
                        tp += 1
                    else:
                        tn += 1
                else:
                    if pre_label == "yes":
                        fp += 1
                    else:
                        fn += 1
                fout.write(line + "\n")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        #csv_fout.write("{},{},{},{}\n".format(recall, precision, f1, accuracy))
