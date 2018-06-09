import os

thresholds = [x / 100 for x in range(0, 100, 2)]
type_dict = {}
for file_name in os.listdir("."):
    if not file_name.endswith(".txt") or not os.path.isfile(file_name):
        continue
    type = file_name.split("_")[0]
    if type not in type_dict:
        type_dict[type] = []
    with open(file_name) as fin:
        for line in fin:
            parts = line.split("\t")
            label = parts[0]
            if label == "yes" or label == "no":
                predict_score = parts[2]
                score_parts = predict_score.strip("[] ").split(" ")
                score_parts = [x for x in score_parts if x != ""]
                pos_score = float(score_parts[1])
                correctness = parts[1]
                type_dict[type].append((label, pos_score, correctness))
t_tp = 0
t_fp = 0
t_tn = 0
t_fn = 0
t_correct = 0
t_incorrect = 0
for type in type_dict:
    with open("{}_roc.csv".format(type), "w") as fout:
        for threshold in thresholds:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            entries = type_dict[type]
            for entry in entries:
                label = entry[0]
                pos_score = entry[1]
                correctness = entry[2]

                if correctness == "Correct":
                    t_correct += 1
                    if label == "yes":
                        t_tp += 1
                    else:
                        t_tn += 1
                else:
                    t_incorrect += 1
                    if label == "yes":
                        t_fp += 1
                    else:
                        t_fn += 1

                if correctness == "Correct":
                    real_label = label;
                else:
                    if label == "yes":
                        real_label = 'no'
                    else:
                        real_label = 'yes'

                if pos_score >= threshold:
                    if real_label == "yes":
                        tp += 1
                    else:
                        fp += 1
                else:
                    if real_label == 'yes':
                        fn += 1
                    else:
                        tn += 1
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            fout.write("{},{}\n".format(recall, precision))

recall = t_tp / (t_tp + t_fn)
precision = t_tp / (t_tp + t_fp)
accuracy = t_correct / (t_incorrect + t_correct)
f1 = 2 * precision * recall / (precision + recall)
print("average:{},{},{},{}".format(recall, precision, f1, accuracy))
print("Done")
