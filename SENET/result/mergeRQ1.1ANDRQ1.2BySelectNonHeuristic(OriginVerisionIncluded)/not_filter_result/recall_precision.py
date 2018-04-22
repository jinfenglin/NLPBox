import os

thresholds = [x / 100 for x in range(0, 100, 2)]
type_dict = {}
for file_name in os.listdir("./"):
    if not file_name.endswith("csv"):
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

for type in type_dict:
    with open("{}.roc".format(type),"w") as fout:
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
