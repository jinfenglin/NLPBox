thresholds = [x / 100 for x in range(0, 100, 2)]
with open("recall_precision.csv",'w') as fout:
    for threshold in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        with open("../RNN_result15.txt") as fin:
            for line in fin.readlines():
                parts = line.split("\t")
                label = parts[0]
                if label == "yes" or label == "no":
                    predict_score = parts[5]
                    score_parts = predict_score.strip("[] ").split(" ")
                    score_parts = [x for x in score_parts if x != ""]
                    pos_score = float(score_parts[1])
                    if pos_score > threshold:
                        if label == "yes":
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if label == "yes":
                            fn += 1
                        else:
                            tn += 1
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        fout.write("{},{}\n".format(recall, precision))
