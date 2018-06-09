import os

for file in os.listdir("."):
    if os.path.isfile(file) and "FeedForward" in file and file.endswith("txt"):
        lines = []
        with open(file) as fin, open("./csv/{}".format(file),'w') as fout:
            for line in fin:
                lines.append(line)
            slice_size = int(len(lines) / 10)
            for i in range(10):
                slice = lines[i * slice_size:(i + 1) * slice_size]
                tp = 0
                tn = 0
                fn = 0
                fp = 0
                t_correct = 0

                for line in slice:
                    parts = line.split("\t")
                    predict = parts[0]
                    correctness = parts[1]
                    if predict == 'yes':
                        if correctness == "Correct":
                            t_correct += 1
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if correctness == "Correct":
                            t_correct += 1
                            tn += 1
                        else:
                            fn += 1
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f1 = 2 * (recall * precision) / (recall + precision)
                accuracy = t_correct / slice_size
                fout.write("{},{},{},{}\n".format(recall,precision,f1,accuracy))
