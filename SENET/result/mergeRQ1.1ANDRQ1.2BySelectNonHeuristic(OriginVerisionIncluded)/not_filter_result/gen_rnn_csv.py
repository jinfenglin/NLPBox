for i in range(5, 15):
    with open("RNN_result{}.txt".format(i)) as fin, open("csv/RNN_result{}.csv".format(i), "w") as fout:
        for line in fin:
            if line.startswith("recall"):
                parts = line.split(",")
                parts = [x.split(":")[1].strip("\n\t\r ") for x in parts]
                fout.write(",".join(parts) + "\n")
