visited_exp = set()
visited_vocab = set()
with open("expension.txt") as exp_fin, open("small_vocabulary.txt") as vocab_fin:
    for line in exp_fin:
        line = line.strip("\n")
        visited_exp.add(line)
    print("Distinct expansion:", len(visited_exp))
    print("-----------------")
    for line in vocab_fin:
        line = line.strip("\n")
        if line not in visited_exp:
            visited_vocab.add(line)
    print("Distinct vocab:", len(visited_vocab))