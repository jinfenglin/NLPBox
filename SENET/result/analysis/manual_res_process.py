with open("0.95-manual.csv") as fin:
    syn = set()
    ukn = set()
    hyper = set()
    related = set()
    not_re = set()
    for line in fin.readlines():
        line = line.strip()
        w1, w2, re = line.split(",")
        wp = (w1, w2)
        if re == "s":
            syn.add(wp)
        elif re == "unknown":
            ukn.add(wp)
        elif re == "re":
            related.add(wp)
        elif re == "hyper":
            hyper.add(wp)
        elif re == "hypon":
            hyper.add((w2, w1))
        else:
            not_re.add(wp)
    print(len(syn), len(ukn), len(hyper), len(related), len(not_re))
