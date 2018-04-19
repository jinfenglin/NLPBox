import re, os
from config import *

in_path = os.path.join(RESULT_DIR, "analysis", "0.8.txt")
out_path = os.path.join(RESULT_DIR, "analysis", "0.8_blank.txt")
visited = set()
with open(in_path) as fin, open(out_path, "w") as fout:
    p = re.compile("\[\([\'\"](.+)[\'\"], [\'\"](.+)[\'\"]\)\]")
    for line in fin.readlines():
        if line.startswith("["):
            res = p.match(line)
            if res is None:
                print(line)
            w1 = res.group(1)
            w2 = res.group(2)
            if (w1, w2) not in visited and (w2, w1) not in visited:
                fout.write("{},{}\n".format(w1, w2))
                visited.add((w1, w2))
