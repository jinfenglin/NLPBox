import os

doc_types = dict()
for file_name in os.listdir("."):
    if not os.path.isfile(file_name) or not file_name.endswith("csv") or not "_" in file_name:
        continue
    type = "_".join(file_name.split("_")[:-1])
    if type not in doc_types:
        doc_types[type] = []
    doc_types[type].append(file_name)

for type in doc_types.keys():
    file_names = doc_types[type]
    with open(type + "Merged.csv", "w") as fout:
        for file_name in file_names:
            with open(file_name) as fin:
                for line in fin:
                    fout.write(line)
