import json
import os
import numpy as np

dataset = "skype_review"
data_dir = "C:\\Users\\jichuanzeng\\Dropbox\\ICSE'19\\src\\skype\\reviews"
out_fn = dataset + ".json"

fn = os.path.join(data_dir, "Skype_for_iPhone_it_clean_data.txt")

data_dic = {}
data_dic["dataset"] = dataset
data_dic["data"] = []
data_ptr = data_dic["data"]
with open(fn) as fin:
    lines = fin.readlines()
    for ind, line in enumerate(lines):
        iterms = line.strip().split("******")
        if len(iterms) == 7:
            data_ptr.append({"title": iterms[3], "text": iterms[2], "category": "none", "meta": iterms[5]})
        elif 3 == len(iterms):
            data_ptr.append({"title": "none", "text": iterms[2], "category": "none", "meta": "none"})
        else:
            print("format corrupted in line %d ..." % ind)

json.dump(data_dic, open(os.path.join(data_dir, out_fn), 'w'))

print("finished")
