import json
import os
import numpy as np

dataset = "skype_forum"
data_dir = "C:\\Users\\jichuanzeng\\Dropbox\\ICSE'19\\src\\skype"
out_fn = dataset + ".json"

fn_lst = os.listdir(data_dir)
fn_lst = [os.path.join(data_dir, fn) for fn in fn_lst if "json" in fn]

data_train_dic = {}
data_eval_dic = {}
data_train_dic["dataset"] = dataset + ".train"
data_eval_dic["dataset"] = dataset + ".eval"
data_lst = []
for fn in fn_lst:
    with open(fn) as fin:
        cat = os.path.basename(fn)[:-5]
        lines = fin.readlines()
        if len(lines) < 100:
            print("too small samples, ignoring %s" % cat)
            continue
        for ind, line in enumerate(lines):
            iterms = line.strip().split("******")
            if len(iterms) > 4:
                data_lst.append({"title": iterms[1], "text": " ".join(iterms[2:-1]), "category": cat, "meta": iterms[-1]})
            elif len(iterms) == 4:
                data_lst.append({"title": iterms[1], "text": iterms[2], "category": cat, "meta": iterms[-1]})
            else:
                print("format corrupted in %s line %d..." % (cat, ind))

indices = np.arange(len(data_lst))
np.random.shuffle(indices)
nb_test_samples = int(0.2 * len(data_lst))
data_lst = np.array(data_lst)[indices]
data_train_dic["data"] = list(data_lst[:-nb_test_samples])
data_eval_dic["data"] = list(data_lst[-nb_test_samples:])

json.dump(data_train_dic, open(os.path.join(data_dir, dataset + ".train.json"), 'w'))
json.dump(data_eval_dic, open(os.path.join(data_dir, dataset + ".eval.json"), 'w'))

print("finished")
