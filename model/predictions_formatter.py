import sys
import json

raw = sys.argv[1]
idx = sys.argv[2] if len(sys.argv) >= 2 else ''
res = open("formatted_predictions_%s.json" % idx, "w")
with open(raw) as f:
    dic = eval(f.read())
    ids = set()
    for key in dic:
        id_ = key.rsplit("#", 1)[0]
        if id_ not in ids:
            ids.add(id_)
    # print(len(ids))
    for id_ in ids:
        best_span_str = []
        qids = []
        num = 0
        qid = id_ + "#" + str(num)
        while qid in dic:
            qids.append(qid)
            best_span_str.append(dic[qid])
            num += 1
            qid = id_ + "#" + str(num)
        res.write(json.dumps({"best_span_str": best_span_str, "qid": qids, "yesno": ["y"] * len(qids), "followup": ["y"] * len(qids)}))
        res.write('\n')
res.close()