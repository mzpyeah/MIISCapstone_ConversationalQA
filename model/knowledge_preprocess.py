import commonsense_utils.general as CG
from commonsense_utils.general import sample_relations
from tqdm import tqdm
import json
import nltk
import sys

concept_net = "ConceptNet_data/cn_relations_orig.txt"
quac_train = "QuAC_data/train.json"
quac_dev = "QuAC_data/dev.json"
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def read_data(quac):
    with open(quac, "r", encoding='utf-8') as reader:
        source = json.load(reader)
        return source

def create_processed_dataset(quac, with_history=2):
    relations = CG.get_relations(concept_net)
    orig_data = read_data(quac)
    data = orig_data['data']
    print("Building commonsense...")
    for i in tqdm(range(len(data))):
        data_pt = data[i]['paragraphs'][0]
        qas = data_pt['qas']
        context = nltk.word_tokenize(data_pt['context'].lower())
        qs = []
        for idx, qa in enumerate(qas):
            q = nltk.word_tokenize(qa['question'].lower())
            temp_context = context + [token for token in q for q in qs[::-1]]
            qs.append(q)
            if len(qs) > with_history:
                qs = qs[-with_history:]
                assert len(qs) == with_history
            subgraph = CG.build_trees(relations, q, stop_words, temp_context)
            selected_relations = sample_relations(subgraph, temp_context)
            orig_data['data'][i]['paragraphs'][0]['qas'][idx]['commonsense'] = selected_relations
    print("Done!")
    return orig_data

if __name__ == "__main__":
    nltk.download('punkt')
    with_history = int(sys.argv[1])
    train = create_processed_dataset(quac_train, with_history)
    with open("QuAC_data/train_cs_his%d.json" % with_history, "w") as f:
        json.dump(train, f)
    dev = create_processed_dataset(quac_dev, with_history)
    with open("QuAC_data/dev_cs_his%d.json" % with_history, "w") as f:
        json.dump(dev, f)