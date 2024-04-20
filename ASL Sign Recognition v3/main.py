import scipy.io.matlab as mat
from SignInfo import SignInfo
from Sign import Sign
from Dictionary import Dictionary
import time
from tslearn.metrics import dtw as ts_dtw


# (data, keys) = load_annotations(filename)
def load_annotations(filename):
    data_raw = mat.loadmat(filename)
    data = data_raw["data"][0, 0]
    keys = data.dtype.fields.keys()

    return data, list(keys)


# data = load_handface(filename)
def load_handface(filename):
    data_raw = mat.loadmat(filename)
    data = data_raw["handface"]

    return data


directory = "sga2010/"
ann_filename1 = directory + "annotation_gb1113.mat"
hf_filename1 = directory + "handface_manual_gb1113.mat"

ann_filename2 = directory + "annotation_lb1113.mat"
hf_filename2 = directory + "handface_manual_lb1113.mat"

ann_filename3 = directory + "annotation_tb1113.mat"
hf_filename3 = directory + "handface_manual_tb1113.mat"


gb_ann, _ = load_annotations(ann_filename1)
gb_hf = load_handface(hf_filename1)

lb_ann, _ = load_annotations(ann_filename2)
lb_hf = load_handface(hf_filename2)

tb_ann, _ = load_annotations(ann_filename3)
tb_hf = load_handface(hf_filename3)

vocab: list[Sign] = []
start = time.time()
for i in range(1113):
    gloss = gb_ann["lexicon"][i, 0][0]
    # the number of hands in the sign (either 1 or 2)
    # signs are only compared to those with the same handedness
    num_hands = 1 if (gb_ann["type"][i, 0] == 1) else 2

    sign = Sign(gloss, num_hands, [SignInfo(i, num_hands, tb_hf), SignInfo(i, num_hands, lb_hf)])
    vocab.append(sign)

dictionary = Dictionary(vocab)

print(dictionary.vocab[0].examples[0].all_features)
print(f"preprocessing finished in {time.time() - start} seconds")

num_correct = 0
for i in range(1113):
    gloss = lb_ann["lexicon"][i, 0][0]
    num_hands = 1 if (lb_ann["type"][i, 0] == 1) else 2

    start = time.time()
    print(gloss)
    predictions = dictionary.sign_to_word(10, num_hands, SignInfo(i, num_hands, gb_hf))
    print(predictions)
    print(f"calculated in {time.time() - start} seconds")

    if gloss in predictions:
        num_correct += 1

print(num_correct / 1113)
