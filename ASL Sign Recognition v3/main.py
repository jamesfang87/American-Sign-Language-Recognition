import scipy.io.matlab as mat
from SignInfo import SignInfo
from Session import Session
from Dictionary import Dictionary
import time


directory = "sga2010/"
ann_filename1 = directory + "annotation_gb1113.mat"
hf_filename1 = directory + "handface_manual_gb1113.mat"

ann_filename2 = directory + "annotation_lb1113.mat"
hf_filename2 = directory + "handface_manual_lb1113.mat"

ann_filename3 = directory + "annotation_tb1113.mat"
hf_filename3 = directory + "handface_manual_tb1113.mat"

s1 = Session(ann_filename1, hf_filename1)
s2 = Session(ann_filename2, hf_filename2)
s3 = Session(ann_filename3, hf_filename3)

start = time.time()
dictionary = Dictionary(s2, s3)
print(f"preprocessing finished in {time.time() - start} seconds")

num_correct = 0
for i in range(1113):
    gloss = s1.annotations["lexicon"][i, 0][0]
    num_hands = 1 if (s1.annotations["type"][i, 0] == 1) else 2

    start = time.time()
    print(gloss)
    top_n, _ = dictionary.sign_to_word(10, num_hands, SignInfo(i, num_hands, s1.handface))
    print(top_n)
    print(f"calculated in {time.time() - start} seconds")

    if gloss in top_n:
        num_correct += 1


print(f"accuracy: {num_correct / 1113}")
