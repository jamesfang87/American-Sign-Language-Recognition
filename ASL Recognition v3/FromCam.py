import pandas as pd
from VidReader import VideoReader
from SignInfo import SignInfo
from Dictionary import Dictionary
from Session import Session


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
dictionary = Dictionary(s1, s2, s3)

naomi = pd.read_csv("annotations/naomi_firsthalf")
num_correct = 0
prev_path = "no path"
t2 = open("test4", "a+")
for index, sign in list(naomi.iterrows())[600:655]:
    if sign["path"] != prev_path:
        vid_reader = VideoReader(1, sign["path"])

    info = vid_reader.read(sign["start frame"] - 50, sign["end frame"] + 50, show=True)
    temp = SignInfo(1, sign["handedness"], raw_vid_info=info)
    top_n, _ = dictionary.sign_to_word(10, sign["handedness"], temp, True)
    print(sign["gloss"])
    print(top_n)

    if sign["gloss"] in top_n:
        num_correct += 1
        t2.write('1\n')
    else:
        t2.write('0\n')

print(f"accuracy: {num_correct / 100}")