from classification import classify
from createexamples import create_examples, annotate
import pandas as pd

video_path = "ASL_2008_01_11"
#annotate(video_path, 100)
examples = create_examples(video_path)


video_path = "ASL_2008_08_04"
#annotate(video_path, 100)
examples2 = create_examples(video_path)

all_examples = examples + examples2

video_path = "ASL_2008_05_12a"
#annotate(video_path, 100)
unknown = create_examples(video_path)
print(classify(unknown, all_examples))  # put in an array (or any iterable) of data
