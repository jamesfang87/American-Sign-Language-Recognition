from classification import classify
from createexamples import create_examples


video_path = "ASL_2008_01_11"
examples = create_examples(video_path)

video_path = "ASL_2008_08_04"
examples2 = create_examples(video_path)

all_examples = examples + examples2

video_path = "ASL_2008_05_12a"
unknown = create_examples(video_path)
classify(unknown, all_examples)  # put in an array (or any iterable) of data
