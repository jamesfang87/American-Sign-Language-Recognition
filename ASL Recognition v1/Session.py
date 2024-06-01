import scipy.io.matlab as mat


# data = load_annotations(filename)
def load_annotations(filename):
    data_raw = mat.loadmat(filename)
    data = data_raw["data"][0, 0]
    keys = data.dtype.fields.keys()

    return data


# data = load_handface(filename)
def load_handface(filename):
    data_raw = mat.loadmat(filename)
    data = data_raw["handface"]

    return data


class Session:
    def __init__(self, annotation_filename, handface_filename):
        self.annotations = load_annotations(annotation_filename)
        self.handface = load_handface(handface_filename)

