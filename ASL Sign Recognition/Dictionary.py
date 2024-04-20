from SignInfo import SignInfo
from Sign import Sign


class Dictionary:
    def __init__(self, vocab: list[Sign]):
        # this is an array of all the signs that the dictionary supports translation of
        self.vocab: list[Sign] = vocab

    def __len__(self):
        """
        Returns the number of signs
        """
        return len(self.vocab)

    def __getitem__(self, key):
        """
        indexing a dictionary returns the ith word
        """
        return self.vocab[key.gloss]

    def word_to_sign(self, word: str):
        """
        opens video files of ASL sign equivalents for word
        """
        raise NotImplementedError

    def sign_to_word(self, top_n, num_hands, unknown: SignInfo):
        predictions = []
        for sign in self.vocab:
            # compare only signs with the same number of hands
            if sign.num_hands == num_hands:
                # costs found through DTW, lower is better
                score = sign.calc_distance(unknown)
                predictions.append({"sign": sign.gloss, "score": score})

        # sort based on scores
        predictions.sort(key=lambda x: x['score'])
        # save just the gloss of predicted signs
        predictions = dict.fromkeys([prediction['sign'] for prediction in predictions])
        # filter out top n predictions
        predictions = list(predictions)[:top_n]

        return predictions
