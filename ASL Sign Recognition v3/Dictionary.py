from SignInfo import SignInfo
from Sign import Sign


class Dictionary:
    def __init__(self, *sessions):
        # this is an array of all the signs that the dictionary supports translation of
        self.vocab: list[Sign] = []
        for i in range(1113):
            # glosses are all the same order in sessions, organized alphabetically
            gloss = sessions[0].annotations['lexicon'][i, 0][0]
            # hand nums are all the same for each session
            num_hands = 1 if (sessions[0].annotations["type"][i, 0] == 1) else 2

            examples = []
            for session in sessions:
                examples.append(SignInfo(i, num_hands, session.handface))

            self.vocab.append(Sign(gloss, num_hands, examples))

    def __len__(self):
        """
        Returns the number of signs
        """
        return len(self.vocab)

    def __getitem__(self, key):
        """
        indexing a dictionary with an integer returns the ith word

        accessing with a string returns the sign for the gloss
        this is inefficient as it must search through the whole vocab list
        """
        if isinstance(key, int):
            return self.vocab[key]
        elif isinstance(key, str):
            for sign in self.vocab:
                if sign.gloss == key:
                    return sign

    def word_to_sign(self, word: str):
        """
        opens video files of ASL sign equivalents for word
        """
        raise NotImplementedError

    def sign_to_word(self, top_n, num_hands, unknown: SignInfo) -> tuple[list[str], list[str]]:
        """
        :param top_n: the top n signs to return
        :param num_hands: the number of hands in the unknown sign, used to minimize the search space
        :param unknown: SignInfo object representing the unknown sign
        :return: (1) the top n signs and (2) the full ranking of all signs in the dictionary
        """
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

        # full predictions of all 1113 signs
        full_predictions = list(predictions)
        # filter out top n predictions
        top_n_predictions = full_predictions[:top_n]

        return top_n_predictions, full_predictions
