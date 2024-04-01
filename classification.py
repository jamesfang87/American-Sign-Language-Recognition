from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math
from sign import Sign
import pandas as pd


def top_n_accuracy(n: int, actual_gloss: str, pred: list[dict]) -> bool:
    """
    Calculates the top n accuracy for one unknown sign
    :param n: the top n predictions to search for the correct
    :param actual_gloss: the actual glosses.csv of the sign
    :param pred: the predicted glosses.csv of the sign: a list of dictionaries with keys 'score' and 'prediction
    :return: true if the sign is in the top n predictions, false otherwise
    """
    pred.sort(key=lambda x: x['score'])
    pred = list(dict.fromkeys([pred[i]['prediction'] for i in range(len(pred))]))[:n]
    print(pred)

    return actual_gloss in pred


signs: pd.DataFrame = pd.read_csv(f'../ASL_2008_01_11/list/ASL_2008_01_11_annotate.csv', nrows=60)
actual: list[str] = list(signs['Sign gloss labels from the Gallaudet Dictionary'])


def classify(testing_data: list[Sign], examples: list[Sign]):
    #cost_writer = open('/Users/jamesfang/PycharmProjects/Hand Gesture Recognition/costs/cost.txt', 'w')
    cost_reader = open('/Users/jamesfang/PycharmProjects/Hand Gesture Recognition/costs/cost.txt', 'r')

    global actual
    num_correct = 0
    for a, unknown_sign in zip(actual, testing_data):
        predictions = []
        for i, example in enumerate(examples):

            temp = list(map(float, cost_reader.readline().strip('\n').split(',')[:4]))
            total_cost = (
                    5.0 * temp[0] +
                    0.17 * temp[1] +
                    1.1 * temp[2] +
                    0.19 * temp[3]
            )
            """

            left_hand_cost, _ = fastdtw(unknown_sign.left_hand, example.left_hand, dist=euclidean)
            right_hand_cost, _ = fastdtw(unknown_sign.right_hand, example.right_hand, dist=euclidean)
            hand_pos_cost = 0.5 * left_hand_cost + 0.5 * right_hand_cost

            right_hand_motion_cost, _ = fastdtw(unknown_sign.right_hand_motion, example.right_hand_motion,
                                                dist=euclidean)
            left_hand_motion_cost, _ = fastdtw(unknown_sign.left_hand_motion, example.left_hand_motion,
                                               dist=euclidean)
            hand_motion_cost = 0.5 * left_hand_motion_cost + 0.5 * right_hand_motion_cost

            hand_dist_cost, _ = fastdtw(unknown_sign.hand_dist, example.hand_dist, dist=euclidean)
            hand_dist_change_cost, _ = fastdtw(unknown_sign.hand_dist_change, example.hand_dist_change,
                                               dist=euclidean)

            cost_writer.write(f'{hand_pos_cost},'
                              f'{hand_motion_cost},'
                              f'{hand_dist_cost},'
                              f'{hand_dist_change_cost},'
                              f'{a == example.gloss}\n')
            
            total_cost = (
                    1.78 * hand_pos_cost +
                    0.17 * hand_motion_cost +
                    1.11 * hand_dist_cost +
                    0.19 * hand_dist_change_cost
            )
            """

            predictions.append({
                'prediction': example.gloss,
                'score': total_cost
            })

        num_correct += top_n_accuracy(10, a, predictions)

    return num_correct / len(testing_data)
