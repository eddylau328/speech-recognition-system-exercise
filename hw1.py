from hw1_3 import answer_q3
from hw1_4 import answer_q4
import os


def init():
    if (not os.path.exists('./plot')):
        os.makedirs('./plot')


if (__name__ == '__main__'):
    # student id: 1155093989
    recordings = ['s1A', 's0A', 's3A', 's5A', 's8A', 's9A']
    path = 'recordings'
    # Create a new directory because it does not exist
    init()
    # answer_q3(path, recordings)
    answer_q4(path, recordings)
