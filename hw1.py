from hw1_3 import answer_q3
from hw1_4 import answer_q4
import os

from hw1_5 import answer_q5


def init():
    if (not os.path.exists('./plot')):
        os.makedirs('./plot')
    if (not os.path.exists('./visualize')):
        os.makedirs('./visualize')


if (__name__ == '__main__'):
    # student id: 1155093989
    path = 'recordings'
    record_pairs = [('s1A', 's1B', 's8B')]
    recordings_A = ['s1A', 's0A', 's3A', 's5A', 's8A', 's9A']
    recordings_B = ['s1B', 's0B', 's3B', 's5B', 's8B', 's9B']
    # Create a new directory because it does not exist
    init()
    print('----------------Q3-----------------')
    answer_q3(path, recordings_A)
    print('----------------Q4-----------------')
    answer_q4(path, recordings_A)
    print('----------------Q5-----------------')
    answer_q5(path, record_pairs, recordings_A, recordings_B)
    print('-----------------------------------')
