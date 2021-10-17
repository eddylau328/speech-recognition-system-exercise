import matplotlib.pyplot as plt
import numpy as np
import wave
from librosa.feature import mfcc


def get_signal(path, filename):
    spf = wave.open(
        "{path}/{filename}.wav".format(path=path, filename=filename), "r")
                # Extract Raw Audio from Wav File
    raw_signal = spf.readframes(-1)
    frame_rate = spf.getframerate()
    signal = np.frombuffer(raw_signal, "int16").astype('float')
    return signal, frame_rate
 

def find_mfcc_paramter(signal, frame_rate):
    return mfcc(y=signal, sr=frame_rate, n_mfcc=13)


def get_distortion_matrix(mfcc_parameter_A, mfcc_parameter_B):
    size_x, size_y = mfcc_parameter_A.shape[1], mfcc_parameter_B.shape[1]
    distortion_matrix = []
    for t_y in range(size_y):
        row = []
        for t_x in range(size_x):
            acc =  0
            for j in range(1, 13):
                mx = mfcc_parameter_A[j][t_x]
                my = mfcc_parameter_B[j][t_y]
                acc += (mx - my)**2
            acc = np.sqrt(acc)
            row.append(acc)
        distortion_matrix.append(row)
    distortion_matrix = np.array(distortion_matrix, 'float')
    return distortion_matrix


def get_accumulation_matrix(distortion_matrix, mfcc_parameter_A, mfcc_parameter_B):
    size_x, size_y = mfcc_parameter_A.shape[1], mfcc_parameter_B.shape[1]
    accumulation_matrix = []
    for t_y in range(size_y):
        row = []
        accumulation_matrix.append([])
        for t_x in range(size_x):
            compare = []
            current = distortion_matrix[t_y][t_x]
            if (t_y > 0 and t_x > 0):
                compare.append(accumulation_matrix[t_y-1][t_x-1])
            if (t_y > 0):
                compare.append(accumulation_matrix[t_y-1][t_x])
            if (t_x > 0):
                compare.append(accumulation_matrix[t_y][t_x-1])
            if (len(compare) > 0):
                minimum = min(compare)
            else:
                minimum = 0
            accumulation_matrix[t_y].append(current + minimum)
    accumulation_matrix = np.array(accumulation_matrix, 'float')
    return accumulation_matrix


def find_min_accumulated_score(mfcc_parameter_A, mfcc_parameter_B):
    distortion_matrix = get_distortion_matrix(mfcc_parameter_A, mfcc_parameter_B)
    accumulation_matrix = get_accumulation_matrix(distortion_matrix, mfcc_parameter_A, mfcc_parameter_B)
    last_column = accumulation_matrix[:, -1]
    last_row = accumulation_matrix[-1, :]
    minimum_accumulated_score = np.min(np.concatenate((last_column, last_row)))
    return minimum_accumulated_score


def find_optimal_path(path, record_pairs):
    record_pair = record_pairs[0]
    record_A, record_B = record_pair[0], record_pair[1]
    signal_A, frame_rate_A = get_signal(path, record_A)
    signal_B, frame_rate_B = get_signal(path, record_B)
    mfcc_paramter_A = find_mfcc_paramter(signal_A, frame_rate_A)
    mfcc_paramter_B = find_mfcc_paramter(signal_B, frame_rate_B)
    distortion_matrix = get_distortion_matrix(mfcc_paramter_A, mfcc_paramter_B)
    accumulation_matrix = get_accumulation_matrix(distortion_matrix, mfcc_paramter_A, mfcc_paramter_B)
    accumulation_matrix = np.flip(accumulation_matrix, axis=0)

    # find starting point
    start_x, start_y = 0, 0
    size_y, size_x = accumulation_matrix.shape[0], accumulation_matrix.shape[1]
    minimum_accumulated_score = accumulation_matrix[0][0]
    for i in range(size_x):
        if (minimum_accumulated_score > accumulation_matrix[0][i]):
            start_x, start_y = i, 0
            minimum_accumulated_score = accumulation_matrix[0][i]
    for i in range(size_y):
        if (minimum_accumulated_score > accumulation_matrix[i][size_x-1]):
            start_x, start_y = 0, i
            minimum_accumulated_score = accumulation_matrix[i][size_x-1]

    paths = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    while (current_x > 0 and current_y < size_y - 1):
        compare = []
        compare.append(accumulation_matrix[current_y+1][current_x-1])
        compare.append(accumulation_matrix[current_y+1][current_x])
        compare.append(accumulation_matrix[current_y][current_x-1])
        min_value = np.min(compare)
        if (min_value == accumulation_matrix[current_y+1][current_x-1]):
            current_x, current_y = current_x-1, current_y+1
            paths.append((current_x, current_y))
        elif (min_value == accumulation_matrix[current_y+1][current_x]):
            current_x, current_y = current_x, current_y+1
            paths.append((current_x, current_y))
        else:
            current_x, current_y = current_x-1, current_y
            paths.append((current_x, current_y))

    plt.matshow(accumulation_matrix)
    plt.title('Accumulation Matrix for {} to {}'.format(record_A, record_B))
    for point in paths:
        rect = plt.Rectangle((point[0]-.5, point[1]-.5), 1,1, fill=False, edgecolor='red')
        ax = plt.gca()
        ax.add_patch(rect)
    plt.colorbar()
    plt.savefig('plot/{}-to-{}-accumulation-matrix.jpg'.format(record_A, record_B))
    plt.show()
    print('Save plot/{}-to-{}-accumulation-matrix.jpg'.format(record_A, record_B))


def find_confusion_matrix(path, recordings_A, recordings_B):
    confusion_matrix = []
    for record_B in recordings_B:
        row = []
        for record_A in recordings_A:
            signal_A, frame_rate_A = get_signal(path, record_A)
            signal_B, frame_rate_B = get_signal(path, record_B)
            mfcc_paramter_A = find_mfcc_paramter(signal_A, frame_rate_A)
            mfcc_paramter_B = find_mfcc_paramter(signal_B, frame_rate_B)
            minimum_accumulated_score = find_min_accumulated_score(mfcc_paramter_A, mfcc_paramter_B)
            row.append(minimum_accumulated_score)
        confusion_matrix.append(row)
    confusion_matrix = np.array(confusion_matrix, 'float')
    plt.matshow(confusion_matrix)
    plt.title('Confusion Matrix')
    plt.xlabel('Actual class (Set A)')
    plt.ylabel('Predicted class (Set B)')

    ax = plt.gca()
    ax.set_xticks(np.arange(len(recordings_A)))
    ax.set_xticklabels(recordings_A)

    ax.set_yticks(np.arange(len(recordings_B)))
    ax.set_yticklabels(recordings_B)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, '{}'.format(int(z)), ha='center', va='center', fontdict={'color': 'red'})

    plt.colorbar()
    plt.savefig('plot/confusion-matrix.jpg')
    plt.show()
    print('Save plot/confusion-matrix.jpg')
    return confusion_matrix


def answer_q5(path, record_pairs, recordings_A, recordings_B):

    # 5a
    for record in recordings_A + recordings_B:
        signal, frame_rate = get_signal(path, record)
        mfcc_paramter = find_mfcc_paramter(signal, frame_rate)
        np.savetxt("plot/mfcc_parameter_{}.csv".format(record), mfcc_paramter, delimiter=',')


    for pair in record_pairs: 
        signal_A, frame_rate_A = get_signal(path, pair[0])
        signal_B, frame_rate_B = get_signal(path, pair[1])
        signal_C, frame_rate_C = get_signal(path, pair[2])

        # 5b
        mfcc_paramter_A = find_mfcc_paramter(signal_A, frame_rate_A)
        mfcc_paramter_B = find_mfcc_paramter(signal_B, frame_rate_B)
        mfcc_paramter_C = find_mfcc_paramter(signal_C, frame_rate_C)

        minimum_accumulated_score = find_min_accumulated_score(mfcc_paramter_A, mfcc_paramter_B)
        print('Minimum Accumulated Score for {} to {} = {}'.format(pair[0], pair[1], minimum_accumulated_score))
        minimum_accumulated_score = find_min_accumulated_score(mfcc_paramter_A, mfcc_paramter_C)
        print('Minimum Accumulated Score for {} to {} = {} (for comparison)'.format(pair[0], pair[2], minimum_accumulated_score))

    # 5c
    confusion_matrix = find_confusion_matrix(path, recordings_A, recordings_B)
    np.savetxt("plot/confusion-matrix.csv", confusion_matrix, delimiter=',')



    # 5d
    find_optimal_path(path, record_pairs)


if (__name__ == '__main__'):
    record_pairs = [('s1A', 's1B', 's8B')]
    recordings_A = ['s1A', 's0A', 's3A', 's5A', 's8A', 's9A']
    recordings_B = ['s1B', 's0B', 's3B', 's5B', 's8B', 's9B']
    answer_q5('recordings', record_pairs, recordings_A, recordings_B)
