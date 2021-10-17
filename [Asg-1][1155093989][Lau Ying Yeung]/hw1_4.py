from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import wave


def sgn(value):
    return -1 if value < 0 else 1


def split_frame(signal, size, step):
    signal_list = signal.tolist()
    result = [signal_list[i: i + size]
              for i in range(0, len(signal_list) - step, step)]
    result[-1] = result[-1] + [0 for i in range(0, size - len(result[-1]))]
    return np.array(result, dtype='float')


def get_zero_crossing_rate(signal, size, step):
    # zero_crossings = np.nonzero(np.diff(signal > 0))[0]
    acc, index, start, end = 0, 1, 0, size
    results = []
    while (index <= len(signal)):
        if (start <= index <= end and index < len(signal)):
            acc += np.absolute(sgn(signal[index]) - sgn(signal[index - 1]))
        else:
            results.append(acc / size)
            acc = 0
            start += step
            end += step
            if (index < len(signal)):
                index = index - step
            else:
                break
        index += 1
    return np.array(results, 'float')


def normalized(signal):
    # root_mean_square = np.sqrt(np.mean(signal**2))
    # return np.array(signal, dtype='float') / root_mean_square
    return np.array(signal, dtype='float') / 32767.0


def detect_endpoint(filename, raw_signal, frame_rate):
    # convert signal to numpy array
    signal = np.frombuffer(raw_signal, "int16")
    # plt.figure(1)
    # plt.title("{filename} Signal Wave".format(filename=filename))
    # plt.plot(signal)
    # plt.show()
    normalized_signal = normalized(signal)
    plt.subplot(3, 1, 1)
    plt.title("{filename} Signal".format(filename=filename))
    plt.plot(
        np.array([i/frame_rate for i in range(len(signal))]), normalized_signal)

    size = np.floor((20 / 1000) / (1 / frame_rate)).astype('int')
    step = np.floor((10 / 1000) / (1 / frame_rate)).astype('int')

    split_signal_frame = split_frame(normalized_signal, size, step)
    square_split_signal_frame = np.square(split_signal_frame)
    energy_level = np.sum(square_split_signal_frame, axis=1)
    plt.subplot(3, 1, 2)
    plt.title("{filename} Energy Level".format(filename=filename))
    plt.plot(energy_level)

    zero_crossing_rate = get_zero_crossing_rate(
        normalized_signal, size, step)
    plt.subplot(3, 1, 3)
    plt.title("{filename} Zero Crossing Rate".format(filename=filename))
    plt.plot(zero_crossing_rate)
    plt.tight_layout()

    energy_thershold = 1.5
    zero_crossing_rate_thershold = (0.02, 0.1)
    start_successive_frame, end_successive_frame = 5, 8

    start, end = -1, -1
    for i in range(len(energy_level)):
        if (start == -1):
            count = 0
            for j in range(i, len(energy_level)):
                is_energy_exceed = energy_level[i] >= energy_thershold
                is_within_zero_crossing_rate_thershold = zero_crossing_rate_thershold[
                    0] <= zero_crossing_rate[j] <= zero_crossing_rate_thershold[1]
                if (is_energy_exceed and is_within_zero_crossing_rate_thershold):
                    count += 1
                elif (count >= start_successive_frame):
                    break
                else:
                    break
            if (count >= start_successive_frame):
                start = i
            else:
                count = 0
        elif (end == -1):
            count = 0
            for j in range(i, len(energy_level)):
                is_energy_below = energy_level[i] < energy_thershold
                is_within_zero_crossing_rate_thershold = zero_crossing_rate_thershold[
                    0] <= zero_crossing_rate[j] <= zero_crossing_rate_thershold[1]
                if (is_energy_below and is_within_zero_crossing_rate_thershold):
                    count += 1
                elif (count >= end_successive_frame):
                    break
                else:
                    break
            if (count >= end_successive_frame):
                end = i
            else:
                count = 0

    if (start != -1 and end != -1):
        start_time = start * step / frame_rate
        end_time = end * step / frame_rate
        print('End Point: start_frame = {} , end_frame = {}'.format(start, end))
        print('End Point: start_time = {}, end_time = {}'.format(
            np.round(start_time, 2), np.round(end_time, 2)))
        ax = plt.subplot(3, 1, 1)
        ax.add_patch(Rectangle((start_time, -500), end_time -
                     start_time, 1000, edgecolor='red', facecolor='white'))
        ax = plt.subplot(3, 1, 2)
        ax.add_patch(Rectangle((start, -500), end -
                     start, 1000, edgecolor='red', facecolor='white'))
        ax = plt.subplot(3, 1, 3)
        ax.add_patch(Rectangle((start, -500), end - start,
                     1000, edgecolor='red', facecolor='white'))
    else:
        print('End Point cannot find')
    plt.savefig('visualize/{}-end-point-detection.jpg'.format(filename))
    plt.show()
    return signal, normalized_signal, start_time, end_time


def extract_segment(signal, start_time, end_time, frame_rate):
    start = int(start_time * frame_rate)
    end = int(end_time * frame_rate)
    result = np.array([signal[x] for x in range(start, end)], 'float')
    return np.array([result[x] for x in range(0, int(0.02 * frame_rate))], 'float')


def plot_fourier_transform(filename, signal):
    N = len(signal)
    X_real = []
    X_img = []
    for m in range(int(N / 2)):
        tmp_real, tmp_img = 0, 0
        for k in range(N):
            theta = 2 * np.pi * k * m / N
            tmp_real += signal[k] * np.cos(theta)
            tmp_img -= signal[k] * np.sin(theta)
        X_real.append(tmp_real)
        X_img.append(tmp_img)
    result = np.sqrt(np.square(X_real) + np.square(X_img))
    plt.title("{filename} Fourier Transform".format(filename=filename))
    plt.plot(result)
    plt.savefig('plot/fourier_{}.jpg'.format(filename))
    plt.show()


def pre_emphasis(filename, signal, pre_emphasis_constant):
    result = [signal[0]]
    for k in range(1, len(signal)):
        result.append(signal[k] - pre_emphasis_constant * signal[k-1])
    pre_emphasis_signal = np.array(result, 'float')
    plt.title(
        "{filename} Pre-emphasis Signal & Signal Segment".format(filename=filename))
    plt.plot(signal)
    plt.plot(pre_emphasis_signal)
    plt.savefig('plot/pre_emphasis_{}.jpg'.format(filename))
    plt.show()
    return pre_emphasis_signal


def auto_correlation(signal, order):
    auto_coeff = []
    for i in range(order + 1):
        auto_coeff.append(0)
        for j in range(i, len(signal)):
            auto_coeff[i] += signal[j] * signal[j-i]
    return np.array(auto_coeff, 'float')


def find_lpc(signal, order):
    auto_coeff = auto_correlation(signal, order)
    matrix = []
    for i in range(order):
        row = []
        for j in range(i, 0, -1):
            row.append(auto_coeff[j])
        for j in range(order - i):
            row.append(auto_coeff[j])
        matrix.append(row)
    matrix = np.array(matrix, 'float')
    lpc_parameter = np.dot(np.linalg.inv(matrix), np.transpose(auto_coeff[1:]))
    print('LPC parameters: {}'.format(lpc_parameter))


def answer_q4(path, filenames=[]):
    for filename in filenames:
        spf = wave.open(
            "{path}/{filename}.wav".format(path=path, filename=filename), "r")

        # Extract Raw Audio from Wav File
        raw_signal = spf.readframes(-1)
        frame_rate = spf.getframerate()
        total_frame = spf.getnframes()
        print('Target: {}.wav'.format(filename))
        print('framerate: {}'.format(frame_rate))
        print('Total frames: {}'.format(total_frame))
        print()

        # 4a
        signal, normalized_signal, start_time, end_time = detect_endpoint(
            filename, raw_signal, frame_rate)

       # 4b
        segment = extract_segment(
            normalized_signal, start_time, end_time, frame_rate)

        # 4c
        plot_fourier_transform(filename, segment)

        # 4d
        pre_emphasis_segment = pre_emphasis(filename, segment, 0.945)

        # 4e
        find_lpc(pre_emphasis_segment, 10)


if (__name__ == '__main__'):
    answer_q4('recordings', ['s1A'])
