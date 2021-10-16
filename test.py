import wave
import numpy as np
import math
import matplotlib.pyplot as plt

FRAME_SIZE = 882
STEP = 441
SAMPLING_FREQUENCY = 44100
ZERO_CROSSING_RATE = 2500000000
SEGMENT_LENGTH = 882


def plot_to_file(*args):
    outputfilename = args[-1]
    signals = args[0:-1]
    plt.figure(1)
    plt.title(f"Signal Wave {outputfilename}")
    [plt.plot(signal) for signal in signals]
    plt.savefig(f'./plot-test/{outputfilename}.jpg')
    plt.cla()


def detect_endpoint(signal):
    length = len(signal)
    print('Detecting endpoint')
    print('# of samples:', length)
    itr = [idx * STEP for idx in range(math.floor(length / STEP))]
    counter = 0
    n_start = 0
    n_end = 0

    for idx in itr:
        start = idx
        end = idx + FRAME_SIZE
        if end > length:
            break
        frame = signal[start:end]
        energy = sum([math.pow(sample, 2) for sample in frame])
        if n_start == 0 and energy > ZERO_CROSSING_RATE:
            counter += 1
            if counter >= 3:
                counter = 0
                n_start = start
        if n_start != 0 and energy < ZERO_CROSSING_RATE:
            counter += 1
            if counter >= 3:
                counter = 0
                n_end = end
                break

    print(f'Found end points:\nstart: {n_start}\nend: {n_end}')

    return [n_start, n_end]


def extract_fragment(signal):
    length = len(signal)
    start = math.floor(length/2 - SEGMENT_LENGTH / 2)
    end = start + SEGMENT_LENGTH
    return signal[start:end]


def dft(signal):
    n = len(signal)
    X_real = []
    X_img = []
    for m in range(math.floor(n/2)):
        tmp_real = 0
        tmp_img = 0

        for k in range(n):
            theta = 2 * math.pi * k * m / n
            tmp_real += signal[k] * math.cos(theta)
            tmp_img -= signal[k] * math.sin(theta)

        X_real.append(tmp_real)
        X_img.append(tmp_img)

    return np.sqrt(np.square(X_real) + np.square(X_img))


def pre_emphasis(signal_in, pre_emphasis_constant=0.945):
    signal_out = np.append(
        signal_in[0], signal_in[1:] - pre_emphasis_constant * signal_in[:-1])
    return signal_out


def auto_correlation(signal):
    window = len(signal)
    auto_coeff = np
    for i in range(10):
        auto_coeff[i] = 0.0
        for j in range(i, window):
            auto_coeff[i] += signal[j] * signal[j-i]

    return auto_coeff


def main():
    # filenames = ['s6A']
    filenames = ['s1A', 's2A', 's3A', 's4A', 's5A', 's6A']

    for filename in filenames:
        with wave.open(f'44khz-recordings/{filename}.wav', "r") as spf:
            # Extract Raw Audio from Wav File
            signal = spf.readframes(-1)
            framerate = spf.getframerate()
            nframes = spf.getnframes()
            print('-----------START-----------')
            print('Processing', filename)
            print('framerate:', framerate)
            print('# of frames:', nframes)

            # convert signal to np array
            signal = np.frombuffer(signal, dtype='int16')

            # plot the signal
            plot_to_file(signal, filename)

            # find endpoint for the signal
            n_start, n_end = detect_endpoint(signal)

            # plot the trimmed signal
            plot_to_file(signal[n_start:n_end], f'trimed_{filename}')

            # extract and plot a 20ms fragment from the signal
            segment = extract_fragment(signal[n_start:n_end])
            plot_to_file(segment, f'segment_{filename}')

            # find and plot the dft
            extracted_dft = dft(segment)
            plot_to_file(extracted_dft, f'fourier_{filename}')

            # pre-emphasis and plot the signals
            pem_segment = pre_emphasis(segment)
            plot_to_file(segment, pem_segment, f'pem_{filename}')

            cuto_coeff = auto_correlation(segment)
            print(f'{cuto_coeff}')
            print('-----------END-----------')


if __name__ == '__main__':
    main()
