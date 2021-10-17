import matplotlib.pyplot as plt
import numpy as np
import wave


def plot_signal(path, filename):
    spf = wave.open(
        "{path}/{filename}.wav".format(path=path, filename=filename), "r")
    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, "int16")

    plt.figure(1)
    plt.title("{filename} Signal Wave".format(filename=filename))
    plt.plot(signal)
    plt.savefig('plot/{filename}-signal.jpg'.format(filename=filename))
    plt.show()


def answer_q3(path, filenames):
    for filename in filenames:
        plot_signal(path, filename)
        print('Save plot/{filename}-signal.jpg'.format(filename=filename))


if (__name__ == '__main__'):
    answer_q3('recordings', ['s1A'])
