import numpy as np
import wave


# def mfcc_paramter(signal, frame_rate):
# print(mfcc(signal, frame_rate))


def answer_q5(path, filenames):
    for filename in filenames:
        spf = wave.open(
            "{path}/{filename}.wav".format(path=path, filename=filename), "r")
        # Extract Raw Audio from Wav File
        raw_signal = spf.readframes(-1)
        frame_rate = spf.getframerate()
        total_frame = spf.getnframes()
        signal = np.frombuffer(signal, "int16")
        # mfcc_paramter(signal, frame_rate)


if (__name__ == '__main__'):
    answer_q5('recordings', ['s1A'])
