from wave_factory.model.speakerDiarization.ghostvlad import model as spk_model
from wave_factory.model.speakerDiarization import uisrnn
import numpy as np
import librosa
import os
import json

SAVED_MODEL_NAME = r'wave_factory/model/speakerDiarization/pretrained/saved_model.uisrnn_benchmark'
resume = r'wave_factory/model/speakerDiarization/ghostvlad/pretrained/weights.h5'


class Predictor:
    def __init__(self):
        self.network_eval, self.uisrnnModel, self.inference_args = self.load_model()

    # load the model with parameters
    def load_model(self):
        params = {'dim': (257, None, 1),
                  'nfft': 512,
                  'spec_len': 250,
                  'win_length': 400,
                  'hop_length': 160,
                  'n_classes': 5994,
                  'sampling_rate': 16000,
                  'normalize': True,
                  }

        network_eval = spk_model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                        num_class=params['n_classes'],
                                                        mode='eval')
        # mode = 'eval', args = args)
        network_eval.load_weights(resume, by_name=True)
        # !!!!!!!!!!!!!!!!!! save the problem !!!!!!!!!!!!!!!!!!!!!
        network_eval._make_predict_function()

        # load uisrnn model parameters
        model_args, inference_args = uisrnn.parse_arguments()
        model_args.observation_dim = 512
        uisrnnModel = uisrnn.UISRNN(model_args)
        uisrnnModel.load(SAVED_MODEL_NAME)
        return network_eval, uisrnnModel, inference_args

    def predict(self, wav_path):
        network_eval = self.network_eval
        uisrnnModel = self.uisrnnModel
        inference_args = self.inference_args

        embedding_per_second = 1.2
        overlap_rate = 0.4

        specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
        mapTable, keys = genMap(intervals)

        feats = []
        for spec in specs:
            spec = np.expand_dims(np.expand_dims(spec, 0), -1)
            v = network_eval.predict(spec)
            feats += [v]

        feats = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]
        predicted_label = uisrnnModel.predict(feats, inference_args)

        time_spec_rate = 1000 * (1.0 / embedding_per_second) * (1.0 - overlap_rate)  # speaker embedding every ?ms
        center_duration = int(1000 * (1.0 / embedding_per_second) // 2)
        speakerSlice = arrangeResult(predicted_label, time_spec_rate)

        for spk, timeDicts in speakerSlice.items():  # time map to orgin wav(contains mute)
            for tid, timeDict in enumerate(timeDicts):
                s = 0
                e = 0
                for i, key in enumerate(keys):
                    if (s != 0 and e != 0):
                        break
                    if (s == 0 and key > timeDict['start']):
                        offset = timeDict['start'] - keys[i - 1]
                        s = mapTable[keys[i - 1]] + offset
                    if (e == 0 and key > timeDict['stop']):
                        offset = timeDict['stop'] - keys[i - 1]
                        e = mapTable[keys[i - 1]] + offset

                speakerSlice[spk][tid]['start'] = s
                speakerSlice[spk][tid]['stop'] = e

        result = dict()
        for spk, timeDicts in speakerSlice.items():
            result[str(spk)] = ''
            for timeDict in timeDicts:
                s = timeDict['start']
                e = timeDict['stop']
                s = fmtTime(s)
                e = fmtTime(e)
                result[str(spk)] += s + '==>' + e + ';'
            result[str(spk)] = result[str(spk)][:-1]

        result_json = json.dumps(result, sort_keys=True)
        return result_json


##############################################################################
##############################################################################

# load and process wave file with overlap windows
# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5,
              overlap_rate=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while True:  # slide window.
        if cur_slide + spec_len > time:
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


# load wave file
def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals / sr * 1000).astype(int)


# create Short-time Fourier transform of wave file
def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    return linear.T


# interval slices to maptable
def genMap(intervals):
    slicelen = [sliced[1] - sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1, -1]

    keys = [k for k, _ in mapTable.items()]
    keys.sort()
    return mapTable, keys


# {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
def arrangeResult(labels, time_spec_rate):
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i, label in enumerate(labels):
        if (label == lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * (len(labels)))})
    return speakerSlice


def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0] + 0.5)
    timeDict['stop'] = int(value[1] + 0.5)
    if (key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice


# Convert to standard time format
# minute:second.millisecond
def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond % 1000
    minute = timeInMillisecond // 1000 // 60
    second = (timeInMillisecond - minute * 60 * 1000) // 1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time

# predictor = Predictor()
# predictor.load_model()
# predictor.predict(r'speakerDiarization/output_k2wwwDv.wav')
