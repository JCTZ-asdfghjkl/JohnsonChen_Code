import os, pickle
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_multitaper
import pyedflib
import philistine as ph
import re
from tqdm import tqdm


def extract_features(eeg, sfreq):
    """
    从一段EEG信号中提取常用的频谱特征
    Extract power spectral features from an EEG segment
    """
    psds, freqs = psd_array_multitaper(
        eeg, sfreq=sfreq, fmin=0.5, fmax=30, adaptive=True, normalization='full', verbose=False
    )
    def bp(b):
        i = np.logical_and(freqs >= b[0], freqs <= b[1])
        return np.sum(psds[i]) * (freqs[1] - freqs[0])

    delta = bp((0.5, 4))
    theta = bp((4, 8))
    alpha = bp((8, 12))
    beta  = bp((12, 30))
    total = delta + theta + alpha + beta

    # Alpha峰值/1/f斜率/睁眼判别等
    # Alpha peak/1/f slope/eyes open ratio, etc.
    
    raw = mne.io.RawArray(eeg[np.newaxis, :], info=mne.create_info(['eeg'], sfreq, ch_types='eeg'), verbose=False)
    alpha_peak = ph.mne.savgol_iaf(raw, fmin=7, fmax=13).PeakAlphaFrequency
    if alpha_peak is None:
        alpha_peak = np.nan
    elif alpha_peak==7 or alpha_peak==13:
        alpha_peak = np.nan
    else:
        alpha_peak = float(alpha_peak)
    slope = np.polyfit(np.log(freqs), np.log(psds), 1)[1]
    
    ##TODO: Check blinking: eyes_open_alpha_ratio = alpha / total

    # 高级特征（NaN站位，后续可补全）/ Advanced features as NaN (to be filled in future)
    # TODO
    spindle_count = np.nan
    slow_oscillation_count = np.nan
    cap_index = np.nan

    return {
        #'delta': delta / total,
        'theta': theta / total,
        'alpha': alpha / total,
        'beta': beta / total,
        'alpha_peak': alpha_peak,s
        '1/f_slope': slope,
        #TODO: 'alpha_open_eye_ratio': eyes_open_alpha_ratio,
        'spindles': spindle_count,
        'slow_oscillations': slow_oscillation_count,
        'CAP_index': cap_index
    }

def load_eeg_file(path):
    with open(path, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')
    label = subject['labels']
    eeg = subject['data'][:,24-1]  # shape = (#trial=40, T)
    sfreq = 128.
    return eeg, sfreq


def main():
    # 读取数据集路径
    dataset_path = '../../archive/deap-dataset/data_preprocessed_python'
    
    # 处理每个EEG文件
    features = []
    sids = []
    paths = os.listdir(dataset_path)
    for file in tqdm(paths):
        if file.endswith('.dat'):
            sid = file[:-4]
            eeg, sfreq = load_eeg_file(os.path.join(dataset_path, file))
            breakpoint()
            this_features = []
            for trial_id in range(len(eeg)):
                this_features.append(extract_features(eeg[trial_id], sfreq))
            this_features = pd.DataFrame(this_features)
            this_features.insert(0, 'SID', sid)
            this_features.to_csv(f'eeg_features_{sid}.csv', index=False)
            features.append(this_features)
            sids.append(sid)

    # 保存特征到CSV
    df = np.nanmean(np.array([x.iloc[:,1:].values for x in features]), axis=0)
    df = pd.DatFrame(data=df, columns=features[0].columns[1:]) # 平均特征
    df.to_csv('eeg_features_average.csv', index=False)


if __name__ == '__main__':
    main()
