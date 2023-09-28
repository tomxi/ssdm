import os

import numpy as np
import librosa


def mfcc(track, recompute=False):
    feature_path = os.path.join(
        track.salami_dir, f'features/{track.tid}_mfcc.npz')
    if recompute or not os.path.exists(feature_path):
        print(f'computing feature: mfcc for track {track.tid}...')
        track.audio()        
        mfcc = librosa.feature.mfcc(
            y=track.y, sr=track.sr, n_mfcc=40, 
            hop_length=4096, n_fft=8192, lifter=0.6)
        normalized_mfcc = (mfcc - np.mean(mfcc, axis=1)[:, None]) / np.std(mfcc, axis=1, ddof=1)[:,None]
        mfcc_ts = librosa.frames_to_time(
            np.arange(normalized_mfcc.shape[-1]), 
            hop_length=4096, sr=track.sr, n_fft=8192)
        np.savez(feature_path, feature=normalized_mfcc, ts=mfcc_ts)

    return np.load(feature_path)


def crema(track, recompute=False):
    crema_feature_path = os.path.join(
        track.salami_dir, f'crema/{track.tid}.npz')
    feature_path = os.path.join(
        track.salami_dir, f'features/{track.tid}_crema.npz')
    if recompute or not os.path.exists(feature_path):
        print(f'resampling computed feature: crema _chord_pitch for track {track.tid}...')
        crema = np.load(crema_feature_path)
        resampled_crema = librosa.resample(
            crema['chord_pitch'].T, orig_sr=track.sr/2048, 
            target_sr=track.sr/4096)
        crema_ts = librosa.frames_to_time(
            np.arange(resampled_crema.shape[-1]), 
            hop_length=4096, sr=track.sr)
        np.savez(feature_path, feature=resampled_crema, ts=crema_ts)

    return np.load(feature_path)


def yamnet(track, recompute=False):
    feature_path = os.path.join(
        track.salami_dir, f'features/{track.tid}_yamnet.npz')

    if recompute or not os.path.exists(feature_path):
        import tensorflow_hub as hub
        
        print(f'computing feature: yamnet for track {track.tid}...')
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        yamnet_audio, yamnet_sr = track.audio(sr=16000)
        _, yamnet_emb, _ = model(yamnet_audio)
        resampled_yamnet_emb = librosa.resample(
            yamnet_emb.numpy().T,
            orig_sr=1/0.48, target_sr=22050/4096
        )
        yamnet_ts = librosa.frames_to_time(
            np.arange(resampled_yamnet_emb.shape[-1]), 
            sr=22050, hop_length=4096)
        np.savez(feature_path, feature=resampled_yamnet_emb, ts=yamnet_ts)

    return np.load(feature_path)


def tempogram(track, recompute=False):
    feature_path = os.path.join(
        track.salami_dir, f'features/{track.tid}_tempogram.npz')

    if recompute or not os.path.exists(feature_path):
        track.audio()
        print(f'computing feature: tempogram for track {track.tid}...')
        novelty = track._novelty(hop_length=512, sr=track.sr)
        tempogram = librosa.feature.tempogram(
            onset_envelope=novelty, sr=track.sr, 
            hop_length=512, win_length=384)
        resampled_tempogram = librosa.resample(
            tempogram, orig_sr=track.sr/512, 
            target_sr=track.sr/4096)
        ts = librosa.frames_to_time(
            np.arange(resampled_tempogram.shape[-1]), 
            hop_length=4096, sr=track.sr)
        np.savez(feature_path, feature=resampled_tempogram, ts=ts)

    return np.load(feature_path)


def openl3(track, recompute=False):
    feature_path = os.path.join(
        track.salami_dir, f'features/{track.tid}_openl3.npz')

    if recompute or not os.path.exists(feature_path):
        import openl3

        track.audio()
        print(f'computing feature: openl3 for track {track.tid}...')
        emb, ts = openl3.audio_embedding(
            track.y, track.sr, 
            embedding_size=512, content_type='music'
        )
        resampled_emb = librosa.resample(
            emb.T, orig_sr=1 / (ts[1] - ts[0]),
            target_sr= track.sr/4096
        )
        ts = librosa.frames_to_time(
            np.arange(resampled_emb.shape[-1]), 
            hop_length=4096, sr=track.sr
        )
        np.savez(feature_path, feature=resampled_emb, ts=ts)

    return np.load(feature_path)


def chroma(track, recompute=False):
    feature_path = os.path.join(
        track.salami_dir, f'features/{track.tid}_chroma.npz')

    if recompute or not os.path.exists(feature_path):
        track.audio()
        print(f'computing feature: chroma for track {track.tid}...')
        chroma = librosa.feature.chroma_cqt(
            y=librosa.effects.harmonic(track.y, margin=8), 
            sr=track.sr, 
            hop_length=4096, 
            bins_per_octave=36)
        chroma_ts = librosa.frames_to_time(np.arange(chroma.shape[-1]), hop_length=4096, sr=track.sr)
        np.savez(feature_path, feature=chroma, ts=chroma_ts)

    return np.load(feature_path)
