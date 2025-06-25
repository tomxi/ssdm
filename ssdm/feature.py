import numpy as np
import librosa
import os

# Load local model
YAMNET_MODEL = None
_AUDIO_SR = 22050
_HOP_LEN = 4096

def yamnet(audio_path, output_path):
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        import tensorflow_hub as hub
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
    yamnet_audio, _ = librosa.load(audio_path, sr=16000)
    _, yamnet_emb, _ = YAMNET_MODEL(yamnet_audio)
    resampled_yamnet_emb = librosa.resample(
        yamnet_emb.numpy().T,
        orig_sr=1/0.48, target_sr=_AUDIO_SR / _HOP_LEN
    )
    yamnet_ts = librosa.frames_to_time(
        np.arange(resampled_yamnet_emb.shape[-1]), 
        sr=_AUDIO_SR, hop_length=_HOP_LEN)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, feature=resampled_yamnet_emb, ts=yamnet_ts)


def openl3(audio_path, output_path):
    import openl3
    y, sr = librosa.load(audio_path, sr=_AUDIO_SR)
    emb, ts = openl3.get_audio_embedding(
        y, sr, 
        embedding_size=512, content_type='music'
    )
    resampled_emb = librosa.resample(
        emb.T, orig_sr=1 / (ts[1] - ts[0]),
        target_sr= sr/_HOP_LEN
    )
    ts = librosa.frames_to_time(
        np.arange(resampled_emb.shape[-1]), 
        hop_length=_HOP_LEN, sr=sr
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, feature=resampled_emb, ts=ts)


def mfcc(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=_AUDIO_SR)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, 
        hop_length=_HOP_LEN, 
        n_fft=_HOP_LEN*2, 
        lifter=0.6
    )
    normalized_mfcc = (mfcc - np.mean(mfcc, axis=1)[:, None]) / np.std(mfcc, axis=1, ddof=1)[:,None]
    mfcc_ts = librosa.frames_to_time(
        np.arange(normalized_mfcc.shape[-1]), 
        hop_length=_HOP_LEN, sr=sr, n_fft=_HOP_LEN*2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, feature=normalized_mfcc, ts=mfcc_ts)


def tempogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=_AUDIO_SR)
    novelty = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempogram = librosa.feature.tempogram(
        onset_envelope=novelty, sr=sr, 
        hop_length=512, win_length=384)
    resampled_tempogram = librosa.resample(
        tempogram, orig_sr=sr/512, 
        target_sr=sr/_HOP_LEN)
    ts = librosa.frames_to_time(
        np.arange(resampled_tempogram.shape[-1]), 
        hop_length=_HOP_LEN, sr=sr)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, feature=resampled_tempogram, ts=ts)


def crema(audio_path, output_path):
    import crema
    crema_model = crema.models.chord.ChordModel()

    crema_out = crema_model.outputs(audio_path)  
    crema_op = crema_model.pump.ops[2]
    
    resampled_crema_pitch = librosa.resample(
        crema_out['chord_pitch'].T, 
        orig_sr=crema_op.sr / crema_op.hop_length, 
        target_sr=22050/4096
    )
    
    resampled_crema_root = librosa.resample(
        crema_out['chord_root'].T, 
        orig_sr=crema_op.sr / crema_op.hop_length, 
        target_sr=22050/4096
    )
    
    resampled_crema_bass = librosa.resample(
        crema_out['chord_bass'].T, 
        orig_sr = crema_op.sr / crema_op.hop_length, 
        target_sr = 22050/4096
    )

    crema_ts = librosa.frames_to_time(
        np.arange(resampled_crema_pitch.shape[-1]), 
        hop_length=4096, sr=22050)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, 
             pitch=resampled_crema_pitch, 
             root=resampled_crema_root,
             bass=resampled_crema_bass,
             feature=resampled_crema_pitch,
             ts=crema_ts)


def crema_full(audio_path, output_path):
    import crema
    crema_model = crema.models.chord.ChordModel()

    crema_out = crema_model.outputs(audio_path)  
    crema_op = crema_model.pump.ops[2]
    
    resampled_crema_pitch = librosa.resample(
        crema_out['chord_pitch'].T, 
        orig_sr=crema_op.sr / crema_op.hop_length, 
        target_sr=22050/4096
    )
    
    resampled_crema_root = librosa.resample(
        crema_out['chord_root'].T, 
        orig_sr=crema_op.sr / crema_op.hop_length, 
        target_sr=22050/4096
    )
    
    resampled_crema_bass = librosa.resample(
        crema_out['chord_bass'].T, 
        orig_sr = crema_op.sr / crema_op.hop_length, 
        target_sr = 22050/4096
    )

    crema_ts = librosa.frames_to_time(
        np.arange(resampled_crema_pitch.shape[-1]), 
        hop_length=4096, sr=22050)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, 
             pitch=resampled_crema_pitch, 
             root=resampled_crema_root,
             bass=resampled_crema_bass,
             feature=resampled_crema_pitch,
             ts=crema_ts)


FEAT_MAP = dict(
    yamnet = yamnet,
    crema = crema,
    mfcc = mfcc,
    tempogram = tempogram,
    openl3 = openl3,
    crema_full = crema_full,
)

