import os
import pickle
import re
import sys
from glob import glob
from shutil import rmtree

import cv2
import json
import librosa
import numpy as np
import python_speech_features
import toml
from shutil import copyfile
from tqdm import tqdm
from ..meshio import load_mesh

regex = re.compile(r'frame\d+\.png')


def _correct_avoffset_for_visual(vframes, avoffset):
    if avoffset > 0:  # audio ts larger than video, so delay video
        padding = vframes[:1].repeat(avoffset, axis=0)
        new_vframes = np.concatenate((padding, vframes), axis=0)
        return new_vframes
    elif avoffset < 0:  # audio ts smaller than video, so clip video
        return vframes[-avoffset:]


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps, input_timestamps, features[:, feat])
    return output_features


def find_clip_dirs(data_dir, with_train, with_test):
    # find clips
    clip_dirs = []
    for dirpath, subdirs, _ in os.walk(data_dir):
        # check if train data
        if os.path.basename(dirpath) == 'train' and not with_train:
            continue
        # check if test data
        if os.path.basename(dirpath) == 'test' and not with_test:
            continue
        # collect
        for subdir in subdirs:
            if subdir.startswith("clip") and (
                os.path.exists(os.path.join(dirpath, subdir, "crop")) or
                os.path.exists(os.path.join(dirpath, subdir, "coeff"))
            ):
                clip_dirs.append(os.path.join(dirpath, subdir))
    clip_dirs = sorted(clip_dirs)
    return clip_dirs


def get_mfcc_extend(test_file, save_file):
    if os.path.exists(save_file):
        mfcc = np.load(save_file)
        return mfcc
    speech, sr = librosa.load(test_file, sr=16000)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, mfcc)
    return mfcc


def prepare_talk_video(output_root, data_root, data_src, avoffset_ms, training, dest_size=256, debug=False):
    # output root
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, "train" if training else "test")
    # source root
    data_root = os.path.expanduser(data_root)

    # copy fitted identity
    fit_iden_dir = os.path.join(data_root, "fitted", "identity")
    assert os.path.exists(fit_iden_dir)
    save_iden_dir = os.path.join(os.path.dirname(output_root), "identity")
    os.makedirs(save_iden_dir, exist_ok=True)
    if (
        not os.path.exists(os.path.join(save_iden_dir, "identity.obj")) or
        not os.path.exists(os.path.join(fit_iden_dir, "shape.npy"))
    ):
        copyfile(os.path.join(fit_iden_dir, "shape.npy"), os.path.join(save_iden_dir, "shape.npy"))
        copyfile(os.path.join(fit_iden_dir, "identity.obj"), os.path.join(save_iden_dir, "identity.obj"))
    
    iden_verts = load_mesh(os.path.join(save_iden_dir, "identity.obj"))[0]

    # find sequences

    tasks = []
    for cur_root, subdirs, _ in os.walk(data_root):
        for subdir in subdirs:
            absdir = os.path.join(cur_root, subdir)
            if data_src == "celebtalk":
                if training:
                    if re.match(r"trn-\d+", subdir) is not None:
                        tasks.append(absdir)
                else:
                    if re.match(r"vld-\d+", subdir) is not None:
                        tasks.append(absdir)
            else:
                raise NotImplementedError()
        break
    print(tasks, data_src)

    for src_dir in tqdm(tasks, desc=f"[prepare_talk_video]: {os.path.basename(data_root)}"):
        with open(os.path.join(src_dir, "info.json")) as fp:
            info = json.load(fp)
            fps = info['fps']
        # output dir
        seq_id = os.path.basename(src_dir)
        fit_dir = os.path.join(data_root, "fitted", seq_id)
        out_dir = os.path.join(output_root, f"clip-{seq_id}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # audio
        if not os.path.exists(os.path.join(out_dir, "audio/audio.wav")):
            os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
            copyfile(os.path.join(src_dir, "audio.wav"), os.path.join(out_dir, "audio/audio.wav"))
        # audio feature
        if not os.path.exists(os.path.join(out_dir, "audio/mfcc.npy")):
            get_mfcc_extend(os.path.join(out_dir, "audio/audio.wav"), os.path.join(out_dir, "audio/mfcc.npy"))

        # coeffs
        if not os.path.exists(os.path.join(out_dir, "coeffs.npy")):
            coeffs = []
            fpaths = sorted(glob(os.path.join(fit_dir, "frames", "*.npz")))
            for i, fpath in enumerate(fpaths):
                assert i == int(os.path.basename(os.path.splitext(fpath)[0]))
                dat = np.load(fpath)
                jaw = dat['jaw']
                exp = dat['exp']
                coeff = np.concatenate((jaw, exp))
                assert len(coeff) == 53
                coeffs.append(coeff)
            coeffs = np.asarray(coeffs)
            
            # > Correct avoffset
            if avoffset_ms is not None:
                avoffset = int(np.round(fps * avoffset_ms / 1000.0))
                assert avoffset == info['avoffset'] and avoffset_ms == info['avoffset_ms'], \
                    "Given avoffset_ms {} != data's {}".format(avoffset_ms, info['avoffset_ms'])
                tqdm.write("> Correct avoffset {} ms".format(avoffset_ms)) 
                coeffs = _correct_avoffset_for_visual(coeffs, avoffset)

            # downsample to 25fps
            coeffs = interpolate_features(coeffs, fps, output_rate=25.0).astype(np.float32)
            np.save(os.path.join(out_dir, "coeffs.npy"), coeffs)

        # offsets
        if not os.path.exists(os.path.join(out_dir, "offsets.npy")):
            offsets = []
            fpaths = sorted(glob(os.path.join(fit_dir, "meshes", "*.npy")))
            for i, fpath in enumerate(fpaths):
                assert i == int(os.path.basename(os.path.splitext(fpath)[0]))
                verts = np.load(fpath)
                offsets.append(verts - iden_verts)  # remove identity
            offsets = np.asarray(offsets)

            # > Correct avoffset
            if avoffset_ms is not None:
                avoffset = int(np.round(fps * avoffset_ms / 1000.0))
                assert avoffset == info['avoffset'] and avoffset_ms == info['avoffset_ms'], \
                    "Given avoffset_ms {} != data's {}".format(avoffset_ms, info['avoffset_ms'])
                tqdm.write("> Correct avoffset {} ms".format(avoffset_ms)) 
                offsets = _correct_avoffset_for_visual(offsets, avoffset)
            
            # downsample to 25fps
            offsets = np.reshape(offsets, (len(offsets), -1))
            offsets = interpolate_features(offsets, fps, output_rate=25.0)
            offsets = np.reshape(offsets, (len(offsets), -1, 3)).astype(np.float32)
            np.save(os.path.join(out_dir, "offsets.npy"), offsets)


def prepare_for_train_vocaset(out_root, src_root, speakers, avoffset_ms):
    tasks = []
    for speaker in speakers:
        for sent_id in range(1, 41):
            tgt_dir = os.path.join(out_root, speaker, f'clip-sentence{sent_id:02}')
            src_dir = os.path.join(src_root, speaker, f'sentence{sent_id:02}')
            if os.path.exists(src_dir):
                tasks.append((tgt_dir, os.path.abspath(src_dir)))

    for (out_dir, src_dir) in tqdm(tasks):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # audio
        if not os.path.exists(os.path.join(out_dir, "audio/audio.wav")):
            os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
            copyfile(os.path.join(src_dir, "audio.wav"), os.path.join(out_dir, "audio/audio.wav"))
        # audio feature
        if not os.path.exists(os.path.join(out_dir, "audio/mfcc.npy")):
            get_mfcc_extend(os.path.join(out_dir, "audio/audio.wav"), os.path.join(out_dir, "audio/mfcc.npy"))
        # offsets
        if not os.path.exists(os.path.join(out_dir, "offsets.npy")):
            offsets = np.load(os.path.join(src_dir, "offsets.npy"))
            with open(os.path.join(src_dir, "info.json")) as fp:
                info = json.load(fp)
                fps = info['fps']

            # > Correct avoffset
            if avoffset_ms is not None:
                avoffset_ms = info['avoffset_ms']
                avoffset = int(np.round(fps * avoffset_ms / 1000.0))
                assert avoffset == info['avoffset'] and avoffset_ms == info['avoffset_ms'], \
                    "Given avoffset_ms {} != data's {}".format(avoffset_ms, info['avoffset_ms'])
                tqdm.write("> Correct avoffset {} ms for VOCASET".format(avoffset_ms)) 
                offsets = _correct_avoffset_for_visual(offsets, avoffset)

            # downsample to 25fps
            offsets = np.reshape(offsets, (len(offsets), -1))
            offsets = interpolate_features(offsets, fps, output_rate=25.0)
            offsets = np.reshape(offsets, (len(offsets), -1, 3)).astype(np.float32)
            np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        # coeffs
        if not os.path.exists(os.path.join(out_dir, "coeffs.npy")):
            with open(os.path.join(src_dir, "info.json")) as fp:
                info = json.load(fp)
                fps = info['fps']
            ss = src_dir.split('/')
            ss[-3] = 'vocaset_fitted'
            fit_dir = '/'.join(ss)
            coeffs = []
            fpaths = sorted(glob(os.path.join(fit_dir, "frames", "*.npz")))
            for i, fpath in enumerate(fpaths):
                assert i == int(os.path.basename(os.path.splitext(fpath)[0]))
                dat = np.load(fpath)
                jaw = dat['jaw']
                exp = dat['exp']
                coeff = np.concatenate((jaw, exp))
                assert len(coeff) == 53
                coeffs.append(coeff)
            coeffs = np.asarray(coeffs)

            # > Correct avoffset
            if avoffset_ms is not None:
                avoffset_ms = info['avoffset_ms']
                avoffset = int(np.round(fps * avoffset_ms / 1000.0))
                assert avoffset == info['avoffset'] and avoffset_ms == info['avoffset_ms'], \
                    "Given avoffset_ms {} != data's {}".format(avoffset_ms, info['avoffset_ms'])
                tqdm.write("> Correct avoffset {} ms for VOCASET".format(avoffset_ms)) 
                coeffs = _correct_avoffset_for_visual(coeffs, avoffset)

            # downsample to 25fps
            coeffs = interpolate_features(coeffs, fps, output_rate=25.0).astype(np.float32)
            np.save(os.path.join(out_dir, "coeffs.npy"), coeffs)


if __name__ == "__main__":
    import argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHOICES = ['prepare_vocaset', 'prepare_talk_video']

    # # 8 speakers
    # REAL3D_SPEAKERS = [
    #     "FaceTalk_170725_00137_TA",
    #     "FaceTalk_170728_03272_TA",
    #     "FaceTalk_170731_00024_TA",
    #     "FaceTalk_170809_00138_TA",
    #     "FaceTalk_170811_03274_TA",
    #     "FaceTalk_170904_00128_TA",
    #     "FaceTalk_170912_03278_TA",
    #     "FaceTalk_170915_00223_TA",
    # ]

    # 12 speakers (all)
    REAL3D_SPEAKERS = [
        "FaceTalk_170725_00137_TA",
        "FaceTalk_170728_03272_TA",
        "FaceTalk_170731_00024_TA",
        "FaceTalk_170809_00138_TA",
        "FaceTalk_170811_03274_TA",
        "FaceTalk_170811_03275_TA",
        "FaceTalk_170904_00128_TA",
        "FaceTalk_170904_03276_TA",
        "FaceTalk_170908_03277_TA",
        "FaceTalk_170912_03278_TA",
        "FaceTalk_170913_03279_TA",
        "FaceTalk_170915_00223_TA",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=CHOICES)
    parser.add_argument("--data_dir", type=str, default=f"{ROOT}/yk_exp/vocaset/data")
    parser.add_argument("--data_src", type=str)
    parser.add_argument("--speaker", type=str)
    parser.add_argument("--avoffset_ms", type=float)
    parser.add_argument("--source_dir", type=str, default="../../stylized-sa/data/datasets/talk_video/{}/data/{}")
    parser.add_argument("--vocaset_dir", type=str, default="../../stylized-sa/data/datasets/flame_mesh/vocaset_data")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mode == "prepare_vocaset":
        vocaset_data_dir = os.path.join(args.data_dir, "train")
        prepare_for_train_vocaset(vocaset_data_dir, args.vocaset_dir, REAL3D_SPEAKERS, args.avoffset_ms)
    elif args.mode == "prepare_talk_video":
        out_dir = args.data_dir
        src_dir = args.source_dir.format(args.data_src, args.speaker)
        prepare_talk_video(out_dir, src_dir, args.data_src, args.avoffset_ms, debug=args.debug, training=True)
        prepare_talk_video(out_dir, src_dir, args.data_src, args.avoffset_ms, debug=args.debug, training=False)
