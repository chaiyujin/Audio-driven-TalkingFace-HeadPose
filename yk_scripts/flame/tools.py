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

regex = re.compile(r'frame\d+\.png')


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


def prepare_vocaset(output_root, data_root, training, dest_size=256, debug=False):
    # output root
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, "train" if training else "test")
    # source root
    data_root = os.path.expanduser(data_root)

    seq_range = list(range(0, 20)) if training else list(range(20, 40))
    for i_seq in tqdm(seq_range, desc=f"[prepare_vocaset]: {os.path.basename(data_root)}/{seq_range[0]}-{seq_range[-1]}"):
        # data source
        prefix = os.path.join(data_root, f"sentence{i_seq+1:02d}")
        vpath = prefix + ".mp4"
        lpath = prefix + "-lmks-ibug68.toml"

        out_dir = os.path.join(output_root, f"clip-sentence{i_seq+1:02d}")
        _preprocess_video(out_dir, vpath, lpath, dest_size, debug)


def prepare_celebtalk(output_root, data_root, training, dest_size=256, debug=False, use_seqs=()):
    # output root
    output_root = os.path.expanduser(output_root)
    output_root = os.path.join(output_root, "train" if training else "test")
    # source root
    data_root = os.path.expanduser(data_root)

    tasks = []
    for cur_root, _, files in os.walk(data_root):
        for fpath in files:
            seq_id = os.path.splitext(fpath)[0].replace("-fps25", "")
            if seq_id not in use_seqs:
                continue
            if training:
                if re.match(r"trn-\d+-fps25\.mp4", fpath) is not None:
                    tasks.append(os.path.join(cur_root, fpath))
            else:
                if re.match(r"vld-\d+-fps25\.mp4", fpath) is not None:
                    tasks.append(os.path.join(cur_root, fpath))
        break
    print(tasks)

    for vpath in tqdm(tasks, desc=f"[prepare_celebtalk]: {os.path.basename(data_root)}"):
        # data source
        lpath = os.path.splitext(vpath)[0] + "-lmks-ibug68.toml"
        assert os.path.exists(lpath)
        # output dir
        seq_id = os.path.basename(os.path.splitext(vpath)[0]).replace("-fps25", "")
        out_dir = os.path.join(output_root, f"clip-{seq_id}")
        # preprocess
        _preprocess_video(out_dir, vpath, lpath, dest_size, debug)


def _preprocess_video(out_dir, vpath, lpath, dest_size, debug):
    done_flag = os.path.join(out_dir, "done.flag")
    if os.path.exists(done_flag):
        return

    # prepare output dirs
    os.makedirs(os.path.join(out_dir, "full"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crop"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)

    # 1. -> 25 fps images and audio
    # assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} -r 25 {out_dir}/full/%05d.png") == 0
    assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} {out_dir}/full/%05d.png") == 0
    assert os.system(f"ffmpeg -loglevel error -hide_banner -y -i {vpath} {out_dir}/audio/audio.wav") == 0
    
    # 2. dump audio feature
    get_mfcc_extend(f"{out_dir}/audio/audio.wav", f"{out_dir}/audio/mfcc.npy")

    # 3. resize and dump landmarks
    with open(lpath) as fp:
        lmks_data = toml.load(fp)

    img_list = sorted(glob(f"{out_dir}/full/*.png"))
    for i_frm, img_path in enumerate(img_list):
        save_prefix = f"{out_dir}/crop/frame{i_frm}"
        img = cv2.imread(img_path)
        # fetch lmk
        pts = np.asarray(lmks_data['frames'][i_frm]['points'], dtype=np.float32)

        # resize
        pts[:, 0] = pts[:, 0] / img.shape[1] * dest_size
        pts[:, 1] = pts[:, 1] / img.shape[0] * dest_size
        img = cv2.resize(img, (dest_size, dest_size))

        # save
        eyel = np.round(np.mean(pts[36:42,:],axis=0)).astype("int")
        eyer = np.round(np.mean(pts[42:48,:],axis=0)).astype("int")
        nose = pts[33].astype("int")
        mouthl = pts[48].astype("int")
        mouthr = pts[54].astype("int")
        message = '%d %d\n%d %d\n%d %d\n%d %d\n%d %d\n' % (
            eyel[0], eyel[1],
            eyer[0], eyer[1],
            nose[0], nose[1],
            mouthl[0], mouthl[1],
            mouthr[0], mouthr[1]
        )
        with open(save_prefix + ".txt", "w") as fp:
            fp.write(message)
        cv2.imwrite(save_prefix + ".png", img)

        # debug
        if debug:
            for p in pts:
                c = (int(p[0]), int(p[1]))
                cv2.circle(img, c, 2, (0, 255, 0), -1)
            cv2.imshow('img', img)
            cv2.waitKey(1)
    # remove full
    rmtree(f"{out_dir}/full")

    with open(done_flag, "w") as fp:
        fp.write("")
    

def prepare_for_train_vocaset(out_root, src_root, speakers):
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
            coeffs = interpolate_features(np.asarray(coeffs), fps, output_rate=25.0).astype(np.float32)
            np.save(os.path.join(out_dir, "coeffs.npy"), coeffs)


if __name__ == "__main__":
    import argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHOICES = ['prepare_vocaset']

    REAL3D_SPEAKERS = [
        "FaceTalk_170725_00137_TA",
        "FaceTalk_170728_03272_TA",
        "FaceTalk_170731_00024_TA",
        "FaceTalk_170809_00138_TA",
        "FaceTalk_170811_03274_TA",
        "FaceTalk_170904_00128_TA",
        "FaceTalk_170912_03278_TA",
        "FaceTalk_170915_00223_TA",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=CHOICES)
    parser.add_argument("--data_dir", type=str, default=f"{ROOT}/yk_exp/vocaset/data")
    parser.add_argument("--data_src", type=str)
    parser.add_argument("--speaker", type=str)
    parser.add_argument("--source_dir", type=str, default="../../stylized-sa/data/datasets/talk_video/{}/data/{}")
    parser.add_argument("--vocaset_dir", type=str, default="../../stylized-sa/data/datasets/flame_mesh/vocaset_data")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mode == "prepare_vocaset":
        vocaset_data_dir = os.path.join(args.data_dir, "train")
        prepare_for_train_vocaset(vocaset_data_dir, args.vocaset_dir, REAL3D_SPEAKERS)

        # spk_dir = os.path.join(args.celebtalk_dir, "Processed", args.speaker, "clips_cropped")
        # use_seqs = args.use_seqs
        # prepare_celebtalk(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, use_seqs=use_seqs, training=True)
        # prepare_celebtalk(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, use_seqs=use_seqs, training=False)
