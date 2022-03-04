import os
import pickle
import re
import sys
from glob import glob
from shutil import rmtree

import cv2
import librosa
import numpy as np
import python_speech_features
import toml
from tqdm import tqdm

regex = re.compile(r'frame\d+\.png')


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


def fill_hole(trans):
    im_th = trans[..., 0]
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out[:, :, None]


def alpha_blend(srcdir, tardir, debug):
    done_flag = os.path.join(tardir, "done.flag")
    if os.path.exists(done_flag):
        return

    os.makedirs(tardir, exist_ok=True)

    files = glob(os.path.join(srcdir, "frame*.png"))
    files = [x for x in files if regex.match(os.path.basename(x)) is not None]
    files = sorted(files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])[5:]))

    for fpath in files:
        im1 = cv2.imread(fpath)
        im2 = cv2.imread(fpath[:-4] + '_render.png', cv2.IMREAD_UNCHANGED)
        trans = fill_hole(im2[:, :, -1:]).astype(np.float32) / 255.0
        im2 = im2[:, :, :-1]
        im3 = im1.astype(np.float32) * (1-trans) + im2.astype(np.float32) * trans
        im3 = im3.astype(np.uint8)
        fname = os.path.basename(fpath)[:-4]
        cv2.imwrite(os.path.join(tardir, fname + "_renderold_bm.png"), im3)
        if debug:
            ima = (trans * 255).astype(np.uint8).repeat(3, axis=-1)
            cv2.imshow('debug-alpha', np.concatenate((im3, ima)))
            cv2.waitKey(1)
    
    with open(done_flag, "w") as fp:
        fp.write("")


def build_r2v_dataset(data_dir, data_type, debug):
    names = ["bmold_win3", "bmold"]

    data_dir = os.path.abspath(data_dir)

    if data_type == "train":
        clip_dirs = find_clip_dirs(data_dir, data_type == "train", data_type == "test")
        r2v_dir = os.path.abspath(os.path.join(data_dir, "../r2v_dataset"))
    else:
        clip_dirs = [data_dir]
        r2v_dir = os.path.abspath(os.path.join(data_dir, "r2v_dataset"))

    # blend all clips
    for clip_dir in tqdm(clip_dirs, desc="blend", leave=False):
        render_dir = os.path.join(clip_dir, "render" if data_type == "train" else "reenact/render")
        assert os.path.exists(render_dir), "You haven't reconstruct 3D yet! ({})".format(clip_dir)
        if os.path.exists(render_dir):
            alpha_blend(render_dir, os.path.join(render_dir, "bm"), debug=debug)

    for name in names:
        assert "bmold" in name
        start = 0
        start1 = (start + 2) if "win3" in name else start

        if data_type == "train":
            # training dataset
            os.makedirs(os.path.join(r2v_dir, "list/trainA"), exist_ok=True)
            os.makedirs(os.path.join(r2v_dir, "list/trainB"), exist_ok=True)
            f1 = open(os.path.join(r2v_dir, f"list/trainA/{name}.txt"), "w")
            f2 = open(os.path.join(r2v_dir, f"list/trainB/{name}.txt"), "w")
            for clip_dir in clip_dirs:
                real_dir = os.path.join(clip_dir, "render")
                fake_dir = os.path.join(clip_dir, "render", "bm")
                n_frames = len(glob(os.path.join(fake_dir, "*.png")))
                for i in range(start1, start + n_frames):
                    print(os.path.join(fake_dir, "frame%d_renderold_bm.png" % i), file=f1)
                    print(os.path.join(real_dir, "frame%d.png" % i), file=f2)
            f1.close()
            f2.close()
        elif data_type == "test":
            # test dataset
            os.makedirs(os.path.join(r2v_dir, "list/testA"), exist_ok=True)
            os.makedirs(os.path.join(r2v_dir, "list/testB"), exist_ok=True)
            f1 = open(os.path.join(r2v_dir, f"list/testA/{name}.txt"), "w")
            f2 = open(os.path.join(r2v_dir, f"list/testB/{name}.txt"), "w")
            for clip_dir in clip_dirs:
                real_dir = os.path.join(clip_dir, "reenact/render")
                fake_dir = os.path.join(clip_dir, "reenact/render", "bm")
                n_frames = len(glob(os.path.join(fake_dir, "*.png")))
                for i in range(start1, start + n_frames):
                    print(os.path.join(fake_dir, "frame%d_renderold_bm.png" % i), file=f1)
                    print(os.path.join(real_dir, "frame%d.png" % i), file=f2)
            f1.close()
            f2.close()


if __name__ == "__main__":
    import argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHOICES = ['prepare_vocaset', 'prepare_celebtalk', 'build_r2v_dataset']

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=CHOICES)
    parser.add_argument("--data_dir", type=str, default=f"{ROOT}/yk_exp/vocaset/data")
    parser.add_argument("--vocaset_dir", type=str, default="~/assets/vocaset")
    parser.add_argument("--celebtalk_dir", type=str, default="~/assets/CelebTalk")
    parser.add_argument("--dest_size", type=int, default=256)
    parser.add_argument("--speaker", type=str, default="FaceTalk_170908_03277_TA")
    parser.add_argument("--use_seqs", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_type", type=str, choices=['train', 'test'])
    args = parser.parse_args()

    if args.mode == "prepare_vocaset":
        spk_dir = os.path.join(args.vocaset_dir, "Data", "videos_lmks_crop", args.speaker)
        prepare_vocaset(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, training=True)
        prepare_vocaset(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, training=False)
    elif args.mode == "prepare_celebtalk":
        spk_dir = os.path.join(args.celebtalk_dir, "Processed", args.speaker, "clips_cropped")
        use_seqs = args.use_seqs
        prepare_celebtalk(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, use_seqs=use_seqs, training=True)
        prepare_celebtalk(args.data_dir, spk_dir, dest_size=args.dest_size, debug=args.debug, use_seqs=use_seqs, training=False)
    elif args.mode == "build_r2v_dataset":
        build_r2v_dataset(args.data_dir, args.data_type, debug=args.debug)
