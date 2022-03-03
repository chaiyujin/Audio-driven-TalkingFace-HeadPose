import os
import pdb
import sys
import time
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat, savemat
from tqdm import tqdm

from load_data import BFM, load_img, load_lm3d, transferBFM09
from preprocess_img import Preprocess, Preprocess2
from reconstruct_mesh import Reconstruction, Reconstruction_for_render, Render_layer
from yuki11.utils import mesh_viewer, VideoWriter

mesh_viewer.set_template("template.obj")
mesh_viewer.set_shading_mode("smooth")


def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def demo(apath, src_dir, tgt_dir, spk_dir):

    def _iframe(x):
        return int(os.path.basename(os.path.splitext(x)[0]).replace("frame", ""))

    src_dir = os.path.abspath(src_dir)
    tgt_dir = os.path.abspath(tgt_dir)
    spk_dir = os.path.abspath(spk_dir)

    recons_dir = os.path.join(spk_dir, "reconstructed", "train")
    files = glob(os.path.join(recons_dir, "**/frame*.mat"), recursive=True)
    all_ids = []
    for mat_path in files:
        iden = loadmat(mat_path)["coeff"][0]
        all_ids.append(iden)
    iden = np.asarray(all_ids).mean(0, keepdims=True).astype(np.float32)

    # input and output folder
    coeff_list = glob(os.path.join(src_dir, "coeff_pred/*.npy"))
    coeff_list = sorted(coeff_list, key=_iframe)
    # for x in coeff_list:
    #     assert x.find("test") >= 0

    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile("./BFM/BFM_model_front.mat"):
        transferBFM09()

    # read face model
    facemodel = BFM()
    # read standard landmarks for preprocessing images
    lm3D = load_lm3d()
    n = 0
    t1 = time.time()

    # build reconstruction model
    # with tf.Graph().as_default() as graph,tf.device('/cpu:0'):

    writer = VideoWriter(os.path.join(src_dir, "render.mp4"), fps=25, src_audio_path=apath, high_quality=True)

    for iframe, file in enumerate(tqdm(coeff_list, desc="gen3d")):
        n += 1
        assert iframe == _iframe(file)

        # load coeffs
        coeff_pred = np.load(file)

        coef = iden.copy()

        # ! replace expression part from predicted coefficients
        coef[:, 80:144] = coeff_pred[:64]
        # rigid is zero
        coef[:, 224:227] = 0
        coef[:, 254:257] = 0

        face_shape_r, face_norm_r, face_color, tri = Reconstruction_for_render(coef, facemodel)
        verts = face_shape_r[0].astype(np.float32) * 0.15
        im = mesh_viewer.render_verts(verts)
        writer.write(im)
        cv2.imshow('img', im)
        cv2.waitKey(40)

    writer.release()


if __name__ == "__main__":
    demo(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
