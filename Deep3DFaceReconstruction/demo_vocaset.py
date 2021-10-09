import glob
import os
import pdb
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat, savemat
from tqdm import tqdm

from load_data import BFM, load_img, load_lm3d, transferBFM09
from preprocess_img import Preprocess, Preprocess2
from reconstruct_mesh import Reconstruction, Reconstruction_for_render, Render_layer


def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def demo(speaker_dir):
    done_flag = os.path.join(speaker_dir, "done_recons.flag")
    if os.path.exists(done_flag):
        print("Reconstruction is already done for {}".format(speaker_dir))
        return

    # input and output folder
    img_list = glob.glob(os.path.join(speaker_dir, "**/train/clip**/crop/*.txt"), recursive=True)
    img_list = [e[:-4] + ".png" for e in img_list]
    img_list = sorted(img_list)
    for x in img_list:
        assert x.find("test") < 0
    print("img_list len:", len(img_list))

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
    with tf.Graph().as_default() as graph:

        images = tf.placeholder(name="input_imgs", shape=[None, 224, 224, 3], dtype=tf.float32)
        graph_def = load_graph("network/FaceReconModel.pb")
        tf.import_graph_def(graph_def, name="resnet", input_map={"input_imgs:0": images})

        # output coefficients of R-Net (dim = 257)
        coeff = graph.get_tensor_by_name("resnet/coeff:0")

        faceshaper = tf.placeholder(name="face_shape_r", shape=[1, 35709, 3], dtype=tf.float32)
        facenormr = tf.placeholder(name="face_norm_r", shape=[1, 35709, 3], dtype=tf.float32)
        facecolor = tf.placeholder(name="face_color", shape=[1, 35709, 3], dtype=tf.float32)
        rendered = Render_layer(faceshaper, facenormr, facecolor, facemodel, 1)

        rstimg = tf.placeholder(name="rstimg", shape=[224, 224, 4], dtype=tf.uint8)
        encode_png = tf.image.encode_png(rstimg)

        with tf.Session() as sess:
            for file in tqdm(img_list, desc="reconstructing"):
                n += 1
                # load images and corresponding 5 facial landmarks
                img, lm = load_img(file, file[:-4] + ".txt")
                # preprocess input image
                input_img, lm_new, transform_params = Preprocess(img, lm, lm3D)
                if n == 1:
                    transform_firstflame = transform_params
                input_img2, lm_new2 = Preprocess2(img, lm, transform_firstflame)

                coef = sess.run(coeff, feed_dict={images: input_img})

                face_shape_r, face_norm_r, face_color, tri = Reconstruction_for_render(coef, facemodel)
                final_images = sess.run(
                    rendered,
                    feed_dict={
                        faceshaper: face_shape_r.astype("float32"),
                        facenormr: face_norm_r.astype("float32"),
                        facecolor: face_color.astype("float32"),
                    },
                )
                result_image = final_images[0, :, :, :]
                result_image = np.clip(result_image, 0.0, 1.0).copy(order="C")
                result_bytes = sess.run(encode_png, {rstimg: result_image * 255.0})

                fname = os.path.basename(os.path.splitext(file)[0])
                seq_dir = os.path.dirname(os.path.dirname(file))
                save_dir_coeff = os.path.join(seq_dir, "coeff")
                save_dir_render = os.path.join(seq_dir, "render")
                create_dirs(save_dir_coeff)
                create_dirs(save_dir_render)

                result_output_path = os.path.join(save_dir_render, fname + "_render.png")
                with open(result_output_path, "wb") as output_file:
                    output_file.write(result_bytes)

                # reshape outputs
                input_img = np.squeeze(input_img)
                im = Image.fromarray(input_img[:, :, ::-1])
                cropped_output_path = os.path.join(save_dir_render, fname + ".png")
                im.save(cropped_output_path)

                input_img2 = np.squeeze(input_img2)
                im = Image.fromarray(input_img2[:, :, ::-1])
                cropped_output_path = os.path.join(save_dir_render, fname + "_input2.png")
                im.save(cropped_output_path)

                # save output files
                savemat(os.path.join(save_dir_coeff, fname + ".mat"), {"coeff": coef, "lm_5p": lm_new2 - lm_new})
    t2 = time.time()
    print("Total n:", n, "Time:", t2 - t1)

    # done flag
    with open(done_flag, "w") as fp:
        fp.write("")


if __name__ == "__main__":
    demo(sys.argv[1])
