import os
import sys
import time
from glob import glob

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import savemat
from tqdm import tqdm

from load_data import BFM, load_img, load_lm3d, transferBFM09
from preprocess_img import Preprocess
from reconstruct_mesh import Reconstruction_for_render, Render_layer


def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def demo(src_dir, tgt_dir, static_frame):

    def _iframe(x):
        return int(os.path.basename(os.path.splitext(x)[0]).replace("frame", ""))

    src_dir = os.path.abspath(src_dir)  # source data dir
    tgt_dir = os.path.abspath(tgt_dir)  # target re-enacting data dir
    # iden_coeff = np.load(iden_npy)

    # input and output folder
    coeff_list = glob(os.path.join(src_dir, "coeff_pred/*.npy"))
    coeff_list = sorted(coeff_list, key=_iframe)

    # get target images
    image_list = sorted(glob(os.path.join(tgt_dir, "crop/frame*.png")), key=_iframe)
    n_frames_tar = len(image_list)

    # check static_frame is valid
    static_frame = int(static_frame)
    if static_frame < 0:
        static_frame = None

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
            for iframe, file in enumerate(tqdm(coeff_list, desc="reenact")):
                n += 1
                assert iframe == _iframe(file)

                # makedirs for results
                fname = "frame" + str(iframe)
                save_dir_coeff = os.path.join(src_dir, "reenact/coeff")
                save_dir_render = os.path.join(src_dir, "reenact/render")
                create_dirs(save_dir_coeff)
                create_dirs(save_dir_render)

                # load coeffs
                coeff_pred = np.load(file)

                # load images and corresponding 5 facial landmarks
                if static_frame is not None:
                    k = static_frame % n_frames_tar
                else:
                    # bounce at two boundaries
                    d = iframe // n_frames_tar
                    if d % 2 == 0:  # increasing direction
                        k = iframe % n_frames_tar
                    else:  # decreasing direction
                        k = (n_frames_tar - 1) - iframe & n_frames_tar
                file = image_list[k]
                img, lm = load_img(file, file[:-4] + ".txt")

                # preprocess input image
                input_img, _, transform_params = Preprocess(img, lm, lm3D)
                coef = sess.run(coeff, feed_dict={images: input_img})
                
                # replace 
                # identity from reconstruction
                # coef[:, :80] = iden_coeff[:, :80]  # type: ignore
                # expression from prediction
                coef[:, 80:144] = coeff_pred[:64]  # type: ignore
                
                # reconstruct image
                face_shape_r, face_norm_r, face_color, _ = Reconstruction_for_render(coef, facemodel)
                final_images = sess.run(
                    rendered,
                    feed_dict={
                        faceshaper: face_shape_r.astype("float32"),
                        facenormr: face_norm_r.astype("float32"),
                        facecolor: face_color.astype("float32"),  # type: ignore
                    },
                )
                result_image = final_images[0, :, :, :]
                result_image = np.clip(result_image, 0.0, 1.0).copy(order="C")
                result_bytes = sess.run(encode_png, {rstimg: result_image * 255.0})

                result_output_path = os.path.join(save_dir_render, fname + "_render.png")
                with open(result_output_path, "wb") as output_file:
                    output_file.write(result_bytes)

                # reshape outputs
                input_img = np.squeeze(input_img)
                im = Image.fromarray(input_img[:, :, ::-1])
                cropped_output_path = os.path.join(save_dir_render, fname + ".png")
                im.save(cropped_output_path)

                # save output files
                savemat(os.path.join(save_dir_coeff, fname + ".mat"), {"coeff": coef, "transform_params": transform_params})
    t2 = time.time()
    print("Total n:", n, "Time:", t2 - t1)


if __name__ == "__main__":
    demo(sys.argv[1], sys.argv[2], sys.argv[3])
