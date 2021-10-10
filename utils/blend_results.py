import os
import cv2
import argparse
import numpy as np
from PIL import Image
from glob import glob
from scipy.io import loadmat


def inverse_transform(img, transform_params):
    rec_img = Image.fromarray(img)

    w0, h0 = transform_params[:2]
    s = transform_params[2]
    t = transform_params[3:]

    w2 = (w0*s).astype(np.int32)
    h2 = (h0*s).astype(np.int32)
    # crop the image from center
    x2 = (112 - w2/2).astype(np.int32)
    y2 = (112 - h2/2).astype(np.int32)

    rec_img = rec_img.resize((224, 224), resample = Image.LANCZOS)
    rec_img = rec_img.crop((x2, y2, x2+w2, y2+h2))
    rec_img = rec_img.resize((w0, h0), resample = Image.LANCZOS)

    # x2 = x2 * img.shape[1] / 224
    # w2 = w2 * img.shape[1] / 224
    # y2 = y2 * img.shape[0] / 224
    # h2 = h2 * img.shape[0] / 224
    # rec_img = rec_img.transform((w0, h0), Image.AFFINE, (w2/w0, 0, +x2, 0, h2/h0, +y2), resample=Image.BICUBIC)

    rec_img = rec_img.transform((w0, h0), Image.AFFINE, (1, 0, w0/2 - t[0], 0, 1,  t[1] - h0/2))
    warp_dst = np.array(rec_img)
    rx0, ry0 = t[0] - w0/2, h0/2 - t[1] 
    rx1, ry1 = rx0 + w0, ry0 + h0
    rx0, ry0 = max(0,  int(np.ceil (rx0))) + 4, max(0,  int(np.ceil (ry0))) + 4
    rx1, ry1 = min(w0, int(np.floor(rx1))) - 4, min(h0, int(np.floor(ry1))) - 4
    valid = warp_dst.copy()
    valid[ry0:ry1, rx0:rx1] = 255

    return warp_dst, valid


def blend_results(output_dir, r2v_dir, coeff_dir, image_dir):
    def _iframe(path):
        x = os.path.basename(path).split("_")[0]
        return int(x.replace("bm-frame", ""))

    r2v_list = glob(os.path.join(r2v_dir, "bm-frame*_renderold_bm_fake_B.png"))
    r2v_list = sorted(r2v_list, key=lambda x: _iframe(x))

    os.makedirs(output_dir, exist_ok=True)

    for r2v_path in r2v_list:
        iframe = _iframe(r2v_path)
        coe_path = os.path.join(coeff_dir, f"frame{iframe}.mat")
        img_path = os.path.join(image_dir, f"frame{iframe}.png")
        msk_path = os.path.join(r2v_dir, f"bm-frame{iframe}_renderold_bm_fake_B_mask_vis.png")
        # r2v_path = os.path.join(os.path.dirname(r2v_dir), "render_reenact", "bm", f"frame{iframe}_renderold_bm.png")

        # transform back
        transform_params = loadmat(coe_path)["transform_params"]
        transform_params = [x[0,0] for x in transform_params[0]]
        im_r2v = cv2.imread(r2v_path)
        im_inv, valid = inverse_transform(im_r2v, transform_params)
        im_msk, _ = inverse_transform(255-cv2.imread(msk_path), transform_params)

        # merge
        img = cv2.imread(img_path)
        merged = np.where(valid == 255, im_inv, img)
        mask = np.where(valid == 255, im_msk, np.zeros_like(im_msk))

        cv2.imwrite(os.path.join(output_dir, f"{iframe:06d}_real.png"), img)
        cv2.imwrite(os.path.join(output_dir, f"{iframe:06d}_fake.png"), merged)
        cv2.imwrite(os.path.join(output_dir, f"{iframe:06d}_mask.png"), mask)

        # masked = np.where((mask == 255).all(axis=-1, keepdims=True), merged, np.zeros_like(merged))
        # cv2.imshow('img', np.concatenate((img, merged, mask, masked), axis=1))
        # cv2.waitKey(16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--r2v_dir", required=True)
    parser.add_argument("--coeff_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    args = parser.parse_args()

    blend_results(args.output_dir, args.r2v_dir, args.coeff_dir, args.image_dir)
