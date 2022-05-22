import os
import cv2
import argparse
import numpy as np
from PIL import Image
from glob import glob
from scipy.io import loadmat


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
        # r2v_path = os.path.join(os.path.dirname(r2v_dir), "reenact/render", "bm", f"frame{iframe}_renderold_bm.png")

        im_r2v = cv2.imread(r2v_path)
        im_msk = 255-cv2.imread(msk_path)

        def _maskout_boundary(mask):
            P = 4
            mask[:P] = 0
            mask[-P:] = 0
            mask[:, :P] = 0
            mask[:, -P:] = 0
            return mask

        im_msk = _maskout_boundary(im_msk)

        # transform back
        transform_params = loadmat(coe_path)["transform_params"]
        transform_params = [x[0,0] for x in transform_params[0]]
        im_inv, valid = inverse_transform(im_r2v, transform_params)
        im_msk, _ = inverse_transform(im_msk, transform_params)

        # get mask of face
        mask: np.ndarray = np.where(valid == 255, im_msk, np.zeros_like(im_msk))
        mask = (np.all(mask >= 250, axis=-1, keepdims=True) * 255).astype(np.uint8)
        mask = fill_hole(mask).repeat(3, axis=-1)

        # merge
        img = cv2.imread(img_path)
        merged = np.where(mask == 255, im_inv, img)

        cv2.imwrite(os.path.join(output_dir, f"{iframe:06d}_real.png"), img)
        cv2.imwrite(os.path.join(output_dir, f"{iframe:06d}_fake.png"), merged)
        cv2.imwrite(os.path.join(output_dir, f"{iframe:06d}_mask.png"), mask)
        np.save(os.path.join(output_dir, f"{iframe:06d}_trans.npy"), transform_params)
        # print(transform_params)

        # cv2.imshow("valid", mask)
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
