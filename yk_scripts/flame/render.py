import os
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from glob import glob
from Audio.code.flame import FLAME
from ..meshio import load_mesh
from yuki11.utils import mesh_viewer, VideoWriter

mesh_viewer.set_template("assets/FLAME_face.obj")
mesh_viewer.set_shading_mode("smooth")
with open("assets/FLAME_face.txt") as fp:
    line = ' '.join(x.strip() for x in fp)
    vidx = [int(x) for x in line.split()]

device = "cpu"
morphable_model = FLAME()
morphable_model.to(device)


def load_expressions(exp_path):
    fpaths = sorted(glob(os.path.join(exp_path, "*.npy")))
    coeffs = [np.load(x) for x in fpaths]
    return np.asarray(coeffs, dtype=np.float32)


def render_results_for(spk_root, fps, dump_meshes=False):
    # TODO: load identity
    # iden = np.load(os.path.join(spk_root, "data", "identity", "shape.npy"))
    # iden = torch.tensor(iden, dtype=torch.float32, device=device)
    iden_verts = load_mesh("assets/FLAME_sample.obj")[0]

    tasks = []
    for cur_root, subdirs, _ in os.walk(os.path.join(spk_root)):
        for subdir in subdirs:
            dirpath = os.path.join(cur_root, subdir)
            if subdir == "coeff_pred":
                tasks.append(dirpath)

    pbar = tqdm(tasks)
    for dirpath in pbar:
        dirpath = os.path.dirname(dirpath)
        pbar.set_description("Render", os.path.basename(dirpath))
        exps = load_expressions(os.path.join(dirpath, "coeff_pred"))
        exps = torch.tensor(exps, dtype=torch.float32)

        out_vpath = os.path.join(dirpath, "render.mp4")
        out_meshes = os.path.join(dirpath, "meshes")
        flag_meshes = os.path.join(out_meshes, "done.lock")
        if dump_meshes:
            os.makedirs(out_meshes, exist_ok=True)
        else:
            out_meshes = None
        
        if os.path.exists(out_vpath) and (out_meshes is None or os.path.exists(flag_meshes)):
            tqdm.write("Skip to render {}.".format(out_vpath))
            continue

        apath = os.path.join(dirpath, "audio.wav")
        if not os.path.exists(apath):
            apath = None
        writer = VideoWriter(out_vpath, fps=fps, src_audio_path=apath, high_quality=True)

        def _get_codes(coeff):
            codes = dict(jaw_pose=coeff[..., :3], exp=coeff[..., 3:])
            if coeff.ndim == 1:
                for k in codes:
                    codes[k] = codes[k][None, ...]
            for k in codes:
                codes[k] = torch.tensor(codes[k], dtype=torch.float32, device=device)
            return codes

        for i in trange(len(exps), leave=False):
            off = morphable_model(_get_codes(exps[i])) - morphable_model.v_template.unsqueeze(0)
            verts = off[0].cpu().numpy() + iden_verts
            if out_meshes is not None:
                verts_npy = os.path.join(out_meshes, f"{i:06d}.npy")
                np.save(verts_npy, verts)
            verts = verts[vidx] * 1.4
            verts[:, 1] += 0.02
            im = mesh_viewer.render_verts(verts)[:, :, [2, 1, 0]]
            writer.write(im)
        writer.release()
        if out_meshes is not None:
            with open(flag_meshes, "w"):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--dump_meshes", action="store_true")
    args, _ = parser.parse_known_args()

    render_results_for(args.exp_dir, args.fps, args.dump_meshes)
