import face_model
import argparse
import cv2
import numpy as np
import os
import glob
import time
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='models/model-r100-ii/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='models/gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=1, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--imglist', required=True, help='imglist name')
    parser.add_argument('--listdir', default='../datasets/list/', help='dir to imglist')
    args = parser.parse_args()
    model = face_model.FaceModel(args)
    if not os.path.isdir(os.path.join(args.listdir,args.imglist)):
        imglist = open(os.path.join(args.listdir,args.imglist),'r').read().splitlines()
    else:
        imglist = glob.glob(args.imglist+'/*/*.png')
        imglist = [e for e in imglist if '_input2' not in e and '_render' not in e]
    print('imglist len',len(imglist))
    t0 = time.time()
    for i in tqdm(range(len(imglist))):
        imgname = imglist[i]
        clip_dir = os.path.dirname(os.path.dirname(imgname))
        subname = os.path.basename(os.path.dirname(imgname))
        save_dir = os.path.join(clip_dir, "arcface", subname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        savename = os.path.join(save_dir, os.path.basename(os.path.splitext(imgname)[0]) + '.npy')
        # print(imgname)
        # print(savename)
        if os.path.exists(savename):
            continue
        img = cv2.imread(imgname)
        img = model.get_input(img)
        if type(img) != np.ndarray:
            print(imgname,'Not detected')
            continue
        f1 = model.get_feature(img)
        np.save(savename,f1)
        if i % 1000 == 1:
            print('saved',i,time.time()-t0)
            t0 = time.time()
