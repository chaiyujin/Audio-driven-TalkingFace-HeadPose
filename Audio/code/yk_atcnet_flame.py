import argparse
import glob
import os
import pdb
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchvision
from torch.autograd import Variable
from torch.nn import init
from torch.nn.modules.module import _addindent
from torch.utils.data import DataLoader

from models import ATC_net
from yk_dataset_flame import MultiClips_1D_lstm_3dmm


def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id == 1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def initialize_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class Trainer:
    def __init__(self, config):
        if config.lstm:
            if config.pose == 0:
                self.generator = ATC_net(config.para_dim)
            else:
                self.generator = ATC_net(config.para_dim + 6)
        print("---------- Networks initialized -------------")
        num_params = 0
        for param in self.generator.parameters():
            num_params += param.numel()
        print("[Network] Total number of parameters : %.3f M" % (num_params / 1e6))
        print("-----------------------------------------------")
        # pdb.set_trace()
        self.l1_loss_fn = nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(",")]
            if len(device_ids) > 1:
                self.generator = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            else:
                self.generator = self.generator.cuda()
            self.mse_loss_fn = self.mse_loss_fn.cuda()
            self.l1_loss_fn = self.l1_loss_fn.cuda()
        initialize_weights(self.generator)
        if config.continue_train:
            state_dict = multi2single(config.model_name, 0)
            self.generator.load_state_dict(state_dict)
            print("load pretrained [{}]".format(config.model_name))
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        if config.lstm:
            assert config.pose == 0
            assert config.dataset == "multi_clips"
            self.dataset = MultiClips_1D_lstm_3dmm(
                config.dataset_dir,
                train=config.is_train,
                relativeframe=config.relativeframe,
            )
        else:
            raise NotImplementedError()

        self.data_loader = DataLoader(
            self.dataset, batch_size=config.batch_size, num_workers=config.num_thread, shuffle=True, drop_last=True
        )
        if config.dataset == "lrw":
            self.data_loader_val = DataLoader(
                self.dataset2,
                batch_size=config.batch_size,
                num_workers=config.num_thread,
                shuffle=False,
                drop_last=True,
            )

    def fit(self):
        config = self.config
        L = config.para_dim

        num_steps_per_epoch = len(self.data_loader)
        print("num_steps_per_epoch", num_steps_per_epoch)
        cc = 0
        t00 = time.time()
        t0 = time.time()

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (coeff, audio, coeff2) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    coeff = Variable(coeff.float()).cuda()
                    audio = Variable(audio.float()).cuda()
                else:
                    coeff = Variable(coeff.float())
                    audio = Variable(audio.float())

                # print(audio.shape, coeff.shape) # torch.Size([16, 16, 28, 12]) torch.Size([16, 16, 70])
                fake_coeff = self.generator(audio)

                loss = self.mse_loss_fn(fake_coeff, coeff)

                if config.less_constrain:
                    assert self.config.pose == 1
                    loss = self.mse_loss_fn(
                        fake_coeff[:, :, :L], coeff[:, :, :L]
                    ) + config.lambda_pose * self.mse_loss_fn(fake_coeff[:, :, L:], coeff[:, :, L:])

                # put smooth on pose
                # tidu ermo pingfang
                if config.smooth_loss:
                    assert self.config.pose == 1
                    loss1 = loss.clone()
                    frame_dif = fake_coeff[:, 1:, L:] - fake_coeff[:, :-1, L:]  # [16, 15, 6]
                    # norm2 = torch.norm(frame_dif, dim = 1) # default 2-norm, [16, 6]
                    # norm2_ss1 = torch.sum(torch.mul(norm2, norm2), dim=1) # [16, 1]
                    norm2_ss = torch.sum(torch.mul(frame_dif, frame_dif), dim=[1, 2])
                    loss2 = torch.mean(norm2_ss)
                    # pdb.set_trace()
                    loss = loss1 + loss2 * config.lambda_smooth

                # put smooth on expression
                if config.smooth_loss2:
                    loss3 = loss.clone()
                    frame_dif2 = fake_coeff[:, 1:, :L] - fake_coeff[:, :-1, :L]
                    norm2_ss2 = torch.sum(torch.mul(frame_dif2, frame_dif2), dim=[1, 2])
                    loss4 = torch.mean(norm2_ss2)
                    loss = loss3 + loss4 * config.lambda_smooth2

                loss.backward()
                self.opt_g.step()
                self._reset_gradients()

                if (step + 1) % 10 == 0 or (step + 1) == num_steps_per_epoch:
                    steps_remain = (
                        num_steps_per_epoch - step + 1 + (config.max_epochs - epoch + 1) * num_steps_per_epoch
                    )

                    if not config.smooth_loss and not config.smooth_loss2:
                        print(
                            "[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second".format(
                                epoch + 1,
                                config.max_epochs,
                                step + 1,
                                num_steps_per_epoch,
                                loss,
                                t1 - t0,
                                time.time() - t1,
                            )
                        )
                    elif config.smooth_loss and not config.smooth_loss2:
                        print(
                            "[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv: {:.8f},data time: {:.4f},  model time: {} second".format(
                                epoch + 1,
                                config.max_epochs,
                                step + 1,
                                num_steps_per_epoch,
                                loss,
                                loss1,
                                loss2 * config.lambda_smooth,
                                t1 - t0,
                                time.time() - t1,
                            )
                        )
                    elif not config.smooth_loss and config.smooth_loss2:
                        print(
                            "[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv2: {:.8f},data time: {:.4f},  model time: {} second".format(
                                epoch + 1,
                                config.max_epochs,
                                step + 1,
                                num_steps_per_epoch,
                                loss,
                                loss3,
                                loss4 * config.lambda_smooth2,
                                t1 - t0,
                                time.time() - t1,
                            )
                        )
                    else:
                        print(
                            "[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv: {:.8f},losstv2: {:.8f},data time: {:.4f},  model time: {} second".format(
                                epoch + 1,
                                config.max_epochs,
                                step + 1,
                                num_steps_per_epoch,
                                loss,
                                loss1,
                                loss2 * config.lambda_smooth,
                                loss4 * config.lambda_smooth2,
                                t1 - t0,
                                time.time() - t1,
                            )
                        )

                t0 = time.time()
            if (epoch + 1) % config.save_per_epochs == 0:
                print("[{}/{}][{}/{}]   save model".format(epoch + 1, config.max_epochs, step + 1, num_steps_per_epoch))
                torch.save(self.generator.state_dict(), "{}/atcnet_lstm_{}.pth".format(config.model_dir, epoch + 1))
        print("total time: {} second".format(time.time() - t00))

    def _reset_gradients(self):
        self.generator.zero_grad()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda1", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--dataset_dir", type=str, default="../dataset/")
    parser.add_argument("--model_dir", type=str, default="../model/atcnet/")
    parser.add_argument("--sample_dir", type=str, default="../sample/atcnet/")
    parser.add_argument("--device_ids", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="lrw")
    parser.add_argument("--lstm", type=bool, default=True)
    parser.add_argument("--num_thread", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--pretrained_dir", type=str)
    parser.add_argument("--pretrained_epoch", type=int)
    parser.add_argument("--start_epoch", type=int, default=0, help="start from 0")
    parser.add_argument("--rnn", type=bool, default=True)
    parser.add_argument("--para_dim", type=int, default=53)
    parser.add_argument("--index", type=str, default="80,144", help="index ranges")
    parser.add_argument("--pose", type=int, default=0, help="whether predict pose")
    parser.add_argument("--relativeframe", type=int, default=0, help="whether use relative frame value for pose")
    # for personalized data
    # parser.add_argument("--start", type=int, default=0)
    # parser.add_argument("--trainN", type=int, default=0)
    # parser.add_argument("--testN", type=int, default=0)
    # for continnue train
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="../model/atcnet_pose0/atcnet_lstm_24.pth")
    parser.add_argument("--preserve_mouth", type=bool, default=False)
    # for remove jittering
    parser.add_argument("--smooth_loss", type=bool, default=False)  # smooth in time, similar to total variation
    parser.add_argument("--smooth_loss2", type=bool, default=False)  # smooth in time, for expression
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_smooth2", type=float, default=0.0001)
    # for less constrain for pose
    parser.add_argument("--less_constrain", type=bool, default=False)
    parser.add_argument("--lambda_pose", type=float, default=0.2)
    parser.add_argument("--save_per_epochs", type=int, default=10)

    return parser.parse_args()


def main(config):
    t = Trainer(config)
    t.fit()


if __name__ == "__main__":

    config = parse_args()
    str_ids = config.index.split(",")
    config.indexes = []
    for i in range(int(len(str_ids) / 2)):
        start = int(str_ids[2 * i])
        end = int(str_ids[2 * i + 1])
        if end > start:
            config.indexes += range(start, end)
    # print('indexes', config.indexes)
    print("device", config.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = "train"

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    config.cuda1 = torch.device("cuda:{}".format(config.device_ids))
    main(config)