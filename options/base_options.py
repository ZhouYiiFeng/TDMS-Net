import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TDMSNet")
        self.initialized = False

    def initialize(self):

        # data
        self.parser.add_argument('--data_dir', type=str, default='./data', help='path to data folder')
        self.parser.add_argument('--dataset', type=str, default='DAVIS', help='name of the dataset')
        self.parser.add_argument('--list_dir', type=str, default='./lists', help='path to lists folder')
        self.parser.add_argument('--datasets_tasks', type=str, default='W3_D1_C1_I1', help='dataset-task pairs list')
        self.parser.add_argument('--crop_size', type=int, default=192, help='patch size')
        self.parser.add_argument('--geometry_aug', type=int, default=1,help='geometry augmentation (rotation, scaling, flipping)')
        self.parser.add_argument('--order_aug', type=int, default=1, help='temporal ordering augmentation')
        self.parser.add_argument('--scale_min', type=float, default=0.5, help='min scaling factor')
        self.parser.add_argument('--scale_max', type=float, default=2.0, help='max scaling factor')
        self.parser.add_argument('--num_workers', type=int, default=8, help='number of threads for data loader to use')
        self.parser.add_argument('--suffix', type=str, default='', help='name suffix')
        self.parser.add_argument('--batch_size', type=int, default=2, help='training batch size')

        # model
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='path to checkpoint folder')
        self.parser.add_argument('--model', type=str, default='TransformNet', help='chooses which model to use. vid2vid, test')
        self.parser.add_argument('--blocks', type=int, default=5, help='#ResBlocks')
        self.parser.add_argument('--blocks_w', type=int, default=2, help='#ResBlocks for weight')
        self.parser.add_argument('--blocks_f', type=int, default=3, help='#ResBlocks for weight')

        # others
        self.parser.add_argument('--name', type=str, default='experiment_name',help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--alpha', type=float, default=50.0, help='alpha for computing visibility mask')
        self.parser.add_argument('--seed', type=int, default=9487, help='random seed to use')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--cpu', action='store_true', help='use cpu?')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.initialized = True

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        if self.opt.model == 'none':
            self.opt.model = "%s_B%d_nf%d_%s" % (self.opt.model, self.opt.blocks, self.opt.nf, self.opt.norm)
            self.opt.model = "%s_T%d_%s_pw%d_%sLoss_a%s_wST%s_wHT%s_wVGG%s_L%s_%s_lr%s_off%d_step%d_drop%s_min%s_es%d_bs%d" \
                              % (self.opt.model, self.opt.sample_frames, \
                                 self.opt.datasets_tasks, self.opt.crop_size, self.opt.loss, str(self.opt.alpha), \
                                 str(self.opt.w_ST), str(self.opt.w_LT), str(self.opt.w_VGG), self.opt.VGGLayers, \
                                 self.opt.solver, str(self.opt.lr_init), self.opt.lr_offset, self.opt.lr_step, str(self.opt.lr_drop),
                                 str(self.opt.lr_min), \
                                 self.opt.train_epoch_size, self.opt.batch_size)

        return self.opt