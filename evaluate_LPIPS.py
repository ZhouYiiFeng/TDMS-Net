#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch

### custom lib
from options.test_options import TestOptions
import util.util as util

# os.environ['CUDA_VISIBLE_opts.DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if __name__ == "__main__":

    opts = TestOptions().parse()
    opts.net = "squeeze"
    opts.cuda = True
    print(opts)
    if opts.which_epoch == 'latest':
        epoch_st = util.findLatestEpoch(opts)
    else:
        epoch_st = int(opts.which_epoch)
    output_dir = os.path.join(opts.save_dir, opts.phase, opts.name, "epoch_%d" % epoch_st, opts.task, opts.dataset)

    ## print average if result already exists
    if opts.testName == "TDMS":
        # metric_filename = os.path.join(output_dir, "LPIPS.txt")
        metric_filename = "./results/Our/TDMSNet/LP_Sf_%s_%s_%s.txt" % (
        opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    elif opts.testName == "ECCV":
        metric_filename = "./results/ECCV/LP_ECCV_%s_%s_%s.txt" % (
        opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    elif opts.testName == "Sig":
        metric_filename = "./results/SIG/LP_SIG_%s_%s_%s.txt" % (
        opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    elif opts.testName == "Pro":
        metric_filename = "./results/Pro/LP_Pro_%s_%s_%s.txt" % (
        opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    else:
        assert "testName unknown Error"

    if not os.path.isdir("./results/SIG"):
        os.makedirs("./results/SIG/")
        os.makedirs("./results/Pro/")
        os.makedirs("./results/ECCV/")
    if not os.path.isdir("./results/Our/TDMSNet/"):
        os.makedirs("./results/Our/TDMSNet/")


    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists, skip..." % metric_filename)

        cmd = 'tail -n1 %s' % metric_filename
        util.run_cmd(cmd)
        sys.exit()


    ## import LPIPS
    sys.path.append('./LPIPS_models')
    from LPIPS_models import dist_model as dm

    ## Initializing LPIPS model
    print("Initialize Distance model from %s" % opts.net)
    model = dm.DistModel()
    print(os.path.abspath('.'))
    model.initialize(model='net-lin', net=opts.net, use_gpu=True, version='0.1')

    ### load video list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" % (opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    ### start evaluation
    dist_all = np.zeros(len(video_list))
    with torch.no_grad():
        for v in range(len(video_list)):

            video = video_list[v]
            #
            # input_dir = os.path.join(opts.data_dir, opts.phase, "input", opts.dataset, video)
            process_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/processed",
                                         opts.task, opts.dataset, video)
            # output_dir = os.path.join(opts.save_dir, opts.phase, opts.name, "epoch_%d" % epoch_st, opts.task,
            #                           opts.dataset, video)
            # if opts.testName == "ECCV":
            #     output_dir = os.path.join(
            #         "/mnt/datadev_2/std-1/zhouyifeng/datasets/videoTemporalConsistancy/test/ECCV18_release", opts.task,
            #         opts.dataset, video)
            # elif opts.testName == "Sig":
            #     output_dir = os.path.join("/mnt/datadev_2/std-1/zhouyifeng/datasets/videoTemporalConsistancy/test/SigAsia15",
            #                               opts.task, opts.dataset, video)
            # elif opts.testName == "Pro":
            #     output_dir = os.path.join("/mnt/datadev_2/std-1/zhouyifeng/datasets/videoTemporalConsistancy/test/processed",
            #                               opts.task, opts.dataset, video)
            #
            # frame_list = glob.glob(os.path.join(process_dir, "*.jpg"))


            frame_dir = os.path.join(opts.save_dir, opts.phase, opts.name, "epoch_%d" % epoch_st, opts.task,
                                     opts.dataset, video)
            # frame_dir = os.path.join("/mnt/disk1/zhouyifeng/datasets/learingBlindVideoTemporal/test/ECCV18_release/WCT/wave/DAVIS", video)
            # occ_dir = os.path.join(opts.data_dir, opts.phase, "fw_occlusion", opts.dataset, video)
            occ_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/flow",
                                   "fw_occlusion", opts.dataset, video)
            # flow_dir = os.path.join(opts.data_dir, opts.phase, "fw_flow", opts.dataset, video)
            flow_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/flow", "fw_flow",
                                    opts.dataset, video)

            if opts.testName == "ECCV":
                frame_dir = os.path.join(
                    "/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/ECCV18_release", opts.task,
                    opts.dataset,
                    video)  # "/mnt/datadev_2/std-1/zhouyifeng/datasets/videoTemporalConsistancy/test/ECCV18_release"
            elif opts.testName == "Sig":
                frame_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/SigAsia15",
                                         opts.task, opts.dataset, video)
            elif opts.testName == "Pro":
                frame_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/processed",
                                         opts.task, opts.dataset, video)

            frame_list = glob.glob(os.path.join(frame_dir, "*.jpg"))

            dist = 0
            for t in range(1, len(frame_list)):
                ### load processed images
                filename = os.path.join(process_dir, "%05d.jpg" % (t))
                P = util.read_img(filename)

                ### load output images
                filename = os.path.join(frame_dir, "%05d.jpg" % (t))
                O = util.read_img(filename)

                print("Evaluate LPIPS on %s-%s: video %d / %d, %s" % (
                opts.dataset, opts.phase, v + 1, len(video_list), filename))

                ### convert to tensor
                P = util.img2tensor(P)
                O = util.img2tensor(O)

                ### scale to [-1, 1]
                P = P * 2.0 - 1
                O = O * 2.0 - 1

                dist += model.forward(P, O).item()

            dist_all[v] = dist / (len(frame_list) - 1)

    print("\nAverage perceptual distance = %f\n" % (dist_all.mean()))

    dist_all = np.append(dist_all, dist_all.mean())


    print("Save %s" % metric_filename)
    np.savetxt(metric_filename, dist_all, fmt="%f")
