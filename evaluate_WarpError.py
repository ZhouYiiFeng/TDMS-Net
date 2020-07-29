#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from options.test_options import TestOptions
import util.util as util

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if __name__ == "__main__":

    opts = TestOptions().parse()
    opts.cuda = True
    print(opts)
    if opts.which_epoch == 'latest':
        epoch_st = util.findLatestEpoch(opts)
    else:
        epoch_st = int(opts.which_epoch)
    output_dir = os.path.join(opts.save_dir, opts.phase, opts.name, "epoch_%d" % epoch_st, opts.task, opts.dataset)

    ## print average if result already exists
    if opts.testName == "TDMS":
        # metric_filename = os.path.join(output_dir, "proc_WarpError.txt")
        metric_filename = "./results/Our/TDMS0728/Sf_%s_%s_%s.txt" % (
            opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    elif opts.testName =="ECCV":
        metric_filename = "./results/ECCV/ECCV_%s_%s_%s.txt" % (opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    elif opts.testName == "Sig":
        metric_filename = "./results/SIG/SIG_%s_%s_%s.txt" % (opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    elif opts.testName == "Pro":
        metric_filename = "./results/Pro/Pro_%s_%s_%s.txt" % (opts.task.split('/')[0], opts.task.split('/')[1], opts.dataset)
    else:
        assert "testName unknown Error"

    if not os.path.isdir("./results/SIG"):
        os.makedirs("./results/SIG/")
        os.makedirs("./results/Pro/")
        os.makedirs("./results/ECCV/")
    if not os.path.isdir("./results/Our/TDMS0728/"):
        os.makedirs("./results/Our/TDMS0728/")

    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists, skip..." % metric_filename)

        cmd = 'tail -n1 %s' % metric_filename
        util.run_cmd(cmd)
        sys.exit()

    ## flow warping layer
    device = torch.device("cuda" if opts.cuda else "cpu")
    flow_warping = util.Resample2d().to(device)

    ### load video list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" % (opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    ### start evaluation
    err_all = np.zeros(len(video_list))

    for v in range(len(video_list)):

        video = video_list[v]

        frame_dir = os.path.join(opts.save_dir, opts.phase, opts.name, "epoch_%d" % epoch_st, opts.task, opts.dataset, video)
        # frame_dir = os.path.join("/mnt/disk1/zhouyifeng/datasets/learingBlindVideoTemporal/test/ECCV18_release/WCT/wave/DAVIS", video)
        # occ_dir = os.path.join(opts.data_dir, opts.phase, "fw_occlusion", opts.dataset, video)
        occ_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/flow", "fw_occlusion", opts.dataset, video)
        # flow_dir = os.path.join(opts.data_dir, opts.phase, "fw_flow", opts.dataset, video)
        flow_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/flow", "fw_flow", opts.dataset, video)

        if opts.testName == "ECCV":
            frame_dir = os.path.join(
                "/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/ECCV18_release", opts.task,
                opts.dataset, video) #"/mnt/datadev_2/std-1/zhouyifeng/datasets/videoTemporalConsistancy/test/ECCV18_release"
        elif opts.testName == "Sig":
            frame_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/SigAsia15",
                                      opts.task, opts.dataset, video)
        elif opts.testName == "Pro":
            frame_dir = os.path.join("/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis/test/processed", opts.task, opts.dataset, video)

        frame_list = glob.glob(os.path.join(frame_dir, "*.jpg"))

        err = 0
        for t in range(1, len(frame_list)):

            ### load input images
            filename = os.path.join(frame_dir, "%05d.jpg" % (t - 1))
            img1 = util.read_img(filename)
            filename = os.path.join(frame_dir, "%05d.jpg" % (t))
            img2 = util.read_img(filename)

            print("Evaluate Warping Error on %s-%s: video %d / %d, %s" % (
            opts.dataset, opts.phase, v + 1, len(video_list), filename))

            ### load flow
            filename = os.path.join(flow_dir, "%05d.flo" % (t - 1))
            flow = util.read_flo(filename)

            ### load occlusion mask
            filename = os.path.join(occ_dir, "%05d.png" % (t - 1))
            occ_mask = util.read_img(filename)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = util.img2tensor(img2).to(device)
                flow = util.img2tensor(flow).to(device)

                ## warp img2
                warp_img2 = flow_warping(img2, flow)

                ## convert to numpy array
                warp_img2 = util.tensor2img(warp_img2)

            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)

            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err += np.sum(np.square(diff)) / N

        err_all[v] = err / (len(frame_list) - 1)
        # err_all[v] = err


    print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    err_all = np.append(err_all, err_all.mean())

    print("Save %s" % metric_filename)
    np.savetxt(metric_filename, err_all, fmt="%f")

