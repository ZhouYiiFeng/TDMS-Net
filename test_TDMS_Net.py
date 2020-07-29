#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
import networks
from options.test_options import TestOptions
import util.util as util


# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if __name__ == "__main__":

    opts = TestOptions().parse()
    opts.cuda = True
    opts.size_multiplier = 2 ** 2  
    opts.device = torch.device("cuda" if opts.cuda else "cpu")
    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")

    ### load model opts
    # opts_filename = os.path.join(opts.checkpoints_dir, "opts.pth")
    # print("Load %s" % opts_filename)
    # with open(opts_filename, 'rb') as f:
    #     model_opts = pickle.load(f)

    from networks.flownet import FlowNet

    FlowNet = FlowNet()
    FlowNet.initialize(opts)
    FlowNet = FlowNet.to(opts.device)
    FlowNet.eval()

    ### initialize model
    # print('===> Initializing model from %s...' % model_opts.model)
    model = networks.__dict__[opts.model](opts, nc_in=12, nc_out=3)

    ### load trained model
    if opts.which_epoch == 'latest':
        epoch_st = util.findLatestEpoch(opts)
    else:
        epoch_st = int(opts.which_epoch)
    opts.model_dir = opts.checkpoints_dir
    model = util.load_model(model=model, dirc="pf", opts=opts, epoch=epoch_st)

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)

    model.eval()

    ### load video list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" % (opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    ### start testing
    for v in range(len(video_list)):

        video = video_list[v]

        print("Test %s on %s-%s video %d/%d: %s" % (opts.task, opts.dataset, opts.phase, v + 1, len(video_list), video))

        ## setup path
        input_dir = os.path.join(opts.data_dir, opts.phase, "input", opts.dataset, video)
        process_dir = os.path.join(opts.data_dir, opts.phase, "processed", opts.task, opts.dataset, video)
        # process_dir = os.path.join(opts.data_dir, opts.phase, "processed", "ECCV16_colorization/DAVIS-gray", video)
        output_dir = os.path.join(opts.save_dir, opts.phase, opts.name, "epoch_%d" % epoch_st, opts.task, opts.dataset, video)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = glob.glob(os.path.join(process_dir, "*.jpg"))
        output_list = glob.glob(os.path.join(output_dir, "*.jpg"))

        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue

        ## frame 0
        frame_p1 = util.read_img(os.path.join(process_dir, "00000.jpg"))
        output_filename = os.path.join(output_dir, "00000.jpg")
        util.save_img(frame_p1, output_filename)

        for t in range(1, len(frame_list)):
            ### load frames
            frame_i1 = util.read_img(os.path.join(input_dir, "%05d.jpg" % (t - 1)))
            frame_i2 = util.read_img(os.path.join(input_dir, "%05d.jpg" % (t)))
            frame_o1 = util.read_img(os.path.join(output_dir, "%05d.jpg" % (t - 1)))
            frame_p2 = util.read_img(os.path.join(process_dir, "%05d.jpg" % (t)))

            ### resize image
            H_orig = frame_o1.shape[0]
            W_orig = frame_o1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / opts.size_multiplier) * opts.size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / opts.size_multiplier) * opts.size_multiplier)

            frame_i1 = cv2.resize(frame_i1, (W_sc, H_sc))
            frame_i2 = cv2.resize(frame_i2, (W_sc, H_sc))
            frame_o1 = cv2.resize(frame_o1, (W_sc, H_sc))
            frame_p2 = cv2.resize(frame_p2, (W_sc, H_sc))

            with torch.no_grad():
                ### convert to tensor
                frame_i1 = util.img2tensor(frame_i1).to(device)
                frame_i2 = util.img2tensor(frame_i2).to(device)
                frame_o1 = util.img2tensor(frame_o1).to(device)
                frame_p2 = util.img2tensor(frame_p2).to(device)

                ### model input
                inputs = torch.cat((frame_i1, frame_i2, frame_o1, frame_p2), dim=1)

                ### forward
                ts = time.time()

                model(inputs)
                output = model.computeLoss(FlowNet)
                # model.saveInnerInfo()
                te = time.time()
                times.append(te - ts)

            ### convert to numpy array
            frame_o2 = util.tensor2img(output)

            ### resize to original size
            frame_o2 = cv2.resize(frame_o2, (W_orig, H_orig))

            ### save output frame
            output_filename = os.path.join(output_dir, "%05d.jpg" % (t))
            util.save_img(frame_o2, output_filename)

        ## end of frame
    ## end of video

    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" % (time_avg, len(times)))
