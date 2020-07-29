import os, sys, argparse, subprocess
import util


# from options.test_options import TestOptions

def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    # opts = TestOptions().parse()
    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    ### model options
    parser.add_argument('--name', type=str, default="TDMS0728",  help='full model name')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--which_epoch', type=int, default=65, help='gpu device id')
    # parser.add_argument('--metric', type=str, required=True, choices=["LPIPS", "WarpError"])
    parser.add_argument('--data_dir', type=str, default="/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis")
    # parser.add_argument('--redo', action="store_true", help='redo evaluation')

    opts = parser.parse_args()
    print(opts)

    filename = "lists/test_tasks.txt"
    with open(filename) as f:
        dataset_task_list = []
        for line in f.readlines():
            if line[0] != "#":
                dataset_task_list.append(line.rstrip().split())

    for i in range(len(dataset_task_list)):

        dataset = dataset_task_list[i][0]
        task = dataset_task_list[i][1]
        # for testMode in ["BiFVS", "ECCV", "Sig", "Pro"]:
        for testMode in ["BiFVS"]:
            # filename = '../../data/test/%s/%s/%s/%s.txt' % (opts.name, task, dataset, opts.metric)
            #
            # if not os.path.exists(filename) or opts.redo:
            for metric in ["WarpError", "LPIPS"]:
                cmd = "CUDA_VISIBLE_DEVICES=0 python evaluate_%s.py\
                --name %s\
                --dataset %s \
                --data_dir %s \
                --model TDMSNet \
                --phase test \
                --task %s \
                --which_epoch %d \
                --testName %s" \
                      % (metric, opts.name, dataset, opts.data_dir, task, opts.which_epoch, testMode)

                # if opts.redo:
                #     cmd += " -redo"

                run_cmd(cmd)
    #
    # print("%s:" % opts.metric)
    # for i in range(len(dataset_task_list)):
    #     dataset = dataset_task_list[i][0]
    #     task = dataset_task_list[i][1]
    #     cmd = "tail -n1 ../../data/test/%s/%s/%s/%s.txt" % (opts.method, task, dataset, opts.metric)
    #     subprocess.call(cmd, shell=True)
