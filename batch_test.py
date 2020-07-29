import os, sys, argparse, subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    ### model options
    # parser.add_argument('--method',  type=str,   required=True,      help='full model name')
    parser.add_argument('--name', type=str, default="TDMS0728", help='full model name')
    parser.add_argument('--which_epoch', type=int, default=65, help='epoch to test')
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--reverse', action="store_true", help='reverse task list')
    parser.add_argument('--data_dir', type=str, default="/mnt/disk2/liziyuan/zhouyifeng/datasets/videoTemporalConsis", help='data root dir')
    parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints/TDMS_Net", help='checkpoints dir')

    opts = parser.parse_args()

    filename = "./lists/test_tasks.txt"
    dataset_task_list = []
    with open(filename) as f:
        for line in f.readlines():
            if line[0] != "#":
                dataset_task_list.append(line.rstrip().split())

    if opts.reverse:
        dataset_task_list.reverse()

    for i in range(len(dataset_task_list)):
        dataset = dataset_task_list[i][0]
        task = dataset_task_list[i][1]

        cmd = "CUDA_VISIBLE_DEVICES=%d python -W ignore test_TDMS_Net.py \
            --name %s \
            --data_dir %s \
            --model TDMSNet\
            --phase test \
            --dataset %s \
            --task %s \
            --which_epoch %d \
            --checkpoints_dir %s" % (
        opts.gpu, opts.name, opts.data_dir, dataset, task, opts.which_epoch, opts.checkpoints_dir)

        print(cmd)
        subprocess.call(cmd, shell=True)
