import sys
import os
import codecs
import argparse
import string
import csv

import pdb


# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--input', '-i', action="store",
        help="Target file for training")
    parser.add_argument(
        '--gold_standard', '-gs', action="store",
        help="Target file for training")
    parser.add_argument(
        '--dir', '-d', action="store",
        help="output directory")

    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    num_steps = 1
    granularity = 1000

    for i in range(1, num_steps+1):
        num_iterations = i*granularity
        for mode in [1, 2]:
            cur_folder_name = "mode_" + str(mode) + "_" + str(num_iterations)
            cur_folder_name = os.path.join(args.dir, cur_folder_name)
            os.system("mkdir " + cur_folder_name)
            os.system("python3 ../segment.py --mode " + str(mode) + " -i " + args.input + " -ft " + \
            " -o " + os.path.join(cur_folder_name, "exp") +  " -s 1000")

            eval_output_file = os.path.join(cur_folder_name, "eval_output.txt")
            os.system("python3 evaluate_seg.py -i " + args.gold_standard + " -ops " + os.path.join(cur_folder_name, "exp_merge_ops.txt") \
            + " > " + eval_output_file)












