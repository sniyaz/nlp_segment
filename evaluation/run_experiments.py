import sys
import os
import codecs
import argparse
import string
import csv
import matplotlib.pyplot as plt

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

def extract_fmeasure(result_filename):
    result_obj = open(result_filename, "r")
    for line in result_obj:
        line = line.strip()
        line = line.split("  ")
        if line[0] == "F-measure:":
            return float(line[1][:-1])


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    num_steps = 10
    granularity = 200

    colors = ["r", "b"]

    for mode in [1, 2]:
        scores = []
        iterations = []
        for i in range(1, num_steps+1):
            num_iterations = i*granularity
            iterations.append(num_iterations)

            cur_folder_name = "mode_" + str(mode) + "_" + str(num_iterations)
            cur_folder_name = os.path.join(args.dir, cur_folder_name)
            os.system("mkdir " + cur_folder_name)
            os.system("python3 ../segment.py --mode " + str(mode) + " -i " + args.input + " -ft " + \
            " -o " + os.path.join(cur_folder_name, "exp") +  " -s " + str(num_iterations))

            eval_output_file = os.path.join(cur_folder_name, "eval_output.txt")
            os.system("python3 evaluate_seg.py -i " + args.gold_standard + " -ops " + os.path.join(cur_folder_name, "exp_merge_ops.txt") \
            + " > " + eval_output_file)
            scores.append(extract_fmeasure(eval_output_file))


        cur_color = colors[mode-1]
        plt.plot(iterations, scores, cur_color)
        plt.plot(iterations, scores, cur_color + "o")
        plot_filename = os.path.join(args.dir, "mode_" + str(mode) + "_plt.png")
        plt.savefig(plot_filename)
        plt.clf()
        
 




            












