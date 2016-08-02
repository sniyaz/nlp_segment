import sys
import os
import codecs
import argparse
import string
import csv
import matplotlib.pyplot as plt
from run_experiments import extract_fmeasure
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
    power_range = range(-2, 3)
    num_iterations = 6000

    seen_powers = []
    scores = []
    for power in power_range:
        gamma = pow(10, power) 
        seen_powers.append(power)
        cur_folder_name = "pow_" + str(power)
        cur_folder_name = os.path.join(args.dir, cur_folder_name)
        os.system("mkdir " + cur_folder_name)
        os.system("python3 ../segment.py --mode 3 -i " + args.input + " -ft " + \
        " -o " + os.path.join(cur_folder_name, "exp") +  " -s " + str(num_iterations) + " -g " + str(gamma))

        eval_output_file = os.path.join(cur_folder_name, "eval_output.txt")
        os.system("python3 evaluate_seg.py -i " + args.gold_standard + " -ops " + os.path.join(cur_folder_name, "exp_merge_ops.txt") \
        + " > " + eval_output_file)
        scores.append(extract_fmeasure(eval_output_file))


    plt.plot(seen_powers, scores, "b")
    plt.plot(seen_powers, scores, "bo")
    plot_filename = os.path.join(args.dir, "tuning_plt.png")
    plt.savefig(plot_filename)        
 




            












