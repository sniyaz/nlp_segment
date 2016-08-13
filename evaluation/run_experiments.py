import sys
import os
import codecs
import argparse
import string
import csv
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

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
        help="Target file for evaluation")
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
    num_retries = 1000
    max_merges  = 30000
    granularity = 500

    colors = ["r", "b"]

    for mode in [1]:
        all_results = defaultdict(lambda: [])
        for i in range(num_retries):

            cur_folder_name = "mode_" + str(mode) + "_trial_" + str(i)
            cur_folder_name = os.path.join(args.dir, cur_folder_name)
            os.system("mkdir " + cur_folder_name)
            os.system("python3 ../bpe.py --mode " + str(mode) + " -i " + args.input + " -ft " + \
            " -o " + os.path.join(cur_folder_name, "exp") +  " -s " + str(max_merges))

            num_merges = granularity
            while (num_merges <= max_merges):
                save_folder = os.path.join(cur_folder_name, str(num_merges) + "_merges")
                os.system("mkdir " + save_folder)
                eval_output_file = os.path.join(save_folder, "eval_output.txt")
                os.system("python3 evaluate_seg.py -i " + args.gold_standard + " -ops " + os.path.join(cur_folder_name, "exp_merge_ops.txt") \
                + " -s " + str(num_merges) + " > " + eval_output_file)
                all_results[num_merges].append(extract_fmeasure(eval_output_file))
                num_merges += granularity
        
            #Save out the dict of resuts to checkpoint our progress...
	    ckpt_obj = open(os.path.join(args.dir, "ckpt.txt"), "wb+")
            pickle.dump(dict(all_results), ckpt_obj)
            ckpt_obj.close()



        cur_color = colors[mode-1]
        largest_variance = max(all_results, key = lambda x: max(all_results[x]) - min(all_results[x]))	
        iterations = [i for i in range(len(all_results[largest_variance]))]
	scores = all_results[largest_variance]
	pdb.set_trace()
	plt.plot(iterations, scores, cur_color)
        plt.plot(iterations, scores, cur_color + "o")
        plot_filename = os.path.join(args.dir, "mode_" + str(mode) + "_" + str(largest_variance) + "_merges" + "_plt.png")
        plt.savefig(plot_filename)
        plt.clf()
        
 




            












