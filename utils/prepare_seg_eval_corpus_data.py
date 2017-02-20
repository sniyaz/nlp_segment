from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
import os
from collections import defaultdict, Counter

import numpy as np
import pickle

from prepare_seg_eval_data import remove_colons

sys.path.append("../")
from bpe import get_vocabulary

import pdb

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--corpus', '-c', type=argparse.FileType(mode='r', encoding='utf-8', errors='ignore'), default=sys.stdin,
        metavar='PATH',
        help="Input corpus to clean.")
    parser.add_argument(
        '--gold_standard', '-gs', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input gold standard to clean.")
    parser.add_argument(
        '--output', '-o', action="store",
        metavar='PATH',
        help="Output dir")
    
    return parser

        

if __name__ == '__main__':
         
    parser = create_parser()
    args = parser.parse_args()

    clean_corpus = open(os.path.join(args.output, "clean_corpus.txt"), "w+")
    clean_gold_standard = open(os.path.join(args.output, "gold_standard.txt"), "w+")

    args.corpus.seek(0)
    for line in args.corpus:
        clean_line = line.split("\t")
        clean_line = clean_line[1] 
        clean_corpus.write(clean_line)
    
    clean_corpus.seek(0)
    vocab = get_vocabulary(clean_corpus)

    for line in args.gold_standard:
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        if word in vocab:
            word_segs = line_contents[1].split(", ")
            word_segs = [seg.split(" ") for seg in word_segs]
            word_segs = [remove_colons(seg) for seg in word_segs]
            word_segs = [" ".join(seg) for seg in word_segs]
            new_line = ", ".join(word_segs)
            new_line = word + "\t" + new_line + "\n"
            clean_gold_standard.write(new_line)
   
    clean_corpus.close()
    clean_gold_standard.close()
            


    



