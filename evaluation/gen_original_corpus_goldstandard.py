#A script that makes a gold standard set for words that were only in the original corpus...

import sys
import codecs
import argparse
import string
from collections import defaultdict, Counter

import pdb


# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="File of goldstandard segmentations")
    parser.add_argument("--freq_table", "-ft", type=argparse.FileType('r'),
        default=sys.stdin, metavar='PATH',
        help="Original word list, which inluding restricts scoring to words in that word list.")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w+'),
        metavar='PATH',
        help="Output name")

    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

   
    test_words = []
    for line in args.freq_table:
        line = line.strip()
        line_parts = line.split()
        word = line_parts[1]
        test_words.append(word)

    gold_standard = {}
    for line in args.input:
        original_line = line
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        #Including the freq table means that only score segs for words in the freq table.
        if word in test_words:
            args.output.write(original_line)