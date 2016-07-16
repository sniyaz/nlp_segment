#A script that takes rules from BPE and writes the resuting segmentations
# a manner comparable to our algrithm so we can compare!

from segment import get_vocabulary
import sys
sys.path.insert(0, '../subword-nmt-master/')
from apply_bpe import BPE

import codecs
import argparse
import string

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w+'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    
    return parser
    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    bpe = BPE(args.codes, " ")
    vocab = get_vocabulary(args.input)

    to_write = list(vocab.keys())
    to_write.sort()
    for w in to_write:
        args.output.write(w + ": " + bpe.segment(w))
        args.output.write(u'\n')


