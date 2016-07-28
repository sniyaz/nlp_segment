#A script that diffs our results and those from vanilla BPE!

import sys
import codecs
import argparse
import string

import pdb

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
        help="Segmentations text file we produced")
    parser.add_argument(
        '--bpe_segs', '-b', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Segmentations text file from BPE")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w+'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")

    return parser

def seg_from_line(line, seg_dict):
    line_contents = line.split(" ")
    line_contents = list(filter(lambda x: x != '', line_contents))
    key = line_contents[0][:-1]
    seg = line_contents[1:]
    seg_dict[key] = seg


if __name__ == '__main__':

    include_matches = True

    parser = create_parser()
    args = parser.parse_args()

    base_segs = {}
    for line in args.input:
        seg_from_line(line, base_segs)
    
    bpe_segs = {}
    for line in args.bpe_segs:
        seg_from_line(line, bpe_segs)

    to_write = list(base_segs.keys())
    to_write.sort()
        
    for w in to_write:
        if base_segs[w] != bpe_segs[w]:
            args.output.write(w)
            args.output.write(u'\n')
            args.output.write("BASE: " + " ".join(base_segs[w]))
            args.output.write("BPE: " + " ".join(bpe_segs[w]))
            args.output.write(u'\n')
        else:
            if include_matches:
                args.output.write(w)
                args.output.write(u'\n')
                args.output.write("BOTH: " + " ".join(base_segs[w]))
                args.output.write(u'\n')


