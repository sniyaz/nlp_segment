from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict

import re

# hack for python2/3 compatibility
from io import open
argparse.open = open

# python 2/3 compatibility
if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w+'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    for line in args.input:

        letters_only = re.sub("[^a-zA-Z]", " ", line)
        lower_case = letters_only.lower()
        lower_case = lower_case.split()
        new_sentence = [w for w in lower_case if len(w) > 1 and " " not in w]
        args.output.write(" ".join(new_sentence))
        args.output.write('\n')

