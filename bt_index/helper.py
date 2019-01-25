import argparse
import sys


def get_args_parser():
    from . import __version__
    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('Index options',
                                       'config index input, vector size, etc.')
    group1.add_argument('-binary_file', type=argparse.FileType('rb'), required=True,
                        help='the path of the binary file to be indexed')
    group1.add_argument('-bytes_per_vector', type=int, required=True,
                        help='number of bytes per vector')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on additional logging for debug')
    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser


def get_run_args(parser_fn=get_args_parser, printed=True):
    args = parser_fn().parse_args()
    if printed:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args
