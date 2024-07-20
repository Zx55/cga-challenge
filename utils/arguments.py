import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('-l', '--list-envs', action='store_true')
    parser.add_argument('--gt', action='store_true')
    return parser.parse_args()
