import os
import glob

import scipy.io
import pandas as pd
import torch
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default="data/mat/sub1_comp.mat",
                        help="Parent directory of .mat files containing train, val, and test sets")
    parser.add_argument('-o', '--out', type=str,
                        default="data/npz/",
                        help="The output npz file containing train, val, and test sets")
    args = parser.parse_args()

    for fn in glob.glob(os.path.join(args.path, "*.mat")):
        # Load the .mat file
        print(fn)
        mat = scipy.io.loadmat(fn)
        data = {}
        for k in ['train_data', 'test_data', 'train_dg']:
            data[k] = np.array(mat[k])
            print(k, ":", data[k].shape)

        np.savez_compressed(os.path.join(args.out, os.path.basename(fn)[:-4]), data)
if __name__ == '__main__':
    main()