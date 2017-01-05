import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


import csv
from madgwick_py.madgwickahrs import MadgwickAHRS

input_file = "data.csv" # At 50 Hz
skip_seconds = 5
sample_rate = 50

def main():
    # Each entry is (x, y, z) tuple
    accel = []
    rot   = []
    magn  = []

    with open("data.csv", 'r') as csvfile:
        data_reader = csv.reader(csvfile)

        row_count = 0
        for row in data_reader:
            # Skip first two seconds of data
            if row_count < (skip_seconds * sample_rate):
                row_count += 1
                continue

            if row[0]:
                # Convert all data to floats
                rot.append((float(row[4]), float(row[5]), float(row[6])))
                accel.append((float(row[7]), float(row[8]), float(row[9])))
                magn.append((float(row[13]), float(row[14]), float(row[15])))

    # Madgwick normalization
    # TODO

    z = [ tup[2] for tup in accel ]
    time_range = 300
    plt.plot(range(time_range), z[:time_range])

    plt.savefig('myfilename.png')

    # a is indexed by different values of k
    a = [ auto_corr(z, k) for k in range(1, len(z) - 1) ]

    print(a)

    zeta = argrelextrema(np.array(a), np.greater)

    # output for zeta is a 1D tuple, de-tuple zeta
    assert(len(zeta) == 1)
    zeta = zeta[0]

    print(zeta)

    delta_mean = np.ceil(np.sum(np.subtract(zeta[1:], zeta[:-1])) / (len(zeta) - 1))

    print(delta_mean)

    print("Done")

def auto_corr(z, k):
    sigma = np.var(z)
    n = len(z)
    z1 = z[:-k]
    z2 = z[k:]
    assert(len(z1) == len(z2))
    a = np.vdot(z1, z2) / ((n - k) * pow(sigma, 2))

    return a

if __name__ == '__main__':
    main()
