import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, resample



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
    # z = z[:-skip_seconds * sample_rate]
    time_range = 300

    # time_range = 100
    # plt.plot(z[180:180+time_range])
    # plt.show()

    # plt.plot(z[150:150+time_range])
    # plt.show()

    # plt.plot(z[300:300+time_range])
    # plt.show()


    # a is indexed by different values of k
    a = [ auto_corr(z, k) for k in range(1, len(z) - 1) ]

    zeta = argrelextrema(np.array(a), np.greater)

    # output for zeta is a 1D tuple, de-tuple zeta
    assert(len(zeta) == 1)
    zeta = zeta[0]

    print(zeta)

    delta_mean = int(np.ceil(np.sum(np.subtract(zeta[1:], zeta[:-1])) / (len(zeta) - 1)))

    print(delta_mean)

    tau = 10 # Parameter for deviation for each gait half cycle

    mu = []

    for zeta_i in zeta:
        # Index window for z to find each local minum
        z_min_window_indices = range(max(zeta_i - tau, 0), min(zeta_i + delta_mean + tau, len(z) - 1))
        z_min_window = [ z[i] for i in z_min_window_indices ]

        mu_i = z_min_window_indices[np.argmin(z_min_window)] # get back the indices of z
        mu.append(mu_i)

    print(mu)

    mu_mean = int(np.ceil(np.sum(np.subtract(mu[1:], mu[:-1])) / (len(mu) - 1)))

    print(mu_mean)

    # Resegment the data
    rho = 40                    # Number of samples resampled in each gait cycle
    Z = []                      # list of lists
    even_indices = list(range(1, len(mu) - 1, 2))
    print(even_indices)
    for i in even_indices:
        Z_i = z[mu[i-1] : mu[i+1]]
        # Resample
        Z_i_resampled = resample(Z_i, rho)
        Z.append(Z_i_resampled)


    # BANDANA
    # Quantization
    # Average of each sample over all gait cycles
    A = [ sum(z_i_list) / rho for z_i_list in zip(*Z) ]
    assert(len(A) == rho)

    # plt.plot(Z)
    # plt.show()

    # plt.plot(A)
    # plt.show()

    # Extract fingerprint
    b = 4                       # Number of bits for each gait cycle
    delta = []
    for Z_i in Z:
        for Z_chunk, A_chunk in chunks(Z_i, A, b):
            delta_i = sum(A_chunk) - sum(Z_chunk)
            delta.append(delta_i)

    assert(len(delta) == len(Z) * b)

    f = [ 1 if delta_i > 0 else 0 for delta_i in delta ]

    assert(len(delta) == len(f))

    print(delta)
    print(f)
    print(sum(delta))

    # Sort f based on reliability (descending order from most reliable)
    delta_ordered, f_ordered = zip(*[ p for p in sorted(zip(delta, f), key=lambda pair: abs(pair[0]), reverse=True) ])

    print(delta_ordered)
    print(f_ordered)

    # Plot!
    plt.subplot(211)
    plt.plot(range(time_range), z[:time_range])

    for mu_i in mu:
        if mu_i < time_range:
            plt.axvline(x=mu_i, color='r')

    plt.subplot(212)
    temp = [ z_i for Z_i in Z for z_i in Z_i ]
    time_range_resampled = int(np.ceil(time_range * rho / sample_rate / 2))
    plt.plot(range(time_range_resampled), temp[:time_range_resampled])

    total = 0
    for Z_i in Z:
        plt.axvline(x=total, color='r')
        total += len(Z_i)
        if total > time_range_resampled:
            break

    plt.savefig('myfilename.png')

    print("Done")

# Chunk Z and A equally
def chunks(Z, A, n):
    assert(len(Z) == len(A))
    chunk_size = len(Z) // n
    print(chunk_size)
    for i in range(0, len(Z), chunk_size):
        yield Z[i:i+chunk_size], A[i:i+chunk_size]

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
