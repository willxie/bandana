import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, resample
from madgwick.madgwickahrs import MadgwickAHRS
from madgwick.madgwickahrs import Quaternion
from trans import rotation_matrix
import csv

input_file = "short_walk.csv" # At 50 Hz
skip_seconds = 1
sample_rate = 50

tau = 10                    # Parameter for deviation for each gait cycle
rho = 40                    # Number of samples resampled in each gait cycle
b = 4                       # Number of bits for each gait cycle

def main():
    # Each entry is (x, y, z) tuple
    accel = []
    rot   = []
    magn  = []

    with open(input_file, 'r') as csvfile:
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
    # accel = [(1,0,0) for i in range(len(accel))]
    # rot   = [(0,0,0) for i in range(len(accel))]
    e = []
    rad = []
    roll = []
    pitch = []
    yaw = []
    mw = MadgwickAHRS(sampleperiod=1/sample_rate, quaternion=Quaternion(1, 0, 0, 0), beta=0.2)
    for rot_tup, accel_tup, magn_tup in zip(rot, accel, magn):
        # mw.update_imu(rot_tup, accel_tup)
        mw.update(rot_tup, accel_tup, magn_tup)
        roll_i, pitch_i, yaw_i = mw.quaternion.to_euler123()
        rad_i, x_i, y_i, z_i = mw.quaternion.to_angle_axis()
        e.append((x_i, y_i, z_i))
        rad.append(rad_i)
        roll.append(roll_i)
        pitch.append(pitch_i)
        yaw.append(yaw_i)
        # print("{}\t{}\t{}".format(roll_i, pitch_i, yaw_i))


    y_orig = [ tup[1] for tup in accel ]
    # z = z[:-skip_seconds * sample_rate]
    time_range = 300

    # time_range = 100
    # plt.plot(z_orig[180:180+time_range])
    # plt.show()

    # for accel_i, r, p, y in zip(accel, roll, pitch, yaw):
    accel_rect = []
    z = []
    for accel_i, rad_i, e_i in zip(accel, rad, e):
        accel_i_vec = (np.array(accel_i))
        print("orig: {}".format(accel_i_vec))
        # accel_i_rect = R_x(r).dot(R_y(p)).dot(R_z(y)).dot(accel_i_vec)
        # accel_i_rect = np.linalg.inv(R_x(r).dot(R_y(p)).dot(R_z(y))).dot(accel_i_vec)
        # accel_i_rect = R_z(y).dot(R_y(p)).dot(R_x(r)).dot(accel_i_vec)
        # accel_i_rect = np.linalg.inv(R_z(y).dot(R_y(p)).dot(R_x(r))).dot(accel_i_vec)

        R = rotation_matrix(rad_i, e_i)[:3, :3]
        _R = R_rad(rad_i, e_i)

        # print(R)
        # print(_R)

        # accel_i_rect = np.linalg.inv(R).dot(accel_i_vec)
        accel_i_rect = R.dot(accel_i_vec)

        # accel_i_rect = np.linalg.inv(R_rad(rad_i, e_i)).dot(accel_i_vec)
        # accel_i_rect = R_rad(rad_i, e_i).dot(accel_i_vec)

        print("rect: {}".format(accel_i_rect))
        accel_rect.append(accel_i_rect)
        z.append(accel_i_rect[2])

    # temp = list(zip(roll, pitch, yaw))
    # plt.plot(temp)
    # plt.show()

    plt.subplot(211)
    plt.plot(accel[:3000])

    plt.subplot(212)
    plt.plot(accel_rect[:3000])
    # plt.show()

    # Pick final channel as input
    z = [ accel_i[2] for accel_i in accel ]
    z = [ accel_i[2] for accel_i in accel_rect ]

    # a is indexed by different values of k
    a = [ auto_corr(z, k) for k in range(1, len(z) - 1) ]

    zeta = argrelextrema(np.array(a), np.greater)

    # output for zeta is a 1D tuple, de-tuple zeta
    assert(len(zeta) == 1)
    zeta = zeta[0]

    print("zeta:")
    print(zeta)

    # Reject outliers to denoise
    m = 2                       # Reject m std outliers
    zeta_filtered = zeta[abs(zeta - np.mean(zeta)) < m * np.std(zeta)]
    delta_mean = int(np.ceil(np.sum(np.subtract(zeta_filtered[1:], zeta_filtered[:-1])) / (len(zeta) - 1)))

    print(np.subtract(zeta[1:], zeta[:-1]))
    print("delta_mean:")
    print(delta_mean)

    mu = []

    for zeta_i in zeta:
        # Index window for z to find each local minum
        z_min_window_indices = range(max(zeta_i - tau, 0), min(zeta_i + delta_mean + tau, len(z) - 1))
        z_min_window = [ z[i] for i in z_min_window_indices ]

        mu_i = z_min_window_indices[np.argmin(z_min_window)] # get back the indices of z
        mu.append(mu_i)

    print("mu:")
    print(mu)

    mu_mean = int(np.ceil(np.sum(np.subtract(mu[1:], mu[:-1])) / (len(mu) - 1)))

    print("mu_mean:")
    print(mu_mean)

    # Resegment the data
    Z = []                      # list of lists
    even_indices = list(range(1, len(mu) - 1, 2))
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

    # Extract fingerprint
    delta = []
    for Z_i in Z:
        for Z_chunk, A_chunk in chunks(Z_i, A, b):
            delta_i = sum(A_chunk) - sum(Z_chunk)
            delta.append(delta_i)

    assert(len(delta) == len(Z) * b)

    f = [ 1 if delta_i > 0 else 0 for delta_i in delta ]

    assert(len(delta) == len(f))

    print("delta:")
    # print(delta)
    print("Finger print:")
    print(f)

    # Sort f based on reliability (descending order from most reliable)
    delta_ordered, f_ordered = zip(*[ p for p in sorted(zip(delta, f), key=lambda pair: abs(pair[0]), reverse=True) ])

    # print(delta_ordered)
    # print(f_ordered)

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

# orig
def R_z(x):
    return np.array([
        [np.cos(x), -np.sin(x), 0],
        [np.sin(x), np.cos(x), 0],
        [0, 0, 1]
    ])

def R_y(x):
    return np.array([
        [np.cos(x), 0, np.sin(x)],
        [0, 1, 0],
        [-np.sin(x), 0, np.cos(x)]
    ])

def R_x(x):
    return np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

def R_rad(r, e):
    assert(len(e) == 3)
    return np.array([
        [np.cos(r) + e[0]**2*(1-np.cos(r)), e[0]*e[1]*(1-np.cos(r))-e[2]*np.sin(r), e[0]*e[2]*(1-np.cos(r))+e[1]*np.sin(r)],
        [e[1]*e[0]*(1-np.cos(r))+e[2]*np.sin(r), np.cos(r)+e[1]**2*(1-np.cos(r)), e[1]*e[2]*(1-np.cos(r)-e[0]*np.sin(r))],
        [e[2]*e[0]*(1-np.cos(r))-e[1]*np.sin(r), e[2]*e[1]*(1-np.cos(r))+e[0]*np.sin(r), np.cos(r)+e[2]**2*(1-np.cos(r))]
    ])

def rotation_matrix2(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M

# Chunk Z and A equally
def chunks(Z, A, n):
    assert(len(Z) == len(A))
    chunk_size = len(Z) // n
    # print(chunk_size)
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
