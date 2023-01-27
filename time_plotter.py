from enum import Enum
import re

import numpy as np
import matplotlib.pyplot as plt


ALL_PACKET_SIZES = [128, 256, 512]
ALL_NUM_PACKETS = [20_000, 40_000, 60_000, 80_000, 100_000]  # , 120_000, 140_000, 160_000, 180_000, 200_000
ALL_NUM_CONSUMERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Variable(Enum):
    PACKET_SIZE = 'time_v_packet_size'
    NUM_PACKETS = 'time_v_num_packets'
    NUM_CONSUMERS = 'time_v_num_consumers'


def parse_times(filename):
    times = []
    with open(f'./results/time/{filename}') as f:
        lines = f.readlines()
        for line in lines:
            if 'real' in line:
                time = re.findall(r'\d+\.\d+', line)[0]
                time = float(time)
                times.append(time)

    return times


def plot(vary):
    # Specify 2 of 3 arguments
    if vary is Variable.PACKET_SIZE:
        xs = np.array(ALL_NUM_PACKETS)
        base_s_filenames = [f'128B__{z:_}P__2C__s.txt' for z in ALL_NUM_PACKETS]
        base_i_filenames = [f'128B__{z:_}P__2C__i.txt' for z in ALL_NUM_PACKETS]
    else:
        raise Exception

    s_times = []
    for file in base_s_filenames:
        try:
            times = parse_times(file)
            avg_time = sum(times) / len(times)
            s_times.append(avg_time)
        except FileNotFoundError as e:
            s_times.append(None)
            # print(e)

    i_times = []
    for file in base_i_filenames:
        try:
            times = parse_times(file)
            avg_time = sum(times) / len(times)
            i_times.append(avg_time)
        except FileNotFoundError as e:
            i_times.append(None)
            # print(e)

    s_times = np.array(s_times).astype(np.double)
    s_mask = np.isfinite(s_times)
    i_times = np.array(i_times).astype(np.double)
    i_mask = np.isfinite(i_times)

    plt.plot(xs[s_mask], s_times[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_times[i_mask], linestyle='-', marker='o', label='IPC')
    plt.legend()
    plt.savefig(f'./graphs/{vary.value}.png', format='png')
    plt.savefig(f'./graphs/{vary.value}.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    plot(Variable.PACKET_SIZE)
