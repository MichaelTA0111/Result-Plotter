from enum import Enum

import numpy as np
import matplotlib.pyplot as plt


ALL_PACKET_SIZES = [128, 256, 512]
ALL_NUM_PACKETS = [20_000, 40_000, 60_000, 80_000, 100_000]  # , 120_000, 140_000, 160_000, 180_000, 200_000
ALL_NUM_CONSUMERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Variable(Enum):
    PACKET_SIZE = 'total_cpu_v_packet_size'
    NUM_PACKETS = 'total_cpu_v_num_packets'
    NUM_CONSUMERS = 'total_cpu_v_num_consumers'


def parse_utls(filename):
    utls = []
    with open(f'./results/cpu/{filename}') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            try:
                utl = float(line)
                utls.append(utl)
            except ValueError:
                continue

    return utls


def plot(vary):
    # Specify 2 of 3 arguments
    if vary is Variable.PACKET_SIZE:
        xs = np.array(ALL_NUM_PACKETS)
        base_s_filenames = [f'128B__{z:_}P__2C__s.txt' for z in ALL_NUM_PACKETS]
        base_i_filenames = [f'128B__{z:_}P__2C__i.txt' for z in ALL_NUM_PACKETS]
    else:
        raise Exception

    s_utls = []
    for file in base_s_filenames:
        try:
            utls = parse_utls(file)[1:-1]
            avg_utl = sum(utls) / len(utls)
            s_utls.append(avg_utl)
        except FileNotFoundError as e:
            s_utls.append(None)
            # print(e)

    i_utls = []
    for file in base_i_filenames:
        try:
            utls = parse_utls(file)[1:-1]
            avg_utl = sum(utls) / len(utls)
            i_utls.append(avg_utl)
        except FileNotFoundError as e:
            i_utls.append(None)
            # print(e)

    s_utls = np.array(s_utls).astype(np.double)
    s_mask = np.isfinite(s_utls)
    i_utls = np.array(i_utls).astype(np.double)
    i_mask = np.isfinite(i_utls)

    plt.plot(xs[s_mask], s_utls[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_utls[i_mask], linestyle='-', marker='o', label='IPC')
    plt.legend()
    plt.savefig(f'./graphs/{vary.value}.png', format='png')
    plt.savefig(f'./graphs/{vary.value}.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    plot(Variable.PACKET_SIZE)
