from enum import Enum
import ast

import numpy as np
import matplotlib.pyplot as plt


ALL_PACKET_SIZES = [128, 256, 512]
ALL_NUM_PACKETS = [20_000, 40_000, 60_000, 80_000, 100_000]  # , 120_000, 140_000, 160_000, 180_000, 200_000
ALL_NUM_CONSUMERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Variable(Enum):
    PACKET_SIZE = 'per_core_cpu_v_packet_size'
    NUM_PACKETS = 'per_core_cpu_v_num_packets'
    NUM_CONSUMERS = 'per_core_cpu_v_num_consumers'


def parse_utls(filename):
    utls = {0: [], 1: [], 2: [], 3: []}
    with open(f'./results/cpu/{filename}') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '[' in line:
                try:
                    utl = ast.literal_eval(line)
                    utls[0].append(utl[0])
                    utls[1].append(utl[1])
                    utls[2].append(utl[2])
                    utls[3].append(utl[3])
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

    s_utls = {0: [], 1: [], 2: [], 3: []}
    for file in base_s_filenames:
        try:
            utls = parse_utls(file)
            for k, v in utls.items():
                my_utls = v[1:-1]
                avg_utl = sum(my_utls) / len(my_utls)
                s_utls[k].append(avg_utl)
        except FileNotFoundError as e:
            for i in range(4):
                s_utls[i].append(None)
            # print(e)

    i_utls = {0: [], 1: [], 2: [], 3: []}
    for file in base_i_filenames:
        try:
            utls = parse_utls(file)
            for k, v in utls.items():
                my_utls = v[1:-1]
                avg_utl = sum(my_utls) / len(my_utls)
                i_utls[k].append(avg_utl)
        except FileNotFoundError as e:
            for i in range(4):
                i_utls[i].append(None)

    s_utls[0] = np.array(s_utls[0]).astype(np.double)
    s_mask_0 = np.isfinite(s_utls[0])
    s_utls[1] = np.array(s_utls[1]).astype(np.double)
    s_mask_1 = np.isfinite(s_utls[1])
    s_utls[2] = np.array(s_utls[2]).astype(np.double)
    s_mask_2 = np.isfinite(s_utls[2])
    s_utls[3] = np.array(s_utls[3]).astype(np.double)
    s_mask_3 = np.isfinite(s_utls[3])
    i_utls[0] = np.array(i_utls[0]).astype(np.double)
    i_mask_0 = np.isfinite(i_utls[0])
    i_utls[1] = np.array(i_utls[1]).astype(np.double)
    i_mask_1 = np.isfinite(i_utls[1])
    i_utls[2] = np.array(i_utls[2]).astype(np.double)
    i_mask_2 = np.isfinite(i_utls[2])
    i_utls[3] = np.array(i_utls[3]).astype(np.double)
    i_mask_3 = np.isfinite(i_utls[3])

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.set_title('CHERI')
    ax2.set_title('IPC')
    ax1.plot(xs[s_mask_0], s_utls[0][s_mask_0], linestyle='-', marker='o', label='Core 0')
    ax1.plot(xs[s_mask_1], s_utls[1][s_mask_1], linestyle='-', marker='o', label='Core 1')
    ax1.plot(xs[s_mask_2], s_utls[2][s_mask_2], linestyle='-', marker='o', label='Core 2')
    ax1.plot(xs[s_mask_3], s_utls[3][s_mask_3], linestyle='-', marker='o', label='Core 3')
    ax1.legend()
    ax2.plot(xs[i_mask_0], i_utls[0][i_mask_0], linestyle='-', marker='o', label='Core 0')
    ax2.plot(xs[i_mask_1], i_utls[1][i_mask_1], linestyle='-', marker='o', label='Core 1')
    ax2.plot(xs[i_mask_2], i_utls[2][i_mask_2], linestyle='-', marker='o', label='Core 2')
    ax2.plot(xs[i_mask_3], i_utls[3][i_mask_3], linestyle='-', marker='o', label='Core 3')
    ax2.legend()
    plt.tight_layout(h_pad=1)
    plt.savefig(f'./graphs/{vary.value}.png', format='png')
    plt.savefig(f'./graphs/{vary.value}.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    plot(Variable.PACKET_SIZE)
