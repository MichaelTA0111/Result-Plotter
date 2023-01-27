from enum import Enum
import re

import numpy as np
import matplotlib.pyplot as plt


ALL_PACKET_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
ALL_NUM_PACKETS = [20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000]
ALL_NUM_CONSUMERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Metric(Enum):
    TIME = 'time'
    CPU_UTILISATION = 'cpu_utilisation'


class Variable(Enum):
    PACKET_SIZE = 'packet_size'
    NUM_PACKETS = 'num_packets'
    NUM_CONSUMERS = 'num_consumers'


def generate_filepath(metric, variable, format):
    return f'./graphs/{metric.value}_v_{variable.value}.{format}'


def parse_data(filename):
    times = []
    cpu_utilisations = []
    with open(f'./results/{filename}') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(', ')

            time = data[0]
            # TODO: Parse minutes?
            time = re.findall(r'\d+\.\d+', time)[0]
            time = float(time)
            times.append(time)

            cpu_utilisation = data[1]
            cpu_utilisation = float(cpu_utilisation)
            cpu_utilisations.append(cpu_utilisation)

    return times, cpu_utilisations


def plot(variable):
    # Specify 2 of 3 arguments
    if variable is Variable.PACKET_SIZE:
        xs = np.array(ALL_NUM_PACKETS)
        base_s_filenames = [f'512B__{z:_}P__2C__s.txt' for z in ALL_NUM_PACKETS]
        base_i_filenames = [f'512B__{z:_}P__2C__i.txt' for z in ALL_NUM_PACKETS]
    else:
        raise Exception

    s_times = []
    s_utils = []
    for file in base_s_filenames:
        try:
            times, utils = parse_data(file)

            avg_time = sum(times) / len(times)
            s_times.append(avg_time)

            avg_util = sum(utils) / len(utils)
            s_utils.append(avg_util)
        except FileNotFoundError:
            s_times.append(None)
            s_utils.append(None)

    i_times = []
    i_utils = []
    for file in base_i_filenames:
        try:
            times, utils = parse_data(file)

            avg_time = sum(times) / len(times)
            i_times.append(avg_time)

            avg_util = sum(utils) / len(utils)
            i_utils.append(avg_util)
        except FileNotFoundError:
            i_times.append(None)
            i_utils.append(None)

    s_times = np.array(s_times).astype(np.double)
    s_mask = np.isfinite(s_times)
    i_times = np.array(i_times).astype(np.double)
    i_mask = np.isfinite(i_times)

    plt.plot(xs[s_mask], s_times[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_times[i_mask], linestyle='-', marker='o', label='IPC')
    plt.title('Execution Time')
    plt.xlabel('Packet Count')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig(generate_filepath(Metric.TIME, variable.NUM_PACKETS, 'png'), format='png')
    plt.savefig(generate_filepath(Metric.TIME, variable.NUM_PACKETS, 'svg'), format='svg')
    plt.show()

    s_utls = np.array(s_utils).astype(np.double)
    s_mask = np.isfinite(s_utls)
    i_utls = np.array(i_utils).astype(np.double)
    i_mask = np.isfinite(i_utls)

    plt.plot(xs[s_mask], s_utls[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_utls[i_mask], linestyle='-', marker='o', label='IPC')
    plt.title('CPU Utilisation')
    plt.xlabel('Packet Count')
    plt.ylabel('CPU Utilisation (%)')
    plt.legend()
    plt.savefig(generate_filepath(Metric.CPU_UTILISATION, variable.NUM_PACKETS, 'png'), format='png')
    plt.savefig(generate_filepath(Metric.CPU_UTILISATION, variable.NUM_PACKETS, 'svg'), format='svg')
    plt.show()


if __name__ == '__main__':
    plot(Variable.PACKET_SIZE)
