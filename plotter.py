from enum import Enum
import re

import numpy as np
import matplotlib.pyplot as plt


ALL_PACKET_SIZES = [512, 1_024, 2_048, 4_096, 8_192]
ALL_PACKET_COUNTS = [20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000]
ALL_CONSUMER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Metric(Enum):
    TIME = ('time', 'Time (s)')
    PACKET_LATENCY = ('packet_latency', 'Latency (\u03BCs)')
    CPU_UTILISATION = ('cpu_utilisation', 'CPU Utilisation (%)')


class Variable(Enum):
    PACKET_SIZE = ('packet_size', 'Packet Size (bytes)')
    PACKET_COUNT = ('packet_count', 'Packet Count')
    CONSUMER_COUNT = ('consumer_count', 'Consumer Count')


def generate_filepath(metric, variable, format):
    return f'./graphs/{metric.value[0]}_v_{variable.value[0]}.{format}'


def parse_data(filename, variable):
    times = []
    lats = []
    utils = []
    with open(f'./results/{variable.value[0]}/{filename}') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(',')

            time = data[0]
            # TODO: Parse minutes?
            time = re.findall(r'\d+\.\d+', time)[0]
            time = float(time)
            times.append(time)

            lat = data[1]
            lat = float(lat)
            lats.append(lat)

            util = data[3]
            util = float(util)
            utils.append(util)

    return times, lats, utils


def plot(variable):
    # Specify 2 of 3 arguments
    if variable is Variable.PACKET_SIZE:
        xs = np.array(ALL_PACKET_SIZES)
        base_s_filenames = [f'{z:_}B__100_000P__2C__s.txt' for z in ALL_PACKET_SIZES]
        base_i_filenames = [f'{z:_}B__100_000P__2C__i.txt' for z in ALL_PACKET_SIZES]
    elif variable is Variable.PACKET_COUNT:
        xs = np.array(ALL_PACKET_COUNTS)
        base_s_filenames = [f'512B__{z:_}P__2C__s.txt' for z in ALL_PACKET_COUNTS]
        base_i_filenames = [f'512B__{z:_}P__2C__i.txt' for z in ALL_PACKET_COUNTS]
    else:
        raise Exception

    s_times = []
    s_lats = []
    s_utils = []
    for file in base_s_filenames:
        try:
            times, lats, utils = parse_data(file, variable)

            avg_time = sum(times) / len(times)
            s_times.append(avg_time)

            avg_lat = sum(lats) / len(lats)
            s_lats.append(avg_lat)

            avg_util = sum(utils) / len(utils)
            s_utils.append(avg_util)
        except FileNotFoundError:
            s_times.append(None)
            s_lats.append(None)
            s_utils.append(None)

    i_times = []
    i_lats = []
    i_utils = []
    for file in base_i_filenames:
        try:
            times, lats, utils = parse_data(file, variable)

            avg_time = sum(times) / len(times)
            i_times.append(avg_time)

            avg_lat = sum(lats) / len(lats)
            i_lats.append(avg_lat)

            avg_util = sum(utils) / len(utils)
            i_utils.append(avg_util)
        except FileNotFoundError:
            i_times.append(None)
            i_lats.append(None)
            i_utils.append(None)

    s_times = np.array(s_times).astype(np.double)
    s_mask = np.isfinite(s_times)
    i_times = np.array(i_times).astype(np.double)
    i_mask = np.isfinite(i_times)

    plt.plot(xs[s_mask], s_times[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_times[i_mask], linestyle='-', marker='o', label='IPC')
    plt.title('Execution Time')
    plt.xlabel(variable.value[1])
    plt.ylabel(Metric.TIME.value[1])
    plt.legend()
    plt.savefig(generate_filepath(Metric.TIME, variable, 'png'), format='png')
    plt.savefig(generate_filepath(Metric.TIME, variable, 'svg'), format='svg')
    plt.show()

    s_lats = np.array(s_lats).astype(np.double)
    s_mask = np.isfinite(s_lats)
    i_lats = np.array(i_lats).astype(np.double)
    i_mask = np.isfinite(i_lats)

    plt.plot(xs[s_mask], s_lats[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_lats[i_mask], linestyle='-', marker='o', label='IPC')
    plt.title('Packet Latency')
    plt.xlabel(variable.value[1])
    plt.ylabel(Metric.PACKET_LATENCY.value[1])
    plt.legend()
    plt.savefig(generate_filepath(Metric.PACKET_LATENCY, variable, 'png'), format='png')
    plt.savefig(generate_filepath(Metric.PACKET_LATENCY, variable, 'svg'), format='svg')
    plt.show()

    s_utls = np.array(s_utils).astype(np.double)
    s_mask = np.isfinite(s_utls)
    i_utls = np.array(i_utils).astype(np.double)
    i_mask = np.isfinite(i_utls)

    plt.plot(xs[s_mask], s_utls[s_mask], linestyle='-', marker='o', label='CHERI')
    plt.plot(xs[i_mask], i_utls[i_mask], linestyle='-', marker='o', label='IPC')
    plt.title('CPU Utilisation')
    plt.xlabel(variable.value[1])
    plt.ylabel(Metric.CPU_UTILISATION.value[1])
    plt.legend()
    plt.savefig(generate_filepath(Metric.CPU_UTILISATION, variable, 'png'), format='png')
    plt.savefig(generate_filepath(Metric.CPU_UTILISATION, variable, 'svg'), format='svg')
    plt.show()


if __name__ == '__main__':
    plot(Variable.PACKET_SIZE)
    plot(Variable.PACKET_COUNT)
