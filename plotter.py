from enum import Enum
import re
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt


ALL_PACKET_SIZES = [512, 1_024, 2_048, 4_096, 8_192]
ALL_PACKET_COUNTS = [20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000]
ALL_CONSUMER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Metric(Enum):
    TIME = ('Execution Time', 'time', 'Time (s)')
    PACKET_LATENCY = ('Packet Latency', 'packet_latency', 'Latency (\u03BCs)')
    CPU_UTILISATION = ('CPU Utilisation', 'cpu_utilisation', 'CPU Utilisation (%)')

    def __init__(self, title, partial_filename, axis):
        self.title = title
        self.partial_filename = partial_filename
        self.axis = axis


class Variable(Enum):
    PACKET_SIZE = ('Packet Size', 'packet_size', 'Packet Size (bytes)')
    PACKET_COUNT = ('Packet Count', 'packet_count', 'Packet Count')
    CONSUMER_COUNT = ('Consumer Count', 'consumer_count', 'Consumer Count')

    def __init__(self, title, partial_filename, axis):
        self.title = title
        self.partial_filename = partial_filename
        self.axis = axis


def generate_filepath(metric, variable, format):
    return f'./graphs/{metric.partial_filename}_v_{variable.partial_filename}.{format}'


def parse_data(filename, variable):
    times = []
    lats = []
    utils = []
    with open(f'./results/{variable.partial_filename}/{filename}') as f:
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


def plot(xs, s_mean, s_sd, i_mean, i_sd, metric, variable):
    plt.errorbar(xs, s_mean, s_sd, linestyle='None', marker='x', label='CHERI', capsize=5, elinewidth=1)
    plt.errorbar(xs, i_mean, i_sd, linestyle='None', marker='x', label='IPC', capsize=5, elinewidth=1)
    plt.title(f'{metric.title} Vs. {variable.title}')
    plt.xlabel(variable.axis)
    plt.ylabel(metric.axis)
    plt.legend()
    plt.savefig(generate_filepath(metric, variable, 'png'), format='png')
    plt.savefig(generate_filepath(metric, variable, 'svg'), format='svg')
    plt.show()


def plot_all(variable):
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

    s_time_mean = []
    s_time_sd = []
    s_lat_mean = []
    s_lat_sd = []
    s_util_mean = []
    s_util_sd = []
    for file in base_s_filenames:
        try:
            times, lats, utils = parse_data(file, variable)

            s_time_mean.append(mean(times))
            s_time_sd.append(stdev(times))

            s_lat_mean.append(mean(lats))
            s_lat_sd.append(stdev(lats))

            s_util_mean.append(mean(utils))
            s_util_sd.append(stdev(utils))
        except FileNotFoundError:
            s_time_mean.append(None)
            s_time_sd.append(None)
            s_lat_mean.append(None)
            s_lat_sd.append(None)
            s_util_mean.append(None)
            s_util_sd.append(None)

    i_time_mean = []
    i_time_sd = []
    i_lat_mean = []
    i_lat_sd = []
    i_util_mean = []
    i_util_sd = []
    for file in base_i_filenames:
        try:
            times, lats, utils = parse_data(file, variable)

            i_time_mean.append(mean(times))
            i_time_sd.append(stdev(times))

            i_lat_mean.append(mean(lats))
            i_lat_sd.append(stdev(lats))

            i_util_mean.append(mean(utils))
            i_util_sd.append(stdev(utils))
        except FileNotFoundError:
            i_time_mean.append(None)
            i_time_sd.append(None)
            i_lat_mean.append(None)
            i_lat_sd.append(None)
            i_util_mean.append(None)
            i_util_sd.append(None)

    plot(xs, s_time_mean, s_time_sd, i_time_mean, i_time_sd, Metric.TIME, variable)
    plot(xs, s_lat_mean, s_lat_sd, i_lat_mean, i_lat_sd, Metric.PACKET_LATENCY, variable)
    plot(xs, s_util_mean, s_util_sd, i_util_mean, i_util_sd, Metric.CPU_UTILISATION, variable)


if __name__ == '__main__':
    plot_all(Variable.PACKET_SIZE)
    plot_all(Variable.PACKET_COUNT)
