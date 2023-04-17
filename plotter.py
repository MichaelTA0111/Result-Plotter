from enum import Enum
import re
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fmt = '${x:,}'
tick = mtick.StrMethodFormatter(fmt)


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


def generate_filepath(metric, variable, format, adjusted=False, noproc=False):
    if adjusted:
        return f'./graphs/adjusted_{metric.partial_filename}_v_{variable.partial_filename}.{format}'
    elif noproc:
        return f'./graphs/{metric.partial_filename}_v_{variable.partial_filename}_with_noproc.{format}'
    else:
        return f'./graphs/{metric.partial_filename}_v_{variable.partial_filename}.{format}'


def parse_data(filename, variable, proc):
    times = []
    lats = []
    utils = []
    with open(f'./results/{variable.partial_filename}/{proc}/{filename}') as f:
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

            util = data[2]
            util = float(util)
            utils.append(util)

    return times, lats, utils


def plot(xs, s_mean, s_sd, i_mean, i_sd, metric, variable):
    s_mean = np.array(s_mean).astype(np.double)
    s_sd = np.array(s_sd).astype(np.double)
    s_mask = np.isfinite(s_mean)
    i_mean = np.array(i_mean).astype(np.double)
    i_sd = np.array(i_sd).astype(np.double)
    i_mask = np.isfinite(i_mean)

    plt.errorbar(xs[s_mask], s_mean[s_mask], s_sd[s_mask],
                 linestyle='None', marker='x', label='CHERI', capsize=5, elinewidth=1)
    plt.errorbar(xs[i_mask], i_mean[i_mask], i_sd[i_mask],
                 linestyle='None', marker='x', label='IPC', capsize=5, elinewidth=1)
    plt.title(f'{metric.title} Vs. {variable.title}')
    plt.xlabel(variable.axis)
    plt.ylabel(metric.axis)
    if variable is Variable.PACKET_COUNT:
        plt.xticks(ALL_PACKET_COUNTS, fontsize=8)
    elif variable is Variable.PACKET_SIZE:
        plt.xticks(ALL_PACKET_SIZES, fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend()
    plt.savefig(generate_filepath(metric, variable, 'png'), format='png')
    plt.savefig(generate_filepath(metric, variable, 'svg'), format='svg')
    plt.show()


def plot_n(xs, s_mean, s_sd, i_mean, i_sd, n_mean, n_sd, metric, variable):
    s_mean = np.array(s_mean).astype(np.double)
    s_sd = np.array(s_sd).astype(np.double)
    s_mask = np.isfinite(s_mean)
    i_mean = np.array(i_mean).astype(np.double)
    i_sd = np.array(i_sd).astype(np.double)
    i_mask = np.isfinite(i_mean)
    n_mean = np.array(n_mean).astype(np.double)
    n_sd = np.array(n_sd).astype(np.double)
    n_mask = np.isfinite(n_mean)

    plt.errorbar(xs[s_mask], s_mean[s_mask], s_sd[s_mask],
                 linestyle='None', marker='x', label='CHERI', capsize=5, elinewidth=1)
    plt.errorbar(xs[i_mask], i_mean[i_mask], i_sd[i_mask],
                 linestyle='None', marker='x', label='IPC', capsize=5, elinewidth=1)
    plt.errorbar(xs[n_mask], n_mean[n_mask], n_sd[n_mask],
                 linestyle='None', marker='x', label='No Proc', capsize=5, elinewidth=1)
    plt.title(f'{metric.title} Vs. {variable.title}')
    plt.xlabel(variable.axis)
    plt.ylabel(metric.axis)
    if variable is Variable.PACKET_COUNT:
        plt.xticks(ALL_PACKET_COUNTS, fontsize=8)
    elif variable is Variable.PACKET_SIZE:
        plt.xticks(ALL_PACKET_SIZES, fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().ticklabel_format(style='plain')
    plt.legend()
    plt.savefig(generate_filepath(metric, variable, 'png', noproc=True), format='png')
    plt.savefig(generate_filepath(metric, variable, 'svg', noproc=True), format='svg')
    plt.show()


def plot_adjusted(xs, s_mean, s_sd, i_mean, i_sd, n_mean, metric, variable):
    s_mean = [s - n for s, n in zip(s_mean, n_mean)]
    i_mean = [i - n for i, n in zip(i_mean, n_mean)]

    s_mean = np.array(s_mean).astype(np.double)
    s_sd = np.array(s_sd).astype(np.double)
    s_mask = np.isfinite(s_mean)
    i_mean = np.array(i_mean).astype(np.double)
    i_sd = np.array(i_sd).astype(np.double)
    i_mask = np.isfinite(i_mean)

    plt.errorbar(xs[s_mask], s_mean[s_mask], s_sd[s_mask],
                 linestyle='None', marker='x', label='CHERI', capsize=5, elinewidth=1)
    plt.errorbar(xs[i_mask], i_mean[i_mask], i_sd[i_mask],
                 linestyle='None', marker='x', label='IPC', capsize=5, elinewidth=1)
    plt.title(f'Corrected {metric.title} Vs. {variable.title}')
    plt.xlabel(variable.axis)
    plt.ylabel(metric.axis)
    if variable is Variable.PACKET_COUNT:
        plt.xticks(ALL_PACKET_COUNTS, fontsize=8)
    elif variable is Variable.PACKET_SIZE:
        plt.xticks(ALL_PACKET_SIZES, fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().ticklabel_format(style='plain')
    plt.legend()
    plt.savefig(generate_filepath(metric, variable, 'png', adjusted=True), format='png')
    plt.savefig(generate_filepath(metric, variable, 'svg', adjusted=True), format='svg')
    plt.show()


def print_adjusted_times(s_mean, i_mean, n_mean):
    s_mean_adj = [s - n for s, n in zip(s_mean, n_mean)]
    i_mean_adj = [i - n for i, n in zip(i_mean, n_mean)]

    print(f'Adjusted CHERI execution times {s_mean_adj}')
    print(f'Adjusted IPC execution times {i_mean_adj}')


def print_adjusted_latencies(s_mean, i_mean, n_mean):
    s_mean_adj = [s - n for s, n in zip(s_mean, n_mean)]
    i_mean_adj = [i - n for i, n in zip(i_mean, n_mean)]

    print(f'Adjusted CHERI packet processing latencies {s_mean_adj}')
    print(f'Adjusted IPC packet processing latencies {i_mean_adj}')


def print_adjusted_utilisations(s_mean, i_mean, n_mean):
    s_mean_adj = [s - n for s, n in zip(s_mean, n_mean)]
    i_mean_adj = [i - n for i, n in zip(i_mean, n_mean)]

    print(f'Adjusted CHERI CPU utilisations {s_mean_adj}')
    print(f'Adjusted IPC CPU utilisations {i_mean_adj}')


def plot_all(variable):
    # Specify 2 of 3 arguments
    if variable is Variable.PACKET_SIZE:
        xs = np.array(ALL_PACKET_SIZES)
        base_s_filenames = [f'{z:_}B__100_000P__2C.txt' for z in ALL_PACKET_SIZES]
        base_i_filenames = [f'{z:_}B__100_000P__2C.txt' for z in ALL_PACKET_SIZES]
    elif variable is Variable.PACKET_COUNT:
        xs = np.array(ALL_PACKET_COUNTS)
        base_s_filenames = [f'512B__{z:_}P__2C.txt' for z in ALL_PACKET_COUNTS]
        base_i_filenames = [f'512B__{z:_}P__2C.txt' for z in ALL_PACKET_COUNTS]
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
            times, lats, utils = parse_data(file, variable, 'cheri')

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
            times, lats, utils = parse_data(file, variable, 'ipc')

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

    n_time_mean = []
    n_time_sd = []
    n_lat_mean = []
    n_lat_sd = []
    n_util_mean = []
    n_util_sd = []
    for file in base_s_filenames:
        try:
            times, lats, utils = parse_data(file, variable, 'none')

            n_time_mean.append(mean(times))
            n_time_sd.append(stdev(times))

            n_lat_mean.append(mean(lats))
            n_lat_sd.append(stdev(lats))

            n_util_mean.append(mean(utils))
            n_util_sd.append(stdev(utils))
        except FileNotFoundError:
            n_time_mean.append(None)
            n_time_sd.append(None)
            n_lat_mean.append(None)
            n_lat_sd.append(None)
            n_util_mean.append(None)
            n_util_sd.append(None)

    plot(xs, s_time_mean, s_time_sd, i_time_mean, i_time_sd, Metric.TIME, variable)
    plot(xs, s_lat_mean, s_lat_sd, i_lat_mean, i_lat_sd, Metric.PACKET_LATENCY, variable)
    plot(xs, s_util_mean, s_util_sd, i_util_mean, i_util_sd, Metric.CPU_UTILISATION, variable)

    plot_n(xs, s_time_mean, s_time_sd, i_time_mean, i_time_sd, n_time_mean, n_time_sd, Metric.TIME, variable)
    plot_n(xs, s_lat_mean, s_lat_sd, i_lat_mean, i_lat_sd, n_lat_mean, n_lat_sd, Metric.PACKET_LATENCY, variable)
    plot_n(xs, s_util_mean, s_util_sd, i_util_mean, i_util_sd, n_util_mean, n_util_sd, Metric.CPU_UTILISATION, variable)

    print_adjusted_times(s_time_mean, i_time_mean, n_time_mean)
    plot_adjusted(xs, s_time_mean, s_time_sd, i_time_mean, i_time_sd, n_time_mean, Metric.TIME, variable)
    print_adjusted_latencies(s_lat_mean, i_lat_mean, n_lat_mean)
    plot_adjusted(xs, s_lat_mean, s_lat_sd, i_lat_mean, i_lat_sd, n_lat_mean, Metric.PACKET_LATENCY, variable)
    print_adjusted_utilisations(s_util_mean, i_util_mean, n_util_mean)
    plot_adjusted(xs, s_util_mean, s_util_sd, i_util_mean, i_util_sd, n_util_mean, Metric.CPU_UTILISATION, variable)


if __name__ == '__main__':
    plot_all(Variable.PACKET_SIZE)
    plot_all(Variable.PACKET_COUNT)
