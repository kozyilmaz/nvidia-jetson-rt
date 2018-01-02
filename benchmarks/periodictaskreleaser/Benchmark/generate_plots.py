# Copyright 2016, University of North Carolina
# Author: Nathan Otterness (otternes@cs.unc.edu)
#
# This script generates cdf plots for logs of GPU benchmarks in certain
# subdirectories. The logs consist of timestamps in seconds, program names,
# PIDs, the name of the phase of execution, and whether the phase has started
# or ended.
#
# Usage:
# To open windows showing the figures#
#
# python generate_ecdf_plots.py
#
# To save the figures to the current directory:
# python generate_ecdf_plot.py save_figures

from itertools import cycle
import matplotlib
import matplotlib.pyplot as plot
import numpy
import os
import sys
import json

global_known_styles = {}

# Used for sorting benchmark names when plotting.
def benchmark_name_comparator(a, b):
    total_a = 0
    total_b = 0
    for c in a:
        if (c >= '0') and (c <= '9'):
            total_a += ord(c) - ord('0')
    for c in b:
        if (c >= '0') and (c <= '9'):
            total_b += ord(c) - ord('0')
    if total_a != total_b:
        return total_a - total_b
    if a < b:
        return -1
    if a == b:
        return 0
    return 1

# Used for sorting benchmark names by median when plotting.
def name_and_median_comparator(a, b):
    # Sort by name when medians are too close to call.
    if abs(a[1] - b[1]) < 1.0:
        return benchmark_name_comparator(a[0], b[0])
    if a[1] < b[1]:
        return -1
    if a[1] > b[1]:
        return 1
    return 0

def create_single_plot(benchmark, x_axis_label="Benchmark time"):
    """Takes a single benchmark dict mapping <sharing scheme> ->
    <iteration times> and generates the CDF plot. Note that the iteration
    times, in this case, is a 1-D array of only 1 type of time (e.g. overall).
    """
    max_time = 0.0
    min_time = 1e30
    max_num_elements = 0.0
    for s in benchmark:
        if "* isolation" in s:
            continue
        if len(benchmark[s][0]) > max_num_elements:
            max_num_elements = len(benchmark[s])
        m = max(benchmark[s][0])
        if m > max_time:
            max_time = m
        m = min(benchmark[s][0])
        if m < min_time:
            min_time = m

    x_range = max_time - min_time
    # Do this so we don't crash if an experiment provides no range at all.
    if x_range == 0:
        x_range = 0.01
    x_pad = x_range * 0.05
    # Scale the x-axis to give a little "padding" based on the max value.
    plot.axis([min_time - x_pad, max_time + x_pad, -5.0, 105.0])
    plot.xticks(numpy.arange(min_time, max_time + x_pad, x_range / 5.0))
    plot.yticks(numpy.arange(0, 105.0, 20.0))
    plot.xlabel(x_axis_label)
    plot.ylabel("% <= x")

    line_styles = []
    line_styles.append({"color": "k", "linestyle": "-"})
    line_styles.append({"color": "red", "linestyle": "--", "markevery": 0.075, "markersize": 8, "marker": "x", "mew": 1.0})
    line_styles.append({"color": "blue", "linestyle": "-", "markevery": 0.075, "markersize": 6, "marker": "o"})
    line_styles.append({"color": "green", "linestyle": "--", "markevery": 0.075, "markersize": 8, "marker": ">"})
    line_styles.append({"color": "k", "linestyle": "-.", "markevery": 0.075, "markersize": 8, "marker": "*"})
    line_styles.append({"color": "grey", "linestyle": "--"})
    line_styles.append({"color": "k", "linestyle": "-", "dashes": [8, 4, 2, 4, 2, 4]})
    line_styles.append({"color": "grey", "linestyle": "-", "dashes": [8, 4, 2, 4, 2, 4]})
    line_styles.append({"color": "grey", "linestyle": "-."})
    style_cycler = cycle(line_styles)

    # Junk to make sure legends are sorted by the order of the lines in the
    # plot. The "median" is actually at the 70% mark.
    keys_and_medians = []
    for k in benchmark.keys():
        scenario_median = 0.0
        scenario_data = benchmark[k][0]
        scenario_percent = benchmark[k][1]
        for i in range(len(scenario_percent)):
            if scenario_percent[i] >= 70.0:
                scenario_median = scenario_data[i]
                break
        keys_and_medians.append([k, scenario_median])
    keys_and_medians.sort(name_and_median_comparator)

    global global_known_styles
    for key_and_median in keys_and_medians:
        key = key_and_median[0]
        cdf_points = benchmark[key]
        # Additional check to make sure that we always assign the same style
        # to the same distribution, even if the order changes.
        if key in global_known_styles:
            line_style = global_known_styles[key]
        else:
            print "Key " + key + " not found in styles. adding"
            line_style = next(style_cycler)
            global_known_styles[key] = line_style
        if len(cdf_points[0]) == 0:
            continue
        # Grayscale/patterned lines
        plot.plot(cdf_points[0], cdf_points[1], linewidth=1.8, label=key,
            **line_style)
        # Colored lines
        #plot.plot(cdf_points[0], cdf_points[1], linewidth=2.0, label=key)
    #legend = plot.legend(bbox_to_anchor=(1, 1.3), ncol=2)
    legend = plot.legend(loc=3, ncol=2, bbox_to_anchor=(0., 1.02, 1., .102), mode="expand", borderaxespad=0.0)
    legend.draggable()

def get_data():
    """Loads data and groups it by benchmark. Returns a dict mapping
    {benchmark name: {sharing scenario: iteration times}}."""
    to_return = None
    with open("output.json") as data_file:
        to_return = json.load(data_file)
    return to_return

def create_plots():
    """Loads and displays plots of all found benchmark data."""
    data = get_data()
    to_return = dict()
    i = 0
    matplotlib.rc('font', size=16)
    for scenario in data:
        print "Plotting scenario %s" % (scenario)
        ex = data[scenario]
        i += 1
        figure = plot.figure(i, figsize=(10, 6.8))
        create_single_plot(ex, scenario + " (ms)")
        figure.tight_layout()
        plot.subplots_adjust(top=0.85)
        to_return[scenario] = figure
    return to_return

if __name__ == "__main__":
    figures = create_plots()
    if "save_figures" not in sys.argv:
        plot.show()
        exit(0)
    for name in figures:
        figures[name].savefig("output_figures/" + name + ".pdf")
