#!/usr/bin/env python
from __future__ import print_function
import nixio as nix
import numpy as np
import scipy.io as scio
# import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import nixio as nix

from IPython import embed


def write_channel_data(block, data, time, sr):
    group = block.create_group("eeg data", "nix.eeg.channels")
    dt = np.mean(np.diff(time))
    diff = 1./dt - sr
    use_range = diff > np.finfo(np.float32).eps

    if use_range:
        print("sampling rate does not match timestamps using range dimension (need more space!)",
              file=sys.stderr)

    nchan = data.shape[0]

    for ch in range(nchan):
        chdata = data[ch, :]
        da = block.create_data_array("channel %d" % (ch + 1), "nix.eeg.channeldata",
                                     data=chdata.astype(np.double))
        da.unit = "uV"
        da.label = "voltage"
        if use_range:
            dim = da.append_range_dimension(time)
        else:
            dim = da.append_sampled_dimension(dt)
        dim.unit = "s"
        dim.label = "time"
        group.data_arrays.append(da)
    return group


def save_events(block, trigger, group):
    states = trigger[np.nonzero(np.diff(trigger))]
    indices = np.nonzero(np.diff(trigger))
    times = indices[0].astype(np.double) / 512
    corners = times[(states == 8) | (states == 10)]
    exp_start = times[(states == 4) | (states == 6)]

    corner_positions = block.create_data_array("corner_times", "nix.timestamps", data=corners)
    corner_positions.label = "time"
    corner_positions.unit = "s"
    corner_positions.append_alias_range_dimension()
    corner_events = block.create_multi_tag("corners", "nix.eeg.event", corner_positions)

    exp_positions = block.create_data_array("experiment times", "nix.timestamps", data=exp_start)
    exp_positions.label = "time"
    exp_positions.unit = "s"
    exp_positions.append_alias_range_dimension()
    extents = np.ones(len(exp_start))
    extents[-1] = 100.
    exp_extents = block.create_data_array("experiment durations", "nix.extents", data=extents)
    exp_extents.label = "time"
    exp_extents.unit = "s"
    exp_extents.append_alias_range_dimension()
    exp_starts = block.create_multi_tag("experiment starts", "nix.eeg.event", exp_positions)
    exp_starts.extents = exp_extents
    for da in group.data_arrays:
        exp_starts.references.append(da)
        corner_events.references.append(da)

   
def write_trigger_signal(block, trigger, time, da_group):
    trigger_da = block.create_data_array("trigger signal", "nix.eeg.trigger",
                                         data=trigger.astype(np.double))
    trigger_da.label = "voltage"
    trigger_da.unit = "mV"
    dim = trigger_da.append_sampled_dimension(np.mean(np.diff(time)))
    dim.unit = "s"
    dim.label = "time"

    tag = block.create_tag("trigger signal", "nix.eeg.trigger", [0.])
    tag.extent = [time[-1]]  # list of extents, one for each dimension
    tag.units = ["s"]  # list of units, need one entry for each dimension of the data
    for da in da_group.data_arrays:
        tag.references.append(da)
    tag.create_feature(trigger_da, nix.LinkType.Tagged)


def convert(time, trigger, data, parts, sr):
    f = nix.File.open(parts[0] + ".nix", nix.FileMode.Overwrite)
    b = f.create_block(parts[0], "nix.eeg.session")
    g = write_channel_data(b, data, time, sr)
    write_trigger_signal(b, trigger, time, g)
    save_events(b, trigger, g)
    f.close()


def load_data(filename):
    folder = os.path.dirname(filename)
    full_name = os.path.basename(filename)
    name, ext = os.path.splitext(full_name)
    file_parts = name.split("_")
    pattern = "_".join(file_parts[:-1])
    files = glob.glob(os.path.join(folder, pattern + "*.mat"))
    combined_data = None
    for f in sorted(files):
        print("Loading file %s" % f, file=sys.stderr)
        data = scio.matlab.loadmat(f)
        y = np.squeeze(data["y"])
        if combined_data is None:
            combined_data = y
        else:
            last_time = combined_data[0, -1]
            dt = np.mean(np.diff(combined_data[0, :]))
            y[0, :] = y[0, :] + last_time + dt
            combined_data = np.hstack((combined_data, y))
    sr_key = [x for x in data.keys() if x.startswith('SR')][0]
    sr = data[sr_key][0][0]
    time = combined_data[0, :]
    trigger = combined_data[-1, :]
    data_eeg = combined_data[1:-2, :]
    return time, trigger, data_eeg, file_parts, sr


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("filename")
    # parser.add_argument("trigger_csv")
    # parser.add_argument("order")

    args = parser.parse_args()
    time, trigger, data, parts, sr = load_data(args.filename)
    convert(time, trigger, data, parts, sr)

if __name__ == "__main__":
    main()


