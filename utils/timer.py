#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import pdb
import torch
import numpy as np

times = {}
times.setdefault('batch', [])
times.setdefault('data', [])
mark = False  # Use for starting and stopping the timer
max_len = 100


def init(length=100):
    global times, mark, max_len
    times = {}
    times.setdefault('batch', [])
    times.setdefault('data', [])
    times.setdefault('val', [])
    mark = False
    max_len = length


def start():
    global mark
    mark = True


def stop():
    global mark
    mark = False


def add_batch_time(batch_time):
    if len(times['val']) != 0:  # exclude the validation time
        batch_time = batch_time - times['val'][0]
        times['val'] = []

    times['batch'].append(batch_time)

    inner_time = 0
    for k, v in times.items():
        if k not in ('batch', 'data', 'val'):
            inner_time += v[-1]

    times['data'].append(batch_time - inner_time)


def get_times(time_name):
    return_time = []
    for name in time_name:
        return_time.append(np.mean(times[name]))

    return return_time


def print_timer():
    print('---------Time Statistics---------')
    for k, v in times.items():
        print(f'{k}: {np.mean(v):.4f}')

    forward_fps = 1 / np.mean(times['forward'])
    total_fps = 1 / np.mean(times['batch'])
    print(f'forward fps: {forward_fps:.2f}, total fps: {total_fps:.2f}')
    print('---------------------------------')


class counter:
    def __init__(self, name):
        self.name = name
        self.times = times
        self.mark = mark
        self.max_len = max_len

        for v in times.values():
            if len(v) >= self.max_len:  # pop the first item if the list is full
                v.pop(0)

    def __enter__(self):
        if self.mark:
            torch.cuda.synchronize()
            self.times.setdefault(self.name, [])
            self.times[self.name].append(time.perf_counter())

    def __exit__(self, e, ev, t):
        if self.mark:
            torch.cuda.synchronize()
            self.times[self.name][-1] = time.perf_counter() - self.times[self.name][-1]
