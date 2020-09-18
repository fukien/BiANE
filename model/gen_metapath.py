#! ~/conda3/bin/python3
# -*- coding: utf-8 -*-

import argparse
import multiprocessing
import os
import pickle
import random
import sys
import time

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('metapath2vec++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ami',
        # default='mvl'
        )
    parser.add_argument('--path_per_node', type=int, )
    parser.add_argument('--path_length', type=int, )
    args = parser.parse_args()
    adjlist_file = os.path.join('../data', args.dataset, 'adjlist.txt')
    metapath_file = os.path.join('../data', args.dataset, 'metapath_{}.txt'.format(args.dataset))

    adjlist_dict = {}
    with open(adjlist_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split(' ')
            anchor_node = int(line[0])
            neighbor_node_list = [int(node) for node in line[1:]]
            adjlist_dict[anchor_node] = neighbor_node_list

    def worker(proc_id, works):
        paths = []
        for start_node in works:
            for _ in range(args.path_per_node):
                path = []
                cur_node = start_node
                for _ in range(args.path_length):
                    path.append(cur_node)
                    if adjlist_dict[cur_node]:
                        cur_node = random.choice(adjlist_dict[cur_node])
                paths.append(path)
        return paths

    process_num = multiprocessing.cpu_count() - 10
    pool = multiprocessing.Pool(processes=process_num)
    process_input = []
    process_output = []
    for _ in range (process_num):
        process_input.append([])

    for idx, start_node in enumerate(range(len(adjlist_dict))):
        process_input[idx%process_num].append(start_node)

    for idx in range(process_num):
        result = pool.apply_async(worker, (idx, process_input[idx]))
        process_output.append(result)

    pool.close()
    pool.join()

    path_list = []
    for result in process_output:
        paths = result.get()
        path_list += paths

    random.shuffle(path_list)

    with open(metapath_file, 'w', encoding= 'utf-8') as f:
        for path in path_list:
            line = ' '.join([str(node) for node in path]) + '\n'
            f.write(line)