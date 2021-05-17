import numpy as np


def add_node(name, rel):
    edge = {}
    with open('./OpenKE/benchmarks/' + name + '_1/train2id.txt','r') as f:
        f.readline()
        for line in f:
            tmp = line.strip().split('\t')
            tmp[0] = int(tmp[0])
            tmp[1] = int(tmp[1])
            if tmp[0] not in edge.keys():
                edge[tmp[0]] = set()
                edge[tmp[0]].add(tmp[1])
            else:
                edge[tmp[0]].add(tmp[1])
            if tmp[1] not in edge.keys():
                edge[tmp[1]] = set()
                edge[tmp[1]].add(tmp[0])
            else:
                edge[tmp[1]].add(tmp[0])
    with open('./OpenKE/benchmarks/' + name + '_1/test2id.txt','r') as f:
        f.readline()
        for line in f:
            tmp = line.strip().split('\t')
            tmp[0] = int(tmp[0])
            tmp[1] = int(tmp[1])
            if tmp[0] not in edge.keys():
                edge[tmp[0]] = set()
                edge[tmp[0]].add(tmp[1])
            else:
                edge[tmp[0]].add(tmp[1])
            if tmp[1] not in edge.keys():
                edge[tmp[1]] = set()
                edge[tmp[1]].add(tmp[0])
            else:
                edge[tmp[1]].add(tmp[0])
    with open('./OpenKE/benchmarks/' + name + '_1/valid2id.txt','r') as f:
        f.readline()
        for line in f:
            tmp = line.strip().split('\t')
            tmp[0] = int(tmp[0])
            tmp[1] = int(tmp[1])
            if tmp[0] not in edge.keys():
                edge[tmp[0]] = set()
                edge[tmp[0]].add(tmp[1])
            else:
                edge[tmp[0]].add(tmp[1])
            if tmp[1] not in edge.keys():
                edge[tmp[1]] = set()
                edge[tmp[1]].add(tmp[0])
            else:
                edge[tmp[1]].add(tmp[0])
    add_node = set()
    link_node = {}
    ex_node = set(rel[:,0])
    tmp_set = set()
    for key in edge.keys():
        if key in ex_node:
            for tmp in edge[key]:
                if tmp not in ex_node and tmp not in tmp_set:
                    add_node.add(tmp)
                    link_node[key] = tmp
                    tmp_set.add(tmp)
                    break
    return edge,add_node,link_node

