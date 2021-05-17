import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from Parser import Parser


def Fun(x, a1, a2, a3, a4, a5):
    return a1 * x ** 4 + a2 * x ** 3 + a3 * x ** 2 + a4 * x + a5


def error(p, x, y):
    return Fun(p, x) - y


if __name__ == '__main__':
    baseline = int(sys.argv[1])
    strategy_1 = int(sys.argv[2])
    strategy_2 = int(sys.argv[3])

    # get all_kg
    parser = Parser(baseline)
    for kg in parser.result.keys():
        for task_id in [baseline, strategy_1, strategy_2]:
            parser = Parser(task_id)
            x = list()
            y = list()
            round = 0
            for round, record in enumerate(parser.result[kg]):
                x.append(round)
                y.append(record['accuracy'])
            if task_id == baseline:
                co = 'b'
            elif task_id == strategy_1:
                co = 'g'
            elif task_id == strategy_2:
                co = 'r'
            x = np.array(x)
            y = np.array(y)
            
            # para, pcov = curve_fit(Fun, x, y)
            # y = Fun(x, para[0], para[1], para[2], para[3], para[4])
            plt.plot(x, y, co)
        plt.savefig('./tmp/%s.png' % kg)
        plt.clf()
