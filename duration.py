#!/usr/bin/env python
#-*- coding:utf-8 -*-

from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    sim = Simulation(width=20, height_floor=20, length_escalator=10,
                     x_exit_stand = [9], x_exit_walk = [10],
                     speed_stander=[1], speed_walker=[2],
                     mu=0.0, beta=10)

    ns_s = np.linspace(1, 100, 20).astype(np.int)
    N_iter = 10
    duration = np.zeros((len(ns_s), N_iter))
    
    for i_s in range(len(ns_s)):
        for n in range(N_iter):
            sim.initialize(n_stander=ns_s[i_s], n_walker=100-ns_s[i_s])
            duration[i_s,n] = sim.run_all_pass(1000)

    np.save("duration", duration)

    plt.plot(ns_s, np.median(duration, axis=1))
    plt.scatter(np.repeat(ns_s, 10), duration.flatten(), marker="x", s=15,)
    plt.xlabel("number of stander")
    plt.ylabel("number of steps to finish")
    plt.savefig("duration.png")
