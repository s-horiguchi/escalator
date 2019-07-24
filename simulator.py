#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import copy
from tqdm import trange,tqdm

def get_index_nn(y,x, width, height_floor, length_escalator):
    y = y - length_escalator
    index_nn = []
    if x > 0:
        index_nn.append([y,x-1])
    if x < width-1:
        index_nn.append([y,x+1])
    if y > 0:
        index_nn.append([y-1,x])
    if y < height_floor-1:
        index_nn.append([y+1,x])
    index_nn = np.array(index_nn)
    index_nn[:,0] += length_escalator
    return index_nn

class AgentMap(object):
    def __init__(self, width, height_floor, length_escalator,
                 x_exit_stand, x_exit_walk,
                 speed_stander, speed_walker):
        self.width = width
        self.height_floor = height_floor
        self.length_escalator = length_escalator
        self.height = height_floor + length_escalator
        assert len(x_exit_stand) == len(speed_stander)
        assert len(x_exit_walk) == len(speed_walker)
        self.x_exit_stand = x_exit_stand
        self.x_exit_walk = x_exit_walk
        self.speed_stander = speed_stander
        self.speed_walker = speed_walker
        self.escalator_stander = [[] for _ in speed_stander]
        self.escalator_walker = [[] for _ in speed_walker]
        self.map = np.zeros((self.height, width, 2), dtype=np.int)

    def initialize(self, n_stander=10, n_walker=10):
        if (len(self.speed_stander) == 0 and n_stander > 0) or (len(self.speed_walker) == 0 and n_walker > 0):
            raise ValueError("No exit but agents exist")
        self.map[:] = 0
        pos = np.random.choice(np.arange(self.width*self.height_floor),
                               size=(n_stander+n_walker), replace=False)
        y = pos // self.width + self.length_escalator
        x = pos % self.width
        atype = np.random.permutation(
            np.hstack([np.zeros(n_stander), np.ones(n_walker)])).astype(np.int)
        self.map[y, x, atype] = 1

    def copy(self):
        new_map = AgentMap(self.width, self.height_floor, self.length_escalator,
                           self.x_exit_stand, self.x_exit_walk,
                           self.speed_stander, self.speed_walker)
        new_map.map = self.map.copy()
        new_map.escalator_stander = copy.deepcopy(self.escalator_stander)
        new_map.escalator_walker = copy.deepcopy(self.escalator_walker)
        return new_map

    def get_vacant(self, indices):
        # return whether each index is vacant or not
        exist_agent = self.map[indices[:,0], indices[:,1]].astype(np.bool)
        vacant = np.bitwise_and(
            np.bitwise_not(exist_agent[:,0]),
            np.bitwise_not(exist_agent[:,1]))
        return vacant

    def get_next_cell(self, indices, distance_field, beta=1.0):
        # the most nearest cell to the goal among vacant given cells
        # a_type : "stander" or "walker"
        vacant = self.get_vacant(indices)
        if np.alltrue(np.bitwise_not(vacant)):
            return None

        distance = distance_field[indices[:,0], indices[:,1]]

        # transition probability ~ 1/(distance+1)^2
        pvals = np.exp(-beta * distance[vacant])
        pvals /= np.sum(pvals)
        #print(pvals)
        picked = np.random.choice(np.arange(len(pvals)), size=1, p=pvals)[0]
        #print(picked)
        return indices[vacant][picked]

    def move(self, prev_pos, next_pos):
        #print(prev_pos, next_pos)
        assert self.map[prev_pos[0],prev_pos[1],prev_pos[2]] == 1
        assert self.map[next_pos[0],next_pos[1],next_pos[2]] == 0
        self.map[prev_pos[0],prev_pos[1],prev_pos[2]] = 0
        self.map[next_pos[0],next_pos[1],next_pos[2]] = 1
        return

    def disappear(self, pos):
        #print("disappear", pos)
        assert self.map[pos[0],pos[1],pos[2]] == 1
        self.map[pos[0],pos[1],pos[2]] = 0
        return

    def appear(self, pos):
        assert self.map[pos[0],pos[1],pos[2]] == 0
        self.map[pos[0],pos[1],pos[2]] = 1
        return

    def spawn_everywhere(self, atype=0):
        vacant = np.bitwise_and(
            np.bitwise_not(self.map[self.length_escalator:,:,0].astype(np.bool)),
            np.bitwise_not(self.map[self.length_escalator:,:,1].astype(np.bool)))
        vacant_pos = np.array(np.nonzero(vacant)).T
        #print("vacant_pos", vacant_pos)
        i_spawn = np.random.choice(np.arange(len(vacant_pos)), size=1, replace=False)
        spawn_pos = vacant_pos[i_spawn][0]
        #print("spawn_pos", spawn_pos)
        self.appear((self.length_escalator+spawn_pos[0], spawn_pos[1], atype))
        return

    def spawn_endline(self, atype=0):
        vacant = np.bitwise_and(
            np.bitwise_not(self.map[self.height-1,:,0].astype(np.bool)),
            np.bitwise_not(self.map[self.height-1,:,1].astype(np.bool)))
        vacant_pos = np.array(np.nonzero(vacant)).T
        #print("vacant_pos", vacant_pos)
        i_spawn = np.random.choice(np.arange(len(vacant_pos)), size=1, replace=False)
        spawn_pos = vacant_pos[i_spawn][0]
        #print("spawn_pos", spawn_pos)
        self.appear((self.height-1, spawn_pos[0], atype))
        return

    def escalator_step(self):
        new_stander = [[] for _ in self.speed_stander]
        new_walker = [[] for _ in self.speed_walker]
        goal_s = 0
        goal_w = 0

        for i_escalator in range(len(self.speed_stander)):
            x = self.x_exit_stand[i_escalator]
            speed = self.speed_stander[i_escalator]
            for y in self.escalator_stander[i_escalator]:
                if y == 0:
                    #print("stander goal!")
                    self.disappear((0, x, 0))
                    goal_s += 1
                else:
                    next_y = max(y - speed, 0)
                    new_stander[i_escalator].append(next_y)
                    #print("stander", next_y)
                    self.move((y, x, 0), (next_y, x, 0))
        for i_escalator in range(len(self.speed_walker)):
            x = self.x_exit_walk[i_escalator]
            speed = self.speed_walker[i_escalator]
            for y in self.escalator_walker[i_escalator]:
                if y == 0:
                    #print("walker goal!")
                    self.disappear((0, x, 1))
                    goal_w += 1
                else:
                    next_y = max(y - speed, 0)
                    new_walker[i_escalator].append(next_y)
                    #print("walker", next_y)
                    self.move((y, x, 1), (next_y, x, 1))
        self.escalator_stander = new_stander
        self.escalator_walker = new_walker
        #print("list_stander@step", self.escalator_stander)
        #print("list_walker@step", self.escalator_walker)
        return goal_s, goal_w

    def ride(self, pos):
        #print("ride", pos)
        self.move(pos, (pos[0]-1,pos[1], pos[2]))
        
        if pos[2] == 0:
            # stander
            for i in range(len(self.speed_stander)):
                if pos[1] == self.x_exit_stand[i]:
                    self.escalator_stander[i].append(self.length_escalator-1)
                    return
        else:
            # walker
            for i in range(len(self.speed_walker)):
                if pos[1] == self.x_exit_walk[i]:
                    self.escalator_walker[i].append(self.length_escalator-1)
                    return
                    
        raise ValueError("This agent cannot ride")

class Simulation(object):
    def __init__(self, width=10, height_floor=20, length_escalator=10,
                 x_exit_stand = [4], x_exit_walk = [6],
                 speed_stander=[1], speed_walker=[2],
                 mu=0.2, beta=1.0, respawn=False):
        self.width = width
        self.height_floor = height_floor
        self.length_escalator = length_escalator
        self.height = height_floor + length_escalator
        self.history = []
        self.goal_stander = []
        self.goal_walker = []
        assert len(x_exit_stand) == len(speed_stander)
        assert len(x_exit_walk) == len(speed_walker)
        self.exit_stand = [(length_escalator, x) for x in x_exit_stand]
        self.exit_walk = [(length_escalator, x) for x in x_exit_walk]
        self.speed_stander = speed_stander
        self.speed_walker = speed_walker
        
        self.mu = mu
        self.beta = beta
        self.respawn = respawn

        # create static distance_field
        xcoord = np.repeat(np.arange(width).reshape(1,width), self.height, axis=0).reshape(self.height, width, 1)
        ycoord = np.repeat(np.arange(self.height).reshape(self.height, 1), width, axis=1)
        if len(speed_stander) == 0:
            self.distance_field_stand = np.zeros_like(ycoord, dtype=np.int)
        else:
            self.distance_field_stand = (np.min(np.abs(xcoord-np.array(x_exit_stand).reshape(1,1,len(x_exit_stand))), axis=2) \
                                         + np.abs(ycoord-length_escalator) + length_escalator).astype(np.int)
            self.distance_field_stand[:length_escalator, :] = np.max(self.distance_field_stand)+1
            for x in x_exit_stand:
                self.distance_field_stand[:length_escalator, x] = np.arange(length_escalator)

        if len(speed_walker) == 0:
            self.distance_field_walk = np.zeros_like(ycoord, dtype=np.int)
        else:
            self.distance_field_walk = (np.min(np.abs(xcoord-np.array(x_exit_walk).reshape(1,1,len(x_exit_walk))), axis=2) \
                                        + np.abs(ycoord-length_escalator) + length_escalator).astype(np.int)
            self.distance_field_walk[:length_escalator, :] = np.max(self.distance_field_walk)+1
            for x in x_exit_walk:
                self.distance_field_walk[:length_escalator, x] = np.arange(length_escalator)
        # create field for visualization
        self.field = np.zeros_like(self.distance_field_stand, dtype=np.int)
        self.field[:length_escalator, :] = 1
        for x in x_exit_stand + x_exit_walk:
            self.field[:length_escalator, x] = 0
        #print(self.distance_field_stand)
        #print(self.distance_field_walk)

    def initialize(self, **argv):
        agent_map = AgentMap(self.width, self.height_floor, self.length_escalator,
                             [pos[1] for pos in self.exit_stand], [pos[1] for pos in self.exit_walk],
                             self.speed_stander, self.speed_walker)
        agent_map.initialize(**argv)
        self.history = [agent_map]
        return agent_map

    def step(self):
        agent_map_next = self.history[-1].copy()

        # next_pos -> [(prev_pos1, a_type1), (prev_pos2, a_type2), ...]
        dict_move = defaultdict(lambda : [])
        list_ride = []

        for y in range(self.length_escalator, self.height):
            for x in range(self.width):
                #print("--",(y,x), "--")
                if agent_map_next.map[y, x, 0]:
                    # a stander exists
                    # disappear?
                    if (y,x) in self.exit_stand:
                        list_ride.append((y,x,0))
                    else:
                        # move?
                        nn_pos = get_index_nn(y, x, self.width, self.height_floor, self.length_escalator)
                        next_pos = agent_map_next.get_next_cell(nn_pos, self.distance_field_stand, self.beta)
                        if next_pos is not None:
                            #print(tuple(next_pos), "->", (y,x,0))
                            dict_move[tuple(next_pos)].append((y,x,0))
                if agent_map_next.map[y, x, 1]:
                    # a walker exists
                    # disappear?
                    if (y,x) in self.exit_walk:
                        list_ride.append((y,x,1))
                    else:
                        # move?
                        nn_pos = get_index_nn(y, x, self.width, self.height_floor, self.length_escalator)
                        next_pos = agent_map_next.get_next_cell(nn_pos, self.distance_field_walk, self.beta)
                        if next_pos is not None:
                            #print(tuple(next_pos), "->", (y,x,1))
                            dict_move[tuple(next_pos)].append((y,x,1))

        #print(dict(dict_move))
        #print("before")
        #print(agent_map_next.map[:,:,0])

        goal_s, goal_w = agent_map_next.escalator_step()
        self.goal_stander.append(goal_s)
        self.goal_walker.append(goal_w)
        for pos in list_ride:
            agent_map_next.ride(pos)
        #print("after escalator")
        #print(agent_map_next.map[:,:,0])

        for next_pos, prev_candidates in dict_move.items():
            if len(prev_candidates) == 0: continue
            elif len(prev_candidates) == 1:
                agent_map_next.move(
                    prev_candidates[0],
                    (next_pos[0],next_pos[1],prev_candidates[0][2])
                )
            else:
                # friction
                u = np.random.sample() # uniform(0,1)
                if u < self.mu: # stay
                    continue
                else: # move randomly picked agent
                    picked = np.random.choice(np.arange(len(prev_candidates)), size=1)[0]
                    #print(picked)
                    #print(prev_candidates[picked])
                    agent_map_next.move(
                        prev_candidates[picked],
                        (next_pos[0], next_pos[1], prev_candidates[picked][2])
                    )
        #print("after floor")
        #print(agent_map_next.map[:,:,0])

        # spawn on exit
        if self.respawn:
            for pos in list_ride:
                agent_map_next.spawn_endline(pos[2])

        self.history.append(agent_map_next)
        return agent_map_next

    def run(self, n_iter):
        for i in trange(n_iter):
            self.step()
        return

    def run_all_pass(self, max_time=10000):
        assert self.respawn == False
        n_s = np.count_nonzero(self.history[-1].map[:,:,0])
        n_w = np.count_nonzero(self.history[-1].map[:,:,1])
        n_total = n_s + n_w
        remaining = lambda : np.count_nonzero(self.history[-1].map[:,:,:])
        remaining_on_floor = lambda : np.count_nonzero(self.history[-1].map[self.length_escalator:,:,:])
        
        tq = tqdm(desc="s:{} w:{}".format(n_s, n_w), total=n_total)
        t_exit_floor = None
        while remaining() > 0 and len(self.history) < max_time:
            self.step()
            tq.update(self.goal_stander[-1] + self.goal_walker[-1])
            if not t_exit_floor and remaining_on_floor() == 0:
                t_exit_floor = len(self.history)
        if not t_exit_floor:
            t_exit_floor = max_time
        return t_exit_floor#len(self.history)

    def plot(self, time=-1):
        agent_map = self.history[time]

        agent_colors = np.array(["blue", "red"])
        fig = plt.figure(figsize=(8,3))
        ax1 = fig.add_subplot(1,2,1)
        im1 = ax1.imshow(self.distance_field_stand, cmap="Blues")
        fig.colorbar(im1, ax=ax1)
        ax1.set_title("distance field for stand")
        sc1 = ax1.scatter(
            np.array(agent_map.map.nonzero())[1,:],
            np.array(agent_map.map.nonzero())[0,:],
            c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
        )
        ax2 = fig.add_subplot(1,2,2)
        im2 = ax2.imshow(self.distance_field_walk, cmap="Reds")
        fig.colorbar(im2, ax=ax2)
        ax2.set_title("distance field for walk")
        sc2 = ax2.scatter(
            np.array(agent_map.map.nonzero())[1,:],
            np.array(agent_map.map.nonzero())[0,:],
            c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
        )
        #ax1.scatter(self.exit_stand[1], self.exit_stand[0], c="y", marker="*")
        #ax2.scatter(self.exit_walk[1], self.exit_walk[0], c="y", marker="*")
        return

    def animate(self, filename=None):
        agent_colors = np.array(["blue", "red"])
        fig = plt.figure(figsize=(8,3))
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
        ax1.set_title("distance field for stand")
        ax2.set_title("distance field for walk")
        ax3.set_title("t = 0")
        
        im1 = ax1.imshow(self.distance_field_stand, cmap="Blues")
        im2 = ax2.imshow(self.distance_field_walk, cmap="Reds")
        im3 = ax3.imshow(self.field, cmap="Greens", vmin=0, vmax=1)
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)

        agent_map = self.history[0]
        sc1 = ax1.scatter(
            np.array(agent_map.map.nonzero())[1,:],
            np.array(agent_map.map.nonzero())[0,:],
            c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
        )
        sc2 = ax2.scatter(
            np.array(agent_map.map.nonzero())[1,:],
            np.array(agent_map.map.nonzero())[0,:],
            c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
        )
        sc3 = ax3.scatter(
            np.array(agent_map.map.nonzero())[1,:],
            np.array(agent_map.map.nonzero())[0,:],
            c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
        )
        #ax1.scatter(self.exit_stand[1], self.exit_stand[0], c="y", marker="*")
        #ax2.scatter(self.exit_walk[1], self.exit_walk[0], c="y", marker="*")
        
        def _update(t, ax1, ax2):
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax3.set_title("t = {} / {}".format(t, len(self.history)))
            agent_map = self.history[t]
            im1 = ax1.imshow(self.distance_field_stand, cmap="Blues")
            im2 = ax2.imshow(self.distance_field_walk, cmap="Reds")
            im3 = ax3.imshow(self.field, cmap="Greens", vmin=0, vmax=1)
            #fig.colorbar(im1, ax=ax1)
            #fig.colorbar(im2, ax=ax2)
            sc1 = ax1.scatter(
                np.array(agent_map.map.nonzero())[1,:],
                np.array(agent_map.map.nonzero())[0,:],
                c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
            )
            sc2 = ax2.scatter(
                np.array(agent_map.map.nonzero())[1,:],
                np.array(agent_map.map.nonzero())[0,:],
                c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
            )
            sc3 = ax3.scatter(
                np.array(agent_map.map.nonzero())[1,:],
                np.array(agent_map.map.nonzero())[0,:],
                c=agent_colors[np.array(agent_map.map.nonzero())[2,:]]
            )
            #ax1.scatter(self.exit_stand[1], self.exit_stand[0], c="y", marker="*")
            #ax2.scatter(self.exit_walk[1], self.exit_walk[0], c="y", marker="*")
            return sc1,sc2,sc3

        ani = animation.FuncAnimation(fig, _update, fargs=(ax1,ax2), interval = 100, frames = len(self.history)-1)
        if filename:
            ani.save(filename, writer="imagemagick")
        else:
            pass

if __name__ == "__main__":
    sim = Simulation(width=20, height_floor=20, length_escalator=10,
                     x_exit_stand = [9,], x_exit_walk = [10],
                     speed_stander=[1,], speed_walker=[2],
                     mu=0.0, beta=10.0)
    sim.initialize(n_stander=90, n_walker=10)
    t = sim.run_all_pass()
    print("duration:", t)
    sim.animate(filename="animation_one_stand_s90w10.gif")
