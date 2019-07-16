import os
import random
import time
import uuid
from subprocess import Popen

import numpy as np
from lxml import etree
from parallelpy.utils import Letter

from evodevo.moo_interfaces import MOORobotInterface
from evodevo.utils.print_utils import print_all


class SoftbotRobot(MOORobotInterface):
    def __init__(self, robot, seq_num_gen, run_dir, **kwargs):
        self.run_dir = run_dir
        self.seq_num_gen = seq_num_gen
        self.id = self.seq_num_gen()
        self.robot = robot
        if "max_eval_time" in kwargs:
            self.max_eval_time = kwargs["max_eval_time"]
        else:
            self.max_eval_time = 60 * 60 * 24  # 1 day: if we don't have an upper bound and the simulation enters a while true loop... this code also crashes.

        self.fitness = -1000
        self.raw_fit = np.zeros(shape=(2, 2))

        self.needs_eval = True

        self.age = 0

    def __str__(self):
        return "AFPO BOT (id: %s): f: %.2f age: %d" % (str(self.id),
                                                       self.get_fitness(),
                                                       self.get_age())

    def __repr__(self):
        return str(self)


    def get_id(self):
        return self.id


    # Methods for MOORObotInterface class

    def iterate_generation(self):
        self.age += 1

    def needs_evaluation(self):
        return self.needs_eval

    def mutate(self):
        self.needs_eval = True
        self.id = self.seq_num_gen()
        self.fitness = -1000

    def get_minimize_vals(self):
        return [self.get_age()]

    def get_maximize_vals(self):
        return [self.fitness]

    def get_seq_num(self):
        return self.id

    def get_fitness(self, test=False):
        return self.fitness

    def get_data(self):
        ret = [self.get_fitness(), self.get_fitness(test=True), self.get_age()]
        return ret

    def get_data_column_count(self):
        """
        Returns the number of data columns needed to store this robot
        :param num_commands: number of train_commands being sent to the robot
        :return: number of data columns needed to store this robot
        """
        return 3 + 0 # fitness, fitness w/ test, age, + fitness of each environment

    def dominates_final_selection(self, other):
        return self.get_fitness() > other.get_fitness()

    # Methods for Work class
    def cpus_requested(self):
        return 1

    def compute_work(self, test=True, **kwargs):
        pass

    def write_self_description(self, dir):
        with open("%s/%d.txt"%(dir, self.id), "w") as f:
            f.write("%s\n"%str(self))


    def write_letter(self):
        return Letter((0,0), None)

    def open_letter(self, letter):
        self.fitness = random.random()
        self.needs_eval = False
        return None

    def get_num_evaluations(self, test=False):
        return 1

    def get_age(self):
        return self.age

    def _flatten(self, l):
        ret = []
        for items in l:
            ret += items
        return ret


class SimulationError(Exception):
    pass
