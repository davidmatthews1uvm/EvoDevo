import random

import numpy as np
from parallelpy.utils import Letter

from evodevo.moo_interfaces import MOORobotInterface


class SoftbotRobot(MOORobotInterface):
    def __init__(self, robot, run_dir, **kwargs):
        self.run_dir = run_dir
        self.id = -1
        self.parent_id = -1
        self.robot = robot

        self.fitness = -1000

        self.needs_eval = True
        self.age = 0

    def __str__(self):
        return "AFPO BOT (id: %s): f: %.2f age: %d parent: %d" % (str(self.id),
                                                                  self.get_fitness(),
                                                                  self.get_age(),
                                                                  self.get_parent_id())

    def __repr__(self):
        return str(self)

    def set_id(self, newid):
        self.id = newid

    def get_id(self):
        return self.id

    def get_parent_id(self):
        return self.parent_id

    # Methods for MOORObotInterface class

    def iterate_generation(self):
        self.age += 1

    def needs_evaluation(self):
        return self.needs_eval

    def mutate(self):
        self.needs_eval = True
        self.parent_id = self.id
        self.fitness = -1000

    def get_minimize_vals(self):
        return [self.get_age()]

    def get_maximize_vals(self):
        return [self.fitness]

    def get_fitness(self, test=False):
        return self.fitness

    def get_summary_sql_columns(self):
        return "(id INT, parentId INT, age INT, fitnessTrain FLOAT, fitnessTest FLOAT)"

    def get_summary_sql_data(self):
        return (self.get_id(), self.get_parent_id(), self.get_age(), self.get_fitness(), self.get_fitness(test=True))

    def dominates_final_selection(self, other):
        return self.get_fitness() > other.get_fitness()

    # Methods for Work class
    def cpus_requested(self):
        return 1

    def compute_work(self, test=True, **kwargs):
        pass
        # import time
        #
        # time.sleep(0.01)

    def get_description_sql_columns(self):
        return "(id INT, test TEXT)"

    def get_description_sql_data(self):
        return (self.get_id(), "THIS IS A TEST%d"%self.get_fitness())

    def write_letter(self):
        return Letter((0, 0), None)

    def open_letter(self, letter):
        self.fitness = random.random()
        self.needs_eval = False
        return None


    def get_age(self):
        return self.age
