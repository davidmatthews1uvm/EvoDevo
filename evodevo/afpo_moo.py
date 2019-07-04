# Copyright 2018 David Matthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import copy
import time
import random

from parallelpy.utils import Work
from parallelpy.parallel_evaluate import batch_complete_work
from parallelpy.parallel_evaluate import cleanup as _cleanup

from evodevo.moo_interfaces import MOORobotInterface, StudentInterface
from evodevo.students import MOOStudent as Student



class afpo_moo(object):
    def __init__(self, robot_factory, pop_size=50, messages_file = None):
        assert isinstance(robot_factory(), MOORobotInterface), 'robot_factory needs to produce robots which' \
                                                            'conform to the RobotInterface interface'

        self.messages_file = messages_file

        self.pop_size = pop_size
        self.robot_factory = robot_factory

        self.students = [None] * self.pop_size
        self.robot_id = 0
        self.initialize()

    def __str__(self):
        return "afpo population".join([str(s) for s in self.students])

    def initialize(self):
        for i in range(self.pop_size):
            self.students[i] = Student(self.robot_factory(), self.get_robot_id())

    def get_robot_id(self):
        self.robot_id += 1
        return self.robot_id


    def cleanup(self):
        _cleanup()

    def get_data_for_pickling(self):
        return self.students

    def _iterate_generation(self):
        # update generation dependent values of the students.
        for s in self.students:
            s.iterate_generation()

    def _evaluate_all(self):
        # get the robots to evaluate, store how many simulations each robot needs.
        students_to_evaluate = [s for s in self.students if s.get_robot().needs_evaluation()]
        robots_to_evaluate = [s.get_robot() for s in students_to_evaluate]

        batch_complete_work(robots_to_evaluate)


    def generation(self, debug=False):
        if debug: print("starting generation")
        # update the generation dependent behavioral_sem_error of the bots.
        self._iterate_generation()
        if debug: print("adding new student")
        # add a new Student even if the population already is full.
        new_student = Student(self.robot_factory(), self.get_robot_id())
        self.students.append(new_student)

        if debug: print("beginning reproduction")
        # expand the population.
        while len(self.students) < self.pop_size * 2:
            # if debug: print("|", end = '', flush=True)
            parent_index = random.randrange(0, self.pop_size)
            new_student = copy.deepcopy(self.students[parent_index])
            new_student.mutate()
            new_student.set_id(self.get_robot_id())
            self.students.append(new_student)

        if debug: print("beginning evaluation of robots")
        # evaluate all robots
        self._evaluate_all()

        numb_students = self.pop_size * 2

        dominating_individuals = 0
        dom_ind = []
        if debug: print("Calculating dominating individuals")
        # calculate real number of dominating individuals.
        for s in range(len(self.students)):
            dominated = False
            for t in range(len(self.students)):
                if self.students[t].dominates(self.students[s]):
                    dominated = True
                    break
            if (not dominated):
                dom_ind.append(self.students[s])
                dominating_individuals += 1

        if(debug): print("Starting pruning of population")
        while numb_students > max(self.pop_size, dominating_individuals):
            i1 = random.randrange(len(self.students))
            i2 = random.randrange(len(self.students))
            if i1 == i2:
                continue
            if self.students[i1] == None or self.students[i2] == None:
                continue
            if self.students[i1].dominates(self.students[i2]):
                self.students[i2] = None
                numb_students -= 1
        # compress the population
        self.students = [p for p in self.students if p != None]

        # print warnings if necessary
        if (dominating_individuals >= 2 * self.pop_size):
            print("WARNING: unable evolve! All individuals are dominating!")
        elif (dominating_individuals >= self.pop_size):
            print("WARNING: dominating frontier contains more than 100% of individuals in the population!",
                  dominating_individuals)
        elif (dominating_individuals * 0.75 >= self.pop_size):
            print("WARNING: dominating frontier contains more than 75% of individuals in the population!",
                  dominating_individuals)

        return (dominating_individuals, dom_ind)

    def get_all_bots(self):
        bots = [s.interface_bot for s in self.students if s != None]
        return bots

    def get_best(self):
        best_student = None
        for s in self.students:
            if best_student is None:
                best_student = s
            if s is not None and s.dominates_final_selection(best_student):
                best_student = s
        return (best_student.get_fitness(), best_student.get_robot())
