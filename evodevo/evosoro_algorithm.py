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


import copy
import random

from parallelpy.parallel_evaluate import batch_complete_work
from parallelpy.parallel_evaluate import cleanup as _cleanup

from evodevo.moo_interfaces import RobotInterface, EvoAlgorithm
from evodevo.utils.print_utils import print_all


class EvoSoroEvoAlg(EvoAlgorithm):
    def __init__(self, robot_factory, pop_size=50, messages_file=None):
        assert isinstance(robot_factory(), RobotInterface), 'robot_factory needs to produce robots which' \
                                                               'conform to the RobotInterface interface'

        self.messages_file = messages_file

        self.pop_size = pop_size
        self.robot_factory = robot_factory

        self.students = [None] * self.pop_size
        self.initialize()

    def __str__(self):
        return "afpo population".join([str(s) for s in self.students])

    def initialize(self):
        for i in range(self.pop_size):
            self.students[i] = self.robot_factory()

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

    def generation(self):
        # update the generation dependent behavioral_sem_error of the bots.
        self._iterate_generation()

        # add a new Student even if the population already is full.
        new_student = self.robot_factory()
        self.students.append(new_student)

        # expand the population.
        while len(self.students) < self.pop_size * 2:
            parent_index = random.randrange(0, self.pop_size)
            new_student = copy.deepcopy(self.students[parent_index])
            new_student.mutate()
            self.students.append(new_student)

        # evaluate all robots
        self._evaluate_all()


        self.students, (num_dominating_inds, dominating_individuals)  = self.selection()

        # print warnings if necessary
        if num_dominating_inds >= 2 * self.pop_size:
            print_all("WARNING: unable evolve! All individuals are dominating!")
        elif num_dominating_inds >= self.pop_size:
            print_all("WARNING: dominating frontier contains more than 100% of individuals in the population!",
                  num_dominating_inds)
        elif num_dominating_inds * 0.75 >= self.pop_size:
            print_all("WARNING: dominating frontier contains more than 75% of individuals in the population!",
                  num_dominating_inds)

        return num_dominating_inds, dominating_individuals

    def selection(self):
        numb_students = self.pop_size * 2

        num_dominating_inds = 0
        dominating_individuals = []

        # calculate real number of dominating individuals.
        for s in range(len(self.students)):
            dominated = False
            for t in range(len(self.students)):
                if self.students[t].dominates(self.students[s]):
                    dominated = True
                    break
            if not dominated:
                dominating_individuals.append(self.students[s])
                num_dominating_inds += 1

        while numb_students > max(self.pop_size, num_dominating_inds):
            i1 = random.randrange(len(self.students))
            i2 = random.randrange(len(self.students))
            if i1 == i2:
                continue
            if self.students[i1] is None or self.students[i2] is None:
                continue
            if self.students[i1].dominates(self.students[i2]):
                self.students[i2] = None
                numb_students -= 1
        # compress the population
        return [p for p in self.students if p is not None], (num_dominating_inds, dominating_individuals)
        
    def get_all_bots(self):
        bots = [s.interface_bot for s in self.students if s is not None]
        return bots

    def get_best(self):
        best_student = None
        for s in self.students:
            if best_student is None:
                best_student = s
            if s is not None and s.dominates_final_selection(best_student):
                best_student = s
        return best_student.get_fitness(), best_student.get_robot()
