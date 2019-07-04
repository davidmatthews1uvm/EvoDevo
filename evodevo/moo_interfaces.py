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
from abc import ABCMeta, abstractmethod

from parallelpy.utils import Work


class MOORobotInterface(Work):
    __metaclass__ = ABCMeta

    @abstractmethod
    def iterate_generation(self): raise NotImplementedError

    @abstractmethod
    def needs_evaluation(self): raise NotImplementedError

    @abstractmethod
    def mutate(self): raise NotImplementedError

    @abstractmethod
    def get_maximize_vals(self): raise NotImplementedError

    @abstractmethod
    def get_minimize_vals(self): raise NotImplementedError

    @abstractmethod
    def get_seq_num(self): raise NotImplementedError

    @abstractmethod
    def get_fitness(self): raise NotImplementedError

    @abstractmethod
    def get_data(self): raise NotImplementedError

    @abstractmethod
    def get_data_column_count(self): pass

    @abstractmethod
    def dominates_final_selection(self, other): raise NotImplementedError


class AFPORobotInterface(MOORobotInterface):
    __metaclass__ = ABCMeta

    def __init__(self, optimize_mode="fitness"):
        self.age = 0
        self.optimize_mode = optimize_mode

    def iterate_generation(self):
        self.age += 1

    def get_maximize_vals(self):
        if self.optimize_mode == "fitness":
            return [self.get_fitness()]
        elif self.optimize_mode == "error":
            return []

    def get_minimize_vals(self):
        if self.optimize_mode == "error":
            return [self.age, self.get_fitness()]
        elif self.optimize_mode == "fitness":
            return [self.age]

    def get_age(self):
        return self.age


class StudentInterface(object):
    __metaclass__ = ABCMeta

    def __init__(self, robot: MOORobotInterface, id):
        self.interface_bot = robot
        self._id = id

    def __deepcopy__(self, memodict={}):
        ret = copy.copy(self)
        ret.interface_bot = copy.deepcopy(self.interface_bot)
        return ret

    def __str__(self):
        return str(self.interface_bot)

    def __repr__(self):
        return repr(self.interface_bot)

    def set_id(self, id):
        self._id = id

    def get_robot(self):
        return self.interface_bot

    def get_fitness(self):
        return self.interface_bot.get_fitness()

    def get_data(self):
        return self.interface_bot.get_data()

    def mutate(self):
        self.interface_bot.mutate()

    def iterate_generation(self):
        self.interface_bot.iterate_generation()

    def needs_evaluation(self):
        return self.interface_bot.needs_evaluation()

    @abstractmethod
    def dominates(self, other): raise NotImplementedError

    @abstractmethod
    def dominates_final_selection(self, other):
        return self.interface_bot.dominates_final_selection(other)
