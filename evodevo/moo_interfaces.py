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


class RobotInterface(Work):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_id(self, new_id): raise NotImplementedError

    @abstractmethod
    def get_id(self):
        """
        Return a unique identifer for this robot. Must be compareable to other robot ids.
        This is used as a tiebreaker in multi-objective optimization.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def iterate_generation(self):
        """
        This method will be called on each robot in the population every generation.
        If you implementing AFPO, then use this to update the age.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def needs_evaluation(self):
        """
        :return: True if you need to be evaluted, false otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def mutate(self):
        """
        Make some mutations. You decide what mutations to make.
        :return: None
       """
        raise NotImplementedError

    def get_summary_sql_columns(self): raise NotImplementedError

    @abstractmethod
    def get_summary_sql_data(self): raise NotImplementedError

    @abstractmethod
    def get_description_sql_columns(self):
        """
        If you want to disable the RobotDesc table, then return None here.
        :return:
        """
        return None

    def get_description_sql_data(self):
        """
        When saving the best robots, we sometimes might want them to export a version of themselves which can be easily opened for later viewing.\
        Although we can always recreate things from the pickles, this will be faster sometimes.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def dominates(self, other): raise NotImplementedError

    @abstractmethod
    def dominates_final_selection(self, other): raise NotImplementedError

    @abstractmethod
    def get_fitness(self): raise NotImplementedError

class MOORobotInterface(RobotInterface):
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_maximize_vals(self): raise NotImplementedError

    @abstractmethod
    def get_minimize_vals(self): raise NotImplementedError

    @abstractmethod
    def get_seq_num(self): return self.get_id()

    def dominates(self, other):
        """
        returns True if self dominates other
        :param other: the other Student to compare self to.
        :return: True if self dominates other, False otherwise.
        """
        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.get_minimize_vals()
        other_max_traits = other.get_maximize_vals()

        # all min traits must be at least as small as corresponding min traits
        if list(filter(lambda x: x[0] > x[1], zip(self_min_traits, other_min_traits))):
            return False

        # all max traits must be at least as large as corresponding max traits
        if list(filter(lambda x: x[0] < x[1], zip(self_max_traits, other_max_traits))):
            return False

        # any min trait smaller than other min trait
        if list(filter(lambda x: x[0] < x[1], zip(self_min_traits, other_min_traits))):
            return True

        # any max trait larger than other max trait
        if list(filter(lambda x: x[0] > x[1], zip(self_max_traits, other_max_traits))):
            return True

        # all fitness values are the same, default to return False.
        return self.get_seq_num() < other.get_seq_num()



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

