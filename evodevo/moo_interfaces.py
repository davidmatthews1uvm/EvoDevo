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

from abc import ABCMeta, abstractmethod

from parallelpy.utils import Work


class RobotInterface(Work):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_id(self):
        raise NotImplementedError

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

    @abstractmethod
    def get_fitness(self):
        """
        Get the fitness of the robot. This is only used for printing of the best robots.
        If you want, feel free to return error instead of fitness.
        :return: A float.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self):
        """
        Got data to save for post-processing? Return it here.
        :return: A list of length equal to the number of data columms you requested.
        """
        raise NotImplementedError

    def write_self_description(self):
        """
        When saving the best robots, we sometimes might want them to export a version of themselves which can be easily opened for later viewing.
        Although we can always recreate things from the pickles, this will be faster sometimes.
        :return: None
        """
        pass

    @abstractmethod
    def get_data_column_count(self):
        """
        How much data are you going to be logging? Return the length of the list you rwill return when we call get_data()
        :return: An int.
        """
        raise NotImplementedError

    @abstractmethod
    def dominates_final_selection(self, other):
        """
        This method is used to calculate the most fit individual, which is printed and saved seperately.
        :param other: The robot to compare to self.
        :return: True if self dominates other (for printing / saving purposes), False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def dominates(self, other, **kwargs):
        """
        :param other: The robot to compare to self.
        :param kwargs: If you implement a different evo algorithm, and want to make domination dynamic such that the population size is constant, then pass keyword args here.
        :return: True if self dominates other. False otherwise. This is used to determine evolution.
        """
        raise NotImplementedError


class MOORobotInterface(RobotInterface):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_maximize_vals(self):
        """
        Which objectives are we trying to maximize? Examples may include, fitness, etc.
        :return: List of objectives to maximize.
        """
        raise NotImplementedError

    @abstractmethod
    def get_minimize_vals(self):
        """
        Which objectives are we trying to minimize? Examples may inlcude, age, error, etc.
        :return:  List of objecties to minimize.
        """
        raise NotImplementedError

    @abstractmethod
    def get_seq_num(self):
        """
        Deprecated: Will be removed in a future release.
        Please also implement get_id() to prepare for the removal of this method.

        See info about get_id() in the super class.
        :return: int
        """
        raise NotImplementedError

    def get_id(self):
        """
        See note in get_seq_num()
        :return: int (usually)
        """
        return self.get_seq_num()

    def dominates(self, other, **kwargs):
        """
        returns True if self dominates other
        :param other: the other Student to compare self to.
        :param kwargs: Not used in MOO optimization.
        :return: True if self dominates other, False otherwise.
        """
        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.interface_bot.get_minimize_vals()
        other_max_traits = other.interface_bot.get_maximize_vals()

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
        return self.get_id() < other.get_id()


class EvoSoroRobotInterface(RobotInterface):
    @abstractmethod
    def get_objectives(self):
        """
        What objectives are we trying to optimize? Please return a ranked list of objectives where the first one is the most important... to the least important.

        Each objective should be a dictionary that looks like:
            {"name": str, "value": float, "maximize": bool}
        :return: List of objectives to maximize.
        """
        raise NotImplementedError

    @abstractmethod
    def get_seq_num(self):
        """
        Deprecated: Will be removed in a future release.
        Please also implement get_id() to prepare for the removal of this method.

        See info about get_id() in the super class.
        :return: int
        """
        raise NotImplementedError

    def get_id(self):
        """
        See note in get_seq_num()
        :return: int (usually)
        """
        return self.get_seq_num()

    def dominates(self, other, **kwargs):
        """
        returns True if self dominates other
        :param other: the other Student to compare self to.
        :param kwargs: Not used in MOO optimization.
        :return: True if self dominates other, False otherwise.
        """
        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.interface_bot.get_minimize_vals()
        other_max_traits = other.interface_bot.get_maximize_vals()

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
        return self.get_id() < other.get_id()


class EvoAlgorithm(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def cleanup(self):
        """

        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def generation(self):
        """

        :return: (int, list) number of dominating robots, list of them.
        """
        raise NotImplementedError

    @abstractmethod
    def get_best(self):
        """
        Get the best robot for printing / saving.
        :return:
        """
        raise NotImplementedError
