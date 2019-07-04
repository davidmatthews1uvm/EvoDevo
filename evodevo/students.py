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

from evodevo.moo_interfaces import StudentInterface


class MOOStudent(StudentInterface):
    def dominates(self, other):
        """
        returns True if self dominates other
        :param other: the other Student to compare self to.
        :return: True if self dominates other, False otherwise.
        """
        self_min_traits = self.interface_bot.get_minimize_vals()
        self_max_traits = self.interface_bot.get_maximize_vals()

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
        return self.interface_bot.get_seq_num() < other.interface_bot.get_seq_num()
