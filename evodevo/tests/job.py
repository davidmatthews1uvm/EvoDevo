# Copyright 2019 David Matthews
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

import os
import random
import sys
from subprocess import call

from evodevo.evo_run import EvolutionaryRun

from parallelpy import parallel_evaluate
import numpy as np

sys.path.insert(0, "../..")

from evodevo.tests.softbot_robot_base import SoftbotRobot

parallel_evaluate.MAX_THREADS = 24
# parallel_evaluate.DEBUG=True

# check if we are running on the VACC. if so. disable debug mode.
if os.getenv("VACC") is not None:
    parallel_evaluate.DEBUG = False
    parallel_evaluate.MAX_THREADS = 24

np.set_printoptions(suppress=True, formatter={'float_kind': lambda x: '%4.2f' % x})


POP_SIZE = 100
MAX_GENS = 90


SEED = int(sys.argv[1])
np.random.seed(SEED)  # if we end up loading from checkpoint, that will set the state of the random number generators.
random.seed(SEED)
MAX_RUNTIME = float(sys.argv[2])
RUN_DIR = "run_{}".format(SEED)
RUN_NAME = "CiliaSwimmers"

parallel_evaluate.setup(parallel_evaluate.PARALLEL_MODE_MPI_INTER)  # need to do this AFTER all classes have been defined; or MPI will not work. Pool will though...


def robot_factory():
    internal_robot = None
    # print(internal_robot.phenotype.get_phenotype())
    return SoftbotRobot(internal_robot, "run_%d" % SEED)


def create_new_job():
    return EvolutionaryRun(robot_factory, MAX_GENS, SEED, pop_size=POP_SIZE,
                           experiment_name=RUN_NAME, override_git_hash_change=True, max_time=MAX_RUNTIME)




# initialize the evo_run. Create some directories.
evolutionary_run = create_new_job()
evolutionary_run.run_full(printing=True)
