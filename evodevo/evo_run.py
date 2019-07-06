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

import os
import pickle
import random
import time
from subprocess import call, check_output

import numpy as np

from evodevo.afpomoo import AFPOMoo, MOORobotInterface
from evodevo.utils import print_utils
from evodevo.utils.print_utils import print_all


class EvolutionaryRun(object):
    def __init__(self, robot_factory, gens, seed, pop_size=75, experiment_name="", source_code_path="."):
        assert isinstance(robot_factory(), MOORobotInterface)

        self.source_code_path = source_code_path  # used for logging git info.

        # make directory for current evo run.
        self.runDir = "run_%d" % seed
        self.robotDir = "BestRobots"
        self.datDir = "Data"

        if not self.create_directory(delete=False):
            # was the job running?
            if os.path.exists("%s/RUNNING" % self.runDir) or os.path.exists("%s/DONE" % self.runDir):
                print_all("attempting to load from checkpoint")
                if self.load_checkpoint():
                    return
            self.create_directory(delete=True)
            self.saved_robots = {}
            self.num_gens = gens
            self.current_gen = 0
            self.seed = seed
            self.randRandState = None
            self.numpyRandState = None
            self.messages_file = None
            self.experiment_name = experiment_name

            self.afpo_algorithm = AFPOMoo(robot_factory, pop_size=pop_size)  # , messages_file=self.messages_file)

    def create_directory(self, delete=False):

        if os.path.isdir(self.runDir):
            if delete and False:
                call(("rm -rf %s" % self.runDir).split())
                print_all("Deleting directory %s/" % self.runDir)
            else:
                return False
        os.mkdir(self.runDir)
        os.mkdir("%s/%s" % (self.runDir, self.robotDir))
        os.mkdir("%s/%s" % (self.runDir, self.datDir))
        call(("touch %s/RUNNING" % self.runDir).split())

        git_commit_hash = get_git_hash(source_code_path=self.source_code_path)
        call(("touch %s/GITHASH_%s" % (self.runDir, git_commit_hash)).split())

        if git_commit_hash == "UNKNOWN":
            print_all("Evo Run starting with unknown version.")
            print_all("To enable storing the git commit of the code, "
                      "please use git for version control and pass the git repo directory to the constructor."
                      " For Example, EvolutionaryRun(source_code_path='path/to/your/code')")
        else:
            print_all("Evo Run starting with git commit %s" % git_commit_hash)

    def init(self):
        """
        loads needed files in for running evolutionary run
        :return:
        """

        if self.messages_file is None:
            self.messages_file = open("%s/messages_"%(self.runDir) + str(self.seed) + ".txt", "a")
            print_utils.setup(log_file=self.messages_file)

    def cleanup_files(self):
        """
        closes files for pickling
        :return:
        """

        # messages cleanup_files
        self.messages_file.close()
        self.messages_file = None
        print_utils.cleanup()

    def cleanup_mpi(self):
        self.afpo_algorithm.cleanup()

    def cleanup_all(self):
        """
        cleans up files and cleans up mpi
        """
        call(("rm %s/RUNNING" % self.runDir).split())
        call(("touch %s/DONE" % self.runDir).split())
        self.cleanup_files()
        self.cleanup_mpi()

    def do_generation(self, printing=False):
        t0 = time.time()
        if os.path.exists("%s/MORE" % self.runDir):
            call(("rm %s/MORE" % self.runDir).split())
            self.num_gens += 500
        self.current_gen += 1
        if printing:
            print_all("generation %d" % (self.current_gen,))

        dom_data = self.afpo_algorithm.generation()

        if printing:
            print_all("%d individuals are dominating" % (dom_data[0],))
            print_all(dom_data[1])

        best = self.afpo_algorithm.get_best()

        if printing:
            # print_all("age: %f fit: %f" % (best[1], best[0]), self.messages_file)
            print_all(best[1])

        self.save_data(best[1])

        self.create_checkpoint()
        t1 = time.time()
        print_all("Generation took: %f"%(t1-t0))

    def run_full(self, printing=False):
        """
        Does a complete evolutionary run
        """
        self.init()

        while self.current_gen < self.num_gens:
            self.do_generation(printing=printing)

        self.cleanup_all()

    def save_data(self, robot):
        if repr(robot) not in self.saved_robots:
            robot.write_self_description()
            self.saved_robots[repr(robot)] = 1
            # save the robot
            with open("%s/%s/%s.p" % (self.runDir, self.robotDir, str(robot.id)), "wb") as f:
                pickle.dump(robot, f)
            # save the info for the generation
            with open("%s/%s/%sGen.txt" % (self.runDir, self.datDir, self.current_gen), "w") as f:
                to_write = [str(self.seed), str(self.current_gen), str(robot.id)]
                to_write += [str(d) for d in robot.get_data()]
                f.write(', '.join(to_write))

    def create_checkpoint(self):
        self.randRandState = random.getstate()
        self.numpyRandState = np.random.get_state()
        tmp = self.messages_file
        self.messages_file = None
        with open("%s/%s/%sCheckpoint.p" % (self.runDir, self.datDir, self.current_gen), "wb") as f:
            pickle.dump(self, f)

        self.messages_file = tmp

    def load_checkpoint(self):
        gens_to_add = 0
        if os.path.exists("%s/DONE" % self.runDir):
            if os.path.exists("%s/MORE" % self.runDir):
                gens_to_add = 500
                call(("rm %s/DONE" % self.runDir).split())
                call(("rm %s/MORE" % self.runDir).split())
                call(("touch %s/RUNNING" % self.runDir).split())
            else:
                raise Exception("Evo run is already done. Please touch MORE to continue.")

        if os.path.isdir("%s/%s" % (self.runDir, self.datDir)):
            gen = 1
            last_tmp = -1
            tmp = -1

            while tmp is not None:
                try:
                    tmp = pickle.load(open("%s/%s/%sCheckpoint.p" % (self.runDir, self.datDir, gen), "rb"))
                    last_tmp = tmp
                    gen += 1
                except:
                    tmp = None
            if last_tmp == -1:
                print_all("Failed to load from checkpoint. No checkpoints found")
                return False
            self.setstate(last_tmp)
            current_git_commit_hash = get_git_hash(source_code_path=self.source_code_path)
            res = check_output(("ls %s" % self.runDir).split()).decode("utf-8").split()
            for n, b in enumerate(["GITHASH_" in itm for itm in res]):
                if b:
                    last_git_commit_hash = res[n][8:]
                    if last_git_commit_hash != current_git_commit_hash:
                        print_all("Failed to load from checkpoint. The git commit changed.")
                        call(("rm %s/GITHASH_*" % self.runDir).split())
                        return False
                    else:
                        break

            print_all("Successfully loaded checkpoint at gen %d" % self.current_gen)
            if gens_to_add > 0:
                print_all("Adding %d additional generations" % gens_to_add)
                self.num_gens += gens_to_add
            return True

        else:
            print_all("Failed to load from checkpoint. Data directory missing")
            return False

    def setstate(self, other):
        self.saved_robots = other.saved_robots
        self.num_gens = other.num_gens
        self.current_gen = other.current_gen
        self.seed = other.seed
        self.randRandState = other.randRandState
        self.numpyRandState = other.numpyRandState
        self.messages_file = None
        self.experiment_name = other.experiment_name
        self.afpo_algorithm = other.afpo_algorithm

        random.setstate(self.randRandState)
        np.random.set_state(self.numpyRandState)


def get_git_hash(source_code_path="."):
    """
    Attempts to find the git commit of the source code. if this fails, returns "UNKNOWN"
    :param source_code_path: The path to look at.
    :return: The commit hash. UNKNOWN if not found.
    """
    try:
        cmd_to_run = ("git -C %s log -n 1").split()
        cmd_to_run[2] = cmd_to_run[2] % source_code_path

        git_commit_hash = check_output(cmd_to_run).split()[1].decode("utf-8")
        return git_commit_hash
    except Exception as e:
        raise e
        return "UNKNOWN"
