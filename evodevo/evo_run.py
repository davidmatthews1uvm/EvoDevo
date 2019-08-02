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
import shutil
import pickle
import random
import sqlite3
import time
from subprocess import call, check_output

import numpy as np

from evodevo.afpomoo import AFPOMoo
from evodevo.moo_interfaces import RobotInterface
from evodevo.utils import print_utils
from evodevo.utils.print_utils import print_all


class EvolutionaryRun(object):
    def __init__(self, robot_factory, gens, seed, pop_size=75, experiment_name="", source_code_path=".", override_git_hash_change=False, max_time=None):
        example_bot = robot_factory()
        assert isinstance(example_bot, RobotInterface)

        self.source_code_path = source_code_path  # used for logging git info.

        # make directory for current evo run.
        self.runDir = "run_%d" % seed
        # self.best_robot_dir = "BestRobots"
        # self.all_robot_dir = "AllRobots"
        # self.datDir = "Data"
        self.start_time = time.time()
        self.max_time = max_time

        if not self.create_directory(delete=False):
            # was the job running?
            if os.path.exists("%s/RUNNING" % self.runDir) or os.path.exists("%s/DONE" % self.runDir):
                print_all("Attempting to load from a checkpoint")
                if self.load_checkpoint(override_git_hash_change):
                    return
            self.create_directory(delete=True)

        self.saved_robots = {}
        self.num_gens = gens
        self.data_column_cnt = None
        self.current_gen = 0
        self.seed = seed
        self.randRandState = None
        self.numpyRandState = None
        self.messages_file = None
        self.experiment_name = experiment_name

        # set up the Database
        self.con = sqlite3.connect("%s/database.db" % self.runDir)
        self.cur = self.con.cursor()
        self.robot_description_table_enabled = False
        self.setup_db(example_bot)

        self.afpo_algorithm = AFPOMoo(robot_factory, pop_size=pop_size)  # , messages_file=self.messages_file)

    def setup_db(self, example_bot):
        # create the database if needed.

        robot_summary_columns = example_bot.get_summary_sql_columns()
        assert "id INT" in robot_summary_columns, "Robots table must have an id field!"
        self.cur.execute("CREATE TABLE IF NOT EXISTS Robots %s" % robot_summary_columns)
        self.cur.execute("CREATE INDEX IF NOT EXISTS summaryRobotIndex ON Robots (id)")

        robot_desc_columns = example_bot.get_description_sql_columns()
        if robot_desc_columns is not None:
            self.robot_description_table_enabled = True
            assert "id INT" in robot_desc_columns, "RobotsDesc table must have an id field!"
            self.cur.execute("CREATE TABLE IF NOT EXISTS RobotsDesc %s" % robot_desc_columns)
            self.cur.execute("CREATE INDEX IF NOT EXISTS descriptionRobotIndex ON RobotsDesc (id)")

        self.cur.execute("CREATE TABLE IF NOT EXISTS RobotsRaw (id INT, info BLOB)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS pickledRobotIndex ON RobotsRaw (id)")

        self.cur.execute("CREATE TABLE IF NOT EXISTS Generations (generation INT, robot INT)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS genIndex ON Generations (generation)")

        self.cur.execute("CREATE TABLE IF NOT EXISTS Checkpoints (generation INT, checkpoint BLOB)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS checkpointIndex ON Checkpoints (generation)")

    def create_directory(self, delete=False):

        if os.path.isdir(self.runDir):
            if delete:
                shutil.rmtree(self.runDir, ignore_errors=True)
                # call(("rm -rf %s" % self.runDir).split())
                print_all("Deleting directory %s/" % self.runDir)
            else:
                return False

        os.makedirs(self.runDir, exist_ok=True)

        # removing support for individual file creation.
        # os.mkdir("%s/%s" % (self.runDir, self.best_robot_dir))
        # os.mkdir("%s/%s" % (self.runDir, self.all_robot_dir))
        # os.mkdir("%s/%s" % (self.runDir, self.datDir))

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
        return True

    def init(self):
        """
        loads needed files in for running evolutionary run
        :return:
        """

        if self.messages_file is None:
            self.messages_file = open("%s/messages_%s.txt" % (self.runDir, str(self.seed)), "a")
            print_utils.setup(log_file=self.messages_file)

    def cleanup_files(self):
        """
        closes files for pickling
        :return:
        """

        # messages cleanup_files
        if self.messages_file is not None:
            self.messages_file.close()
            self.messages_file = None
        print_utils.cleanup()

    def cleanup_mpi(self):
        self.afpo_algorithm.cleanup()

    def cleanup_all(self, done=False):
        """
        cleans up files and cleans up mpi
        """
        if done:
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

        self.save_data(best[1], best=True)

        all_bots = self.afpo_algorithm.get_all_bots()
        for s in all_bots:
            self.save_data(s)
        self.create_checkpoint()
        self.con.commit()
        t1 = time.time()
        print_all("Generation took: %f" % (t1 - t0))

    def run_full(self, printing=False):
        """
        Does a complete evolutionary run
        """
        self.init()

        while self.current_gen < self.num_gens and self.is_time_remaining():
            self.do_generation(printing=printing)

        self.cleanup_all(done=self.is_time_remaining())
        # depricating old system for saving individual files.
        # if self.is_time_remaining():
        #     self.stitch()

    def save_data(self, robot, best=False):
        if robot.get_id() not in self.saved_robots:
            # log that this robot has been saved. We don't need to re-save it.
            self.saved_robots[robot.get_id()] = 1

            # removing support for storing data in individual files.
            # # save in main file system
            # with open("%s/%s/%d.txt" % (self.runDir, dir, robot.get_id()), "w") as f:
            #     f.write(robot_description)
            #
            # # save the robot
            # with open("%s/%s/%s.p" % (self.runDir, dir, str(robot.id)), "wb") as f:
            #     pickle.dump(robot, f)

            # save in database

            num_fields = len(robot.get_summary_sql_columns().split(","))
            summaryMask = "(" + ", ".join(["?"]*num_fields) + ")"
            self.cur.execute("INSERT INTO Robots VALUES %s"%summaryMask, robot.get_summary_sql_data())
            self.cur.execute("INSERT INTO RobotsRaw VALUES (?, ?)", (robot.get_id(), pickle.dumps(robot)))

            if self.robot_description_table_enabled:
                num_fields = len(robot.get_description_sql_columns().split(","))
                summaryMask = "(" + ", ".join(["?"] * num_fields) + ")"
                self.cur.execute("INSERT INTO RobotsDesc VALUES %s"%summaryMask, robot.get_description_sql_data())

        # save best robot
        if best:
            # removing creation of individual files.
            # with open("%s/%s/%sGen.txt" % (self.runDir, self.datDir, self.current_gen), "w") as f:
            #     to_write = [str(self.seed), str(self.current_gen), str(robot.id)]
            #     if self.data_column_cnt is None:
            #         tmp = robot.get_sql_data()
            #         self.data_column_cnt = len(tmp)
            #         to_write += [str(d) for d in tmp]
            #     else:
            #         to_write += [str(d) for d in robot.get_sql_data()]
            #     f.write(', '.join(to_write))

            self.cur.execute("INSERT INTO Generations VALUES (?, ?)", (self.current_gen, robot.get_id()))

    def create_checkpoint(self):
        self.randRandState = random.getstate()
        self.numpyRandState = np.random.get_state()
        tmp = self.messages_file
        self.messages_file = None
        tmp_cur = self.cur
        self.cur = None
        tmp_con = self.con
        self.con = None

        # removing support for creation of individual files.
        # with open("%s/%s/%sCheckpoint.p" % (self.runDir, self.datDir, self.current_gen), "wb") as f:
        #     pickle.dump(self, f)

        tmp_cur.execute("INSERT INTO Checkpoints VALUES (?, ?)", (self.current_gen, pickle.dumps(self)))
        self.con = tmp_con
        self.cur = tmp_cur
        self.messages_file = tmp

    def load_checkpoint(self, override_git_hash_change=False):
        """
        :param override_git_hash_change: If the git commit hash is different, do we want to re-start the evo run? or do we want to just continue?
        :return: True if successfully loaded the checkpoint. False if an error occured.
        """
        gens_to_add = 0
        if os.path.exists("%s/DONE" % self.runDir):
            if os.path.exists("%s/MORE" % self.runDir):
                gens_to_add = 500
                call(("rm %s/DONE" % self.runDir).split())
                call(("rm %s/MORE" % self.runDir).split())
                call(("touch %s/RUNNING" % self.runDir).split())
            else:
                print("Evo run is already done. Please touch MORE to continue.")
                exit(0)

        if os.path.isdir("%s" % self.runDir):
            # is there a database file?
            if not os.path.isfile("%s/database.db" % self.runDir):
                print_all("Database not found.\nStarting from scratch.")
                return False
            try:
                # try to connect to the database
                self.con = sqlite3.connect("%s/database.db" % self.runDir)
                self.cur = self.con.cursor()

                # get the newest checkpoint and attempt to load it in.
                self.cur.execute("SELECT (checkpoint) FROM Checkpoints ORDER BY Generation DESC limit 1")
                res = self.cur.fetchone()
                if res is None:
                    print_all("No checkpoints were found.\nStarting from scratch.")
                    return False
                candidate_checkpoint = pickle.loads(res[0])
                self.setstate(candidate_checkpoint)


                current_git_commit_hash = get_git_hash(source_code_path=self.source_code_path)
                res = check_output(("ls %s" % self.runDir).split()).decode("utf-8").split()
                for n, b in enumerate(["GITHASH_" in itm for itm in res]):  # mask out for the Git hash.
                    if b:
                        last_git_commit_hash = res[n][8:]
                        if last_git_commit_hash != current_git_commit_hash:
                            if override_git_hash_change:
                                print_all("The git commit changed. Restart overridden; continuing.")
                                call(("touch %s/GITHASH_%s" % (self.runDir, current_git_commit_hash)).split())
                                break
                            print_all("Failed to load from checkpoint. The git commit changed.\nStarting from scratch.")
                            call(("rm %s/GITHASH_*" % self.runDir).split())
                            return False
                        else:
                            break
                print_all("Successfully loaded checkpoint at gen %d" % self.current_gen)
                if gens_to_add > 0:
                    print_all("Adding %d additional generations" % gens_to_add)
                    self.num_gens += gens_to_add
                return True

            except Exception as e:
                print_all("Unable to load checkpoint from Database.")
                print_all("Error was: %s" % e)
                print_all("Starting from scratch.")
                return False

            # Removing individual file logging
            # gen = 1
            # last_tmp = -1
            # tmp = -1
            #
            # while tmp is not None:
            #     try:
            #         tmp = pickle.load(open("%s/%s/%sCheckpoint.p" % (self.runDir, self.datDir, gen), "rb"))
            #         last_tmp = tmp
            #         gen += 1
            #     except:
            #         tmp = None
            # if last_tmp == -1:
            #     print_all("Failed to load from checkpoint: No checkpoints found.\nStarting from scratch.")
            #     return False
            # self.setstate(last_tmp)
            # current_git_commit_hash = get_git_hash(source_code_path=self.source_code_path)
            # res = check_output(("ls %s" % self.runDir).split()).decode("utf-8").split()
            # for n, b in enumerate(["GITHASH_" in itm for itm in res]):
            #     if b:
            #         last_git_commit_hash = res[n][8:]
            #         if last_git_commit_hash != current_git_commit_hash:
            #             if override_git_hash_change:
            #                 print_all("The git commit changed. Restart overridden; continuing.")
            #                 call(("touch %s/GITHASH_%s" % (self.runDir, current_git_commit_hash)).split())
            #                 break
            #             print_all("Failed to load from checkpoint. The git commit changed.\nStarting from scratch.")
            #             call(("rm %s/GITHASH_*" % self.runDir).split())
            #             return False
            #         else:
            #             break
            #
            # print_all("Successfully loaded checkpoint at gen %d" % self.current_gen)
            # if gens_to_add > 0:
            #     print_all("Adding %d additional generations" % gens_to_add)
            #     self.num_gens += gens_to_add
            # return True

        else:
            print_all("Failed to load from checkpoint. Run directory missing.\nStarting from scratch.")
            return False

    def setstate(self, other):
        self.saved_robots = other.saved_robots
        self.data_column_cnt = other.data_column_cnt
        self.num_gens = other.num_gens
        self.current_gen = other.current_gen
        self.seed = other.seed
        self.randRandState = other.randRandState
        self.numpyRandState = other.numpyRandState
        self.messages_file = None
        self.experiment_name = other.experiment_name
        self.afpo_algorithm = other.afpo_algorithm
        self.robot_description_table_enabled = other.robot_description_table_enabled

        random.setstate(self.randRandState)
        np.random.set_state(self.numpyRandState)
        # self.con = sqlite3.connect("%s/database.db" % self.runDir)
        # self.cur = self.con.cursor()

    # Depricating old system for saving individual files
    # def stitch(self):
    #     """
    #     Deprecated. Use the database for this.
    #     stitch the generation logs together into one CSV file.
    #     :return: None
    #     """
    #     return None
    #     # try:
    #     #     out_file = open("%s/Gens.txt" % self.runDir, "w")
    #     #     out_file.write("Seed,Gen,UUID,fit,fit_test, age," + ','.join(["Data_%d" % n for n in range(self.data_column_cnt)]) + "\n")
    #     #     for i in range(1, self.num_gens):
    #     #         with open("%s/%s/%sGen.txt" % (self.runDir, self.datDir, i), "r") as f:
    #     #             out_file.write(f.read() + "\n")
    #     #     out_file.close()
    #     # except Exception as e:
    #     #     print("Failed to stitch the data together.")
    #     #     print(str(e))

    def is_time_remaining(self):
        if self.max_time is None:
            return True

        cur_time = time.time()

        run_time = cur_time - self.start_time
        run_time_hrs = run_time / 3600

        buffer_time = 0.25  # hours // 15 mins

        if run_time_hrs > self.max_time - buffer_time:
            return False
        return True


def get_git_hash(source_code_path="."):
    """
    Attempts to find the git commit of the source code. if this fails, returns "UNKNOWN"
    NOTE: -C is only available in newer versions of git. Please install a newer version of git if needed.

    :param source_code_path: The path to look at.
    :return: The commit hash. UNKNOWN if not found.
    """
    try:
        cmd_to_run = ("git -C %s log -n 1").split()
        cmd_to_run[2] = cmd_to_run[2] % source_code_path

        git_commit_hash = check_output(cmd_to_run).split()[1].decode("utf-8")
        return git_commit_hash
    except Exception as e:
        print(e)
        return "UNKNOWN"
