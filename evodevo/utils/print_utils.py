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

from io import IOBase

global logging_file
logging_file = None


def setup(logging_file_str=None, log_file=None):
    """
    If given a logging_file_str, Opens logging_file_str in append mode for writting log messages to it.
    Else, if given a log_file, saves a reference to it.
    :param logging_file_str:  Filename of file to start logging to.
    :param log_file:  file object to log to.
    :return: None
    """
    global logging_file
    assert (logging_file_str is None or log_file is None), "Either logging_file_str or log_file must be None."
    assert (logging_file_str is not None or log_file is not None),\
        "One of logging_file_str or log_file must be not None."

    if logging_file_str is not None:
        logging_file = open(logging_file_str, "a")
    elif log_file is not None:
        assert isinstance(log_file, IOBase), "log_file must be of type IOBase"
        logging_file = log_file


def cleanup():
    """
    Closes logging_file.
    :return: None
    """
    global logging_file
    logging_file.close()
    logging_file = None


def print_all(*strings, sep=" ", end="\n", flush=True):
    global logging_file
    print(*strings, sep=sep, end=end, flush=flush)
    if logging_file is not None:
        logging_file.write(sep.join([str(s) for s in strings]) + end)
        if flush:
            logging_file.flush()
