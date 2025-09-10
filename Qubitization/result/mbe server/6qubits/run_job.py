#!/usr/bin/env python3
import sys
from datetime import timedelta
from pathlib import Path
import os

import numpy as np
from qe_python.job_builder import JobBuilder
from qe_python.pseudos import atomic_species_from_positions
from qe_python.servers.burgundy import *
from qe_python.types import BANDSInput, PWInput
from qe_python.writer import QECommand


def JSFA(*args):
    return JSF(*args, timedelta(days=3), "special_mne_alicehu")


JS_QISKIT = JSFA(1, 1, 16)

def main():
    with JobBuilder(RA) as jobs:
        command = ['python3 run.py &> result.out']
        jobs.add_job(JS_QISKIT([], None, command).write('queue.sh'))
if __name__ == "__main__":
    main()
