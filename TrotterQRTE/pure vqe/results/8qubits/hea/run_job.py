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
        os.system("rm slurm-*.out")
        path = Path(f".")
            #command = "echo $SLURM_JOB_ID"
            #jobs.add_job(JS_QISKIT([], None, command).write(path / f'queue.sh'))
        command = [f"./run.py &> vqe_hva.out"]
        jobs.add_job(JS_QISKIT([], None, command).write(path / f'queue_vqe_hva.sh'))
 

if __name__ == "__main__":
    main()
