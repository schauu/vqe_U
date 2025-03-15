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


JS_QISKIT = JSFA(1, 1, 64)


def main():
    with JobBuilder(RA) as jobs:
        erates = [1e-3, 1e-4, 1e-5, 0]
        #nqubits = [4, 6, 8]
        nqubits = [10]
        depths = [1, 2, 3, 4, 5]
        os.system("rm slurm-*.out")
        for nqubit in nqubits:
            for depth in depths:
                path = Path(f"./{nqubit}_qubits/depth_{depth}")
                path.mkdir(parents=True, exist_ok=True)
                command = [f"../../run.py vqe {nqubit} {depth} &> vqe.{nqubit}_qubits.depth_{depth}.out"]
                jobs.add_job(JS_QISKIT([], None, command).write(path / f'queue_vqe_{nqubit}_qubits_depth_{depth}.sh'))
            # command = "echo $SLURM_JOB_ID"
            # jobs.add_job(JS_QISKIT([], None, command).write(path / f'queue.sh'))
            for erate in erates:
                path = Path(f"./{nqubit}_qubits/erate_{erate}")
                path.mkdir(parents=True, exist_ok=True)
                command = [f"../../run.py cosine {nqubit} {erate}&> cosine.{nqubit}_qubits.erate_{erate}.out"]
                jobs.add_job(JS_QISKIT([], None, command).write(path / f'queue_cosine_{nqubit}_qubits_erate_{erate}.sh'))
                


if __name__ == "__main__":
    main()
