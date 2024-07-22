#!/usr/bin/env python

import os
from argparse import ArgumentParser


def setup_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--analysis-script",
        type=str,
        help="path to the python script you want your SLURM job to run",
    )
    parser.add_argument(
        "--pe-inj-file",
        type=str,
        help="Label to identify this analysis with",
    )
    parser.add_argument(
        "--python-path",
        type=str,
        help="path to python, e.g: /home/USERNAME/.conda/envs/ENV/bin/python. ENV should be same as --env argument.",
    )
    parser.add_argument(
        "--result-dir", 
        type=str,  
        help="Path of output directory of analysis"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="gwinferno_analysis",
        help="Label to identify this analysis with",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="gwinferno_analysis",
        help="Label to identify this job with, should be short",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="Talapas partition to submit jobs to",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default="1",
        help="number of nodes to request",
    )
    parser.add_argument(
        "--ntasks-per-node",
        type=int,
        default="1",
        help="number of tasks to launch per node",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="gpu-80gb",
        help="desired constraint for requested resource",
    )
    parser.add_argument(
        "--rng-keys",
        type=str,
        default="1",
        help="rngkey for analysis script. Will be input of -a slurm argument, format ex: 1 or 1-3 or 2,4,6.",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=1440,
        help="Time limit of Job (in minutes) --"
        + "times longer than 1440 min need the long partitions, jobs with lower time limits can get started with higher priority",
    )
    parser.add_argument(
        "--mem", type=int, default=8192, help="Requested Memory for analysis (in Mb)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to request. If >1 then automatically sets num-cpus=0",
    )
    parser.add_argument(
        "--env",
        type=str,
        default='gwinferno_gpu',
        help="conda environment",
    )
    parser.add_argument(
        "--mmin",
        type=float,
        default=2.0,
        help="minimum BH mass",
    )
    parser.add_argument(
        "--mmax",
        type=float,
        default=200.0,
        help="maximum BH mass",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1000,
        help="number of samples to collect during inference",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=1000,
        help="number of warmup samples to collect during inference",
    )
    return vars(parser.parse_args())

def main():
    args = setup_parser()
    label = args["label"]
    job = args['job_name']
    outdir = args["result_dir"]
    script = args["analysis_script"]
    mem = args["mem"]
    time = args["time"]
    nodes = args['nodes']
    ntasks = args['ntasks_per_node']
    partition = args["partition"]
    ngpu = args["num_gpus"]
    rng = args['rng_keys']
    env = args['env']
    python = args['python_path']
    mmin = args['mmin']
    mmax = args['mmax']
    samples = args['nsamples']
    warmup = args['nwarmup']
    peinj = args['pe_inj_file']
    constraint = args['constraint']
    if time > 1440 and "long" not in partition:
        time = 1440
    base_submit_str = f"""#!/bin/bash
#SBATCH --partition={partition}       ### queue to submit to
#SBATCH --job-name={job}    ### job name
#SBATCH --output={outdir}/{label}/rng_%a.out   ### file in which to store job stdout
#SBATCH --error={outdir}/{label}/rng_%a.err    ### file in which to store job stderr
#SBATCH --time={time}          ### wall-clock time limit, in minutes
#SBATCH --mem={mem}M              ### memory limit, per cpu, in MB
#SBATCH --nodes={nodes}               ### number of nodes to use
#SBATCH --ntasks-per-node={ntasks}     ### number of tasks per node (ONLY USED FOR MPI LIKE PARALLELIZATION USE cpus-per-task for multithreading)
#SBATCH -a {rng}
#SBATCH --gres=gpu:{ngpu}
#SBATCH --constraint=\"{constraint}\"
"""
    base_submit_str += f"\nconda activate {env}"
    base_submit_str += (
        f"\n{python} {script} --samples {samples} --warmup {warmup} --rngkey $SLURM_ARRAY_TASK_ID --run-label {label} --result-dir {outdir} --mmin {mmin} --mmax {mmax} --pe-inj-file {peinj}"
    )
    base_submit_str += "\nconda deactivate"

    with open(f"submit_{job}_rng{rng}.sh", "w") as f:
        f.write(base_submit_str)

if __name__ == "__main__":
    main()
