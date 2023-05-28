from argparse import ArgumentParser


def load_base_parser():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/bruce.edelman/projects/GWTC3_allevents/")
    parser.add_argument(
        "--inj-file",
        type=str,
        default="/home/bruce.edelman/projects/GWTC3_allevents/o1o2o3_mixture_injections.hdf5",
    )
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--mmin", type=float, default=3.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    return parser
