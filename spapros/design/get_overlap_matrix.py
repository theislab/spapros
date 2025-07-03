############################################
# imports
############################################
import argparse
import fnmatch
import multiprocessing
import os
from functools import reduce

import pandas as pd

############################################
# pipeline
############################################


def args():

    args_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args_parser.add_argument("-ind", "--input_directories", help="path to config yaml file", type=str, required=True)
    args_parser.add_argument("-outd", "--output_directory", help="path to config yaml file", type=str, required=True)
    return args_parser.parse_args()


############################################


def get_files(dirs_in):

    files = []
    number_dirs = 0
    for dir in dirs_in:
        files.append(
            pd.DataFrame(
                [[file.split("probes_")[1].split(".")[0], file] for file in _list_files_in_dir(dir, r"probes_*")],
                columns=["gene", "file_dir{}".format(number_dirs)],
            )
        )
        number_dirs += 1
    files = reduce(lambda left, right: pd.merge(left, right, on=["gene"], how="outer"), files)
    files = files.loc[:, ~files.columns.duplicated()]

    return files


############################################


def _list_files_in_dir(dir, pattern):
    for root, _dirs, files, _rootfd in os.fwalk(dir, follow_symlinks=True):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


############################################


def get_overlap_matrix(files, dir_out):

    jobs = []
    for idx in files.index:
        files_gene = files.iloc[idx]
        for index, value in files_gene.items():
            if index == "gene":
                gene = value
                probes = []
            elif not pd.isna(value):
                probes.append(pd.read_csv(value, sep="\t"))
        probes = pd.concat(probes, axis=0, ignore_index=True)
        probes.to_csv(os.path.join(dir_out, "probes_{}.txt".format(gene)), sep="\t", index=False)

        proc = multiprocessing.Process(
            target=_compute_overlap_matrix,
            args=(
                gene,
                probes,
                dir_out,
            ),
        )
        jobs.append(proc)
        proc.start()

    for job in jobs:
        job.join()


############################################


def _compute_overlap_matrix(gene, probes, dir_out):
    probes["pid"] = ["{}_{}".format(probes.start[i], probes.end[i]) for i in probes.index]
    matrix = pd.DataFrame(0, columns=probes.pid, index=probes.pid)
    for i in probes.index:
        probe1_start = int(probes.loc[i, "start"])
        probe1_end = int(probes.loc[i, "end"])
        probe1_interval = [probe1_start, probe1_end]
        pid1 = "{}_{}".format(probe1_start, probe1_end)
        for j in probes.index:
            probe2_start = int(probes.loc[j, "start"])
            probe2_end = int(probes.loc[j, "end"])
            probe2_interval = [probe2_start, probe2_end]
            pid2 = "{}_{}".format(probe2_start, probe2_end)
            if _get_overlap(probe1_interval, probe2_interval):
                matrix.loc[pid1, pid2] = 1
                matrix.loc[pid2, pid1] = 1
            else:
                matrix.loc[pid1, pid2] = 0
                matrix.loc[pid2, pid1] = 0
            if j > i:
                break
    matrix.to_csv(os.path.join(dir_out, "overlap_matrix_{}.txt".format(gene)), sep="\t")


############################################


def _get_overlap(a, b):
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    return overlap > -1


############################################

if __name__ == "__main__":

    # get comman line arguments
    parameters = args()
    dirs_in = parameters.input_directories
    dirs_in = dirs_in.split(",")

    dir_out = parameters.output_directory

    files = get_files(dirs_in)
    get_overlap_matrix(files, dir_out)
