############################################
# imports
############################################
import argparse
import logging
import os
import time
from datetime import datetime

from src.datamodule import DataModule
from src.probefilter import ProbeFilter
from src.utils import get_config
from src.utils import print_config

timestamp = datetime.now()
logging.basicConfig(
    filename="log_probe_design_{}-{}-{}-{}-{}.txt".format(
        timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute
    ),
    level=logging.NOTSET,
)


############################################
# functions
############################################


def args():
    """Argument parser for command line arguments.

    :return: Command line arguments with their values.
    :rtype: dict
    """
    args_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args_parser.add_argument("-c", "--config", help="path to config yaml file", type=str, required=True)
    args_parser.add_argument("-o", "--output", help="path of output folder", type=str, required=True)
    return args_parser.parse_args()


def probe_pipeline(config, dir_output):
    """Pipeline of probe designer. Sets up all required directories; loads annotations, genes and probes;
    and filters probes based on sequence properties and blast aligment search results.

    :param config: User-defined parameters, where keys are the parameter names and values are the paremeter values.
    :type config: dict
    :param dir_output: User-defined output directory.
    :type dir_output: string
    """
    dir_output = os.path.join(dir_output, "")
    logging.info("Results will be saved to: {}".format(dir_output))

    datamodule = DataModule(config, logging, dir_output)

    t = time.time()
    datamodule.load_annotations()
    datamodule.load_genes()
    datamodule.load_transcriptome()
    datamodule.load_probes()
    t = (time.time() - t) / 60

    probefilter = ProbeFilter(config, logging, dir_output, datamodule.file_transcriptome_fasta, datamodule.genes)
    del datamodule  # free memory

    logging.info("Time to load annotations, genes, transcriptome and probes: {} min".format(t))
    print("Time to load annotations, genes, transcriptome and probes: {} min \n".format(t))

    t = time.time()
    probefilter.filter_probes_by_exactmatch()
    t = (time.time() - t) / 60

    logging.info("Time to filter with extact matches: {} min".format(t))
    print("Time to filter with extact matches: {} min \n".format(t))

    t = time.time()
    probefilter.run_blast_search()
    t = (time.time() - t) / 60

    logging.info("Time to run Blast search: {} min".format(t))
    print("Time to run Blast search: {} min \n".format(t))

    t = time.time()
    probefilter.filter_probes_by_blast_results()
    t = (time.time() - t) / 60

    logging.info("Time to filter with Blast results: {} min".format(t))
    print("Time to filter with Blast results: {} min \n".format(t))

    probefilter.log_statistics(rm_intermediate_files=True)


############################################
# main
############################################

if __name__ == "__main__":
    """Main function of probe designer."""
    # get comman line arguments
    parameters = args()

    dir_output = parameters.output
    config = get_config(parameters.config)
    print_config(config, logging)

    logging.info("#########Start Pipeline#########")
    print("#########Start Pipeline#########")

    t_pipeline = time.time()
    probe_pipeline(config, dir_output)
    t_pipeline = (time.time() - t_pipeline) / 60

    logging.info("Time Pipeline: {} min".format(t_pipeline))
    logging.info("#########End Pipeline#########")

    print("Time Pipeline: {} min \n".format(t_pipeline))
    print("#########End Pipeline#########")
