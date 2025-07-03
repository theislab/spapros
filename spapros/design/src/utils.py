############################################
# imports
############################################
import gzip
import os
import shutil
from ftplib import FTP

import pybedtools
import yaml
from Bio.SeqUtils import MeltingTemp as mt

############################################
# helper functions
############################################


def get_config(config):
    """Load config file in yaml format.

    :param config: Path to yaml config file.
    :type config: string
    :return: User-defined parameters, where keys are the parameter names and values are the paremeter values.
    :rtype: dict
    """
    with open(config, "r") as ymlfile:
        return yaml.safe_load(ymlfile)


def print_config(config, logger):
    """Prints formatted config parameters as <parameter_name>: <parameter_value> to logging file.

    :param config: User-defined parameters, where keys are the parameter names and values are the paremeter values.
    :type config: dict
    :param logging: Logger object to store important information.
    :type logging: logging.basicConfig
    """
    logger.info("#########Parameter settings#########")
    for item, value in config.items():
        logger.info("{}: {}".format(item, value))


def create_dir(dir, subdir):
    """Create subdirectory and return path to subdirectory.

    :param dir: Path to main directory.
    :type dir: str
    :param subdir: Name of subdirectory.
    :type subdir: str
    :return: Pyth to subdirectory.
    :rtype: str
    """
    dir = os.path.join(dir, subdir)
    if not os.path.isdir(dir):
        os.mkdir(dir)

    return dir


def get_Tm_parameters(Tm_parameters):
    """Convert config parameters for 'table' attributes of MeltingTemp function into MeltingTemp attributes.

    :param Tm_parameters: Dictionary with parameters for MeltingTemp function.
    :type Tm_parameters: dict
    :return: Dictionary with parameters for MeltingTemp function.
    :rtype: dict
    """
    Tm_parameters["nn_table"] = getattr(mt, Tm_parameters["nn_table"])
    Tm_parameters["tmm_table"] = getattr(mt, Tm_parameters["tmm_table"])
    Tm_parameters["imm_table"] = getattr(mt, Tm_parameters["imm_table"])
    Tm_parameters["de_table"] = getattr(mt, Tm_parameters["de_table"])

    return Tm_parameters


def ftp_download(ftp_link, directory, file_name, dir_output):
    """Download file from ftp server.

    :param ftp_link: Link to ftp server, e.g. 'ftp.ncbi.nlm.nih.gov'.
    :type ftp_link: string
    :param directory: Directory on ftp server, where the file is located, e.g.
        'refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers'.
    :type directory: string
    :param file_name: Name of file that should be downloaded from ftp server, e.g.
        'GCF_000001405.39_GRCh38.p13_genomic.fna'.
    :type file_name: string
    :param dir_output: Path to directory for downloaded files.
    :type dir_output: string
    :return: Path to downloaded file.
    :rtype: string
    """
    ftp = FTP(ftp_link)
    ftp.login()  # login to ftp server
    ftp.cwd(directory)  # move to directory

    allfiles = []
    ftp.retrlines("LIST ", allfiles.append)

    for file in allfiles:
        if file_name in file:
            if " -> " in file:  # if file is a symbolic link
                file_download = file.split(" -> ")[1]
                raise ValueError(
                    "The provided file path is a symbolic link to {}. Please provide the actual path!".format(
                        file_download
                    )
                )
                # file_output = file_download.split('/')[-1]
                # ftp.retrbinary('RETR ' + file_download, open(os.path.join(dir_output, file_output), 'wb').write)
            else:
                file_download = file.split(" ")[-1]
                file_output = os.path.join(dir_output, file_download)
                ftp.retrbinary("RETR " + file_download, open(file_output, "wb").write)

    ftp.quit()

    return file_output


def decompress_gzip(file_gzip):
    """Decompress zip files.

    :param file_gzip: Path to zipped file.
    :type file_gzip: string
    :return: Path to unzipped file.
    :rtype: string
    """
    file_output = file_gzip.split(".gz")[0]
    with gzip.open(file_gzip, "rb") as f_in:
        with open(file_output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return file_output


def merge(file_bed):
    annotation = pybedtools.BedTool(file_bed)
    annotation = annotation.merge(
        d=0,
        s=True,
    )


def get_fasta(file_bed, file_reference_fasta, file_fasta, split=False):
    """Get sequence for regions annotated in input gft file using genome fasta file.

    :param file_bed: Path to bed file with annotated genomic regions.
    :type file_bed: string
    :param file_reference_fasta: Path to fasta file with reference sequence, e.g. transcriptome.
    :type file_reference_fasta: string
    :param file_fasta: Path to fasta file where retrieved sequences are written to.
    :type file_fasta: string
    :param split: Use -split option of bedtools getfasta, defaults to False
    :type split: bool
    """
    annotation = pybedtools.BedTool(file_bed)
    genome_sequence = pybedtools.BedTool(file_reference_fasta)

    annotation = annotation.sequence(fi=genome_sequence, s=True, name=True, split=split)
    annotation.save_seqs(file_fasta)
