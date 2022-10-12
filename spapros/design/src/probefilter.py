############################################
# imports
############################################
import multiprocessing
import os

import iteration_utilities
import pandas as pd
import src.utils as utils
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast.Applications import NcbimakeblastdbCommandline
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

############################################
# probe filter class
############################################


class ProbeFilter:
    """This class is used to filter the set of all possible guides per gene based on:
    1. exact matches,
    2. results of blast alignment tool.
    For each gene the set of filtered probes is saved and a list of genes where no probes were left after filtering.

    :param config: User-defined parameters, where keys are the parameter names and values are the paremeter values.
    :type config: dict
    :param logging: Logger object to store important information.
    :type logging: logging.basicConfig
    :param dir_output: User-defined output directory.
    :type dir_output: string
    """

    def __init__(self, config, logging, dir_output, file_transcriptome_fasta, genes):
        """Constructor method"""
        # set logger
        self.logging = logging

        # set directory
        self.dir_output_annotations = utils.create_dir(dir_output, "annotations")
        self.dir_output_blast = utils.create_dir(dir_output, "blast")
        self.dir_output_probes = utils.create_dir(dir_output, "probes")

        # set parameters
        self.number_batchs = config["number_batchs"]
        self.num_threads_blast = config["num_threads"]
        self.word_size = config["word_size"]
        self.percent_identity = config["percent_identity"]
        self.min_alignment_length = config["probe_length"] * config["coverage"] / 100
        self.ligation_site = config["ligation_site"]
        self.ligation_site_start = config["probe_length"] // 2 - (self.ligation_site - 1)
        self.ligation_site_end = config["probe_length"] // 2 + self.ligation_site
        self.min_probes_per_gene = config["min_probes_per_gene"]
        self.file_transcriptome_fasta = file_transcriptome_fasta
        self.removed_genes = genes

        # initialize additional paremeters
        self.duplicated_sequences = None

    def filter_probes_by_exactmatch(self):
        """Probes with exact matches are filtered out.
        This process can be executed in a parallele fashion on a user-defined number of threads (defined in config file).
        """

        def _get_duplicated_sequences():
            """Get a list of probe sequences that have a exact match within the pool of all
            possible probe sequences for the list of input genes.

            :return: List of probe sequences with exact matches in the pool of probes.
            :rtype: list
            """
            sequences = []

            for batch_id in range(self.number_batchs):
                file_probe_sequence_batch = os.path.join(
                    self.dir_output_annotations, "probes_sequence_batch{}.txt".format(batch_id)
                )

                with open(file_probe_sequence_batch, "r") as handle:
                    sequences_batch = [line.rstrip() for line in handle]
                sequences.extend(sequences_batch)
                os.remove(file_probe_sequence_batch)

            duplicated_sequences = list(iteration_utilities.unique_everseen(iteration_utilities.duplicates(sequences)))
            print("Number of duplicated probe sequences: {}".format(len(duplicated_sequences)))

            return duplicated_sequences

        def _filter_probes_exactmatch(batch_id):
            """Remove sequences with exact matches within the pool of all possible probe sequences for the list of input genes.

            :param batch_id: Batch ID.
            :type batch_id: int
            """
            file_probe_info_batch = os.path.join(
                self.dir_output_annotations, "probes_info_batch{}.txt".format(batch_id)
            )
            file_probe_fasta_batch = os.path.join(self.dir_output_annotations, "probes_batch{}.fna".format(batch_id))
            batch_logger = os.path.join(self.dir_output_annotations, "logger_batch{}.txt".format(batch_id))

            probes_info = pd.read_csv(
                file_probe_info_batch,
                sep="\t",
                dtype={
                    "gene_id": str,
                    "transcript_id": str,
                    "exon_id": str,
                    "probe_sequence": str,
                    "chromosome": str,
                    "start": str,
                    "end": str,
                    "strand": str,
                    "GC_content": float,
                    "melting_temperature": float,
                },
            )
            os.remove(file_probe_info_batch)

            probes_info_filetred = probes_info[~probes_info.probe_sequence.isin(self.duplicated_sequences)]
            probes_info_filetred.reset_index(inplace=True, drop=True)
            probe_ids = [
                "{}_pid{}".format(probes_info_filetred.gene_id[probe_id], probe_id)
                for probe_id in range(len(probes_info_filetred.index))
            ]
            probes_info_filetred.insert(0, "probe_id", probe_ids)
            _write_probes(probes_info_filetred, file_probe_info_batch, file_probe_fasta_batch)

            with open(batch_logger, "a") as handle:
                handle.write("{}\n".format(len(probes_info_filetred.index)))

        def _write_probes(probes_info_filtered, file_probe_info_batch, file_probe_fasta_batch):
            """Save filtered probe information in tsv file. Save probe sequences as fasta file.

            :param probes_info_filtered: Dataframe with probe information, filtered based on sequence properties.
            :type probes_info_filtered: pandas.DataFrame
            :param file_probe_info_batch: Path to tsv file with probe infos.
            :type file_probe_info_batch: string
            :param file_probe_fasta_batch: Path to fast file with probe sequences.
            :type file_probe_fasta_batch: string
            """
            # save info table
            probes_info_filtered[
                [
                    "probe_id",
                    "probe_sequence",
                    "gene_id",
                    "transcript_id",
                    "exon_id",
                    "chromosome",
                    "start",
                    "end",
                    "strand",
                    "GC_content",
                    "melting_temperature",
                ]
            ].to_csv(file_probe_info_batch, sep="\t", index=False)

            # save sequence of probes in fasta format
            output = []
            for row in probes_info_filtered.index:
                header = probes_info_filtered.iloc[row, probes_info_filtered.columns.get_loc("probe_id")]
                sequence = Seq(probes_info_filtered.iloc[row, probes_info_filtered.columns.get_loc("probe_sequence")])
                output.append(SeqRecord(sequence, header, "", ""))

            with open(file_probe_fasta_batch, "w") as handle:
                SeqIO.write(output, handle, "fasta")

        # get list of exact matches in probes pool
        self.duplicated_sequences = _get_duplicated_sequences()

        # run filter with multiprocess
        jobs = []
        for batch_id in range(self.number_batchs):
            proc = multiprocessing.Process(target=_filter_probes_exactmatch, args=(batch_id,))
            jobs.append(proc)
            proc.start()

        print("\n {} \n".format(jobs))

        for job in jobs:
            job.join()

    def run_blast_search(self):
        """Run BlastN alignment tool to find regions of local similarity between sequences, where sequences are probes
        and transcripts.
        BlastN identifies the transcript regions where probes match with a certain coverage and similarity.
        """

        def _run_blast(batch_id):
            """Run BlastN alignment search for all probes of one batch.

            :param batch_id: Batch ID.
            :type batch_id: int
            """
            file_probe_fasta_batch = os.path.join(self.dir_output_annotations, "probes_batch{}.fna".format(batch_id))
            file_blast_batch = os.path.join(self.dir_output_blast, "blast_batch{}.txt".format(batch_id))

            cmd = NcbiblastnCommandline(
                query=file_probe_fasta_batch,
                db=self.file_transcriptome_fasta,
                outfmt="10 qseqid sseqid length qstart qend",
                out=file_blast_batch,
                word_size=self.word_size,
                perc_identity=self.percent_identity,
                num_threads=self.num_threads_blast,
            )
            out, err = cmd()

        # create blast database
        cmd = NcbimakeblastdbCommandline(input_file=self.file_transcriptome_fasta, dbtype="nucl")
        out, err = cmd()
        print("Blast database created.")

        # run blast with multi process
        jobs = []
        for batch_id in range(self.number_batchs):
            proc = multiprocessing.Process(target=_run_blast, args=(batch_id,))
            jobs.append(proc)
            proc.start()

        print("\n {} \n".format(jobs))

        for job in jobs:
            job.join()

    def filter_probes_by_blast_results(self):
        """Process the results from BlastN alignment search and filter probes based on the results."""

        def _process_blast_results(batch_id):
            """Process the output of the BlastN alignment search.

            :param batch_id: Batch ID.
            :type batch_id: int
            """

            probes_info = _load_probes_info(batch_id)
            blast_results = _read_blast_output(batch_id)
            # print(sys.getsizeof(blast_results))
            _filter_probes_blast(probes_info, blast_results)

        def _load_probes_info(batch_id):
            """Load filtered probe infomration from tsv file.

            :param batch_id: Batch ID.
            :type batch_id: int
            :return: Dataframe with probe information, filtered based on sequence properties.
            :rtype: pandas.DataFrame
            """
            file_probe_info_batch = os.path.join(
                self.dir_output_annotations, "probes_info_batch{}.txt".format(batch_id)
            )
            probes_info = pd.read_csv(
                file_probe_info_batch,
                sep="\t",
                dtype={
                    "gene_id": str,
                    "transcript_id": str,
                    "exon_id": str,
                    "probe_sequence": str,
                    "chromosome": str,
                    "start": str,
                    "end": str,
                    "strand": str,
                    "GC_content": float,
                    "melting_temperature": float,
                },
            )
            return probes_info

        def _read_blast_output(batch_id):
            """Load the output of the BlastN alignment search into a DataFrame and process the results.

            :param batch_id: Batch ID.
            :type batch_id: int
            :return: DataFrame with processed blast alignment search results.
            :rtype: pandas.DataFrame
            """
            file_blast_batch = os.path.join(self.dir_output_blast, "blast_batch{}.txt".format(batch_id))
            blast_results = pd.read_csv(
                file_blast_batch,
                header=None,
                sep=",",
                low_memory=False,
                names=["query", "target", "alignment_length", "query_start", "query_end"],
                engine="c",
                dtype={"query": str, "target": str, "alignment_length": int, "query_start": int, "query_end": int},
            )

            blast_results["query_gene_id"] = blast_results["query"].str.split("_pid").str[0]
            blast_results["target_gene_id"] = blast_results["target"].str.split("::").str[0]

            return blast_results

        def _filter_probes_blast(probes_info, blast_results):
            """Use the results of the BlastN alignement search to remove probes with high similarity,
            probe coverage and ligation site coverage based on user defined thresholds.

            :param probes_info: Dataframe with probe information, filtered based on sequence properties.
            :type probes_info: pandas.DataFrame
            :param blast_results: DataFrame with processed blast alignment search results.
            :type blast_results: pandas.DataFrame
            """
            blast_results_matches = blast_results[
                ~(blast_results[["query_gene_id", "target_gene_id"]].nunique(axis=1) == 1)
            ]
            blast_results_matches = blast_results_matches[
                blast_results_matches.alignment_length > self.min_alignment_length
            ]
            if self.ligation_site > 0:
                blast_results_matches = blast_results_matches[
                    blast_results_matches.query_start < self.ligation_site_start
                ]
                blast_results_matches = blast_results_matches[blast_results_matches.query_end > self.ligation_site_end]

            probes_with_match = blast_results_matches["query"].unique()
            probes_wo_match = blast_results[~blast_results["query"].isin(probes_with_match)]

            for gene_id in blast_results["query_gene_id"].unique():
                probes_wo_match_gene = probes_wo_match[probes_wo_match.query_gene_id == gene_id]
                probes_wo_match_gene = probes_wo_match_gene["query"].unique()

                if len(probes_wo_match_gene) > self.min_probes_per_gene:
                    # print(self.removed_genes)
                    # print(gene_id)
                    # self.removed_genes = self.removed_genes.remove(gene_id)
                    # print(self.removed_genes)
                    _write_output(probes_info, gene_id, probes_wo_match_gene)

            batch_logger = os.path.join(self.dir_output_annotations, "logger_batch{}.txt".format(batch_id))
            with open(batch_logger, "a") as handle:
                handle.write("{}\n".format(len(probes_wo_match["query"].unique())))

        def _write_output(probes_info, gene_id, probes_wo_match):
            """Write results of probe design pipeline to file and create one file with suitable probes per gene.

            :param probes_info: Dataframe with probe information, filtered based on sequence properties.
            :type probes_info: pandas.DataFrame
            :param gene_id: Gene ID of processed gene.
            :type gene_id: string
            :param probes_wo_match: List of suitable probes that don't have matches in the transcriptome.
            :type probes_wo_match: list
            """
            file_output = os.path.join(self.dir_output_probes, "probes_{}.txt".format(gene_id))
            valid_probes = probes_info[probes_info["probe_id"].isin(probes_wo_match)]
            valid_probes.to_csv(file_output, sep="\t", index=False)

        # create file where removed genes are saved
        file_removed_genes = os.path.join(self.dir_output_probes, "genes_with_insufficient_probes.txt")

        print("Process blast results.")

        jobs = []
        for batch_id in range(self.number_batchs):
            proc = multiprocessing.Process(target=_process_blast_results, args=(batch_id,))
            jobs.append(proc)
            proc.start()

        print("\n {} \n".format(jobs))

        for job in jobs:
            job.join()

        print("Blast filter done.")

        _, _, probe_files = next(os.walk(self.dir_output_probes))
        for probe_file in probe_files:
            gene_id = probe_file[len("probes_") : -len(".txt")]
            self.removed_genes.remove(gene_id)

        with open(file_removed_genes, "w") as output:
            for gene_id in self.removed_genes:
                output.write("{}\n".format(gene_id))

    def log_statistics(self, rm_intermediate_files=True):
        """Log some statistics on probes and used disk space.

        :param rm_intermediate_files: Should intermediate result files be deleted, defaults to True
        :type rm_intermediate_files: bool, optional
        """
        statistics = pd.DataFrame(columns=["total", "GC_Tm", "exact_match", "blast"])
        for batch_id in range(self.number_batchs):
            batch_logger = os.path.join(self.dir_output_annotations, "logger_batch{}.txt".format(batch_id))
            with open(batch_logger, "r") as handle:
                batch_statistics = [int(line.rstrip()) for line in handle]
                statistics.loc[batch_id] = batch_statistics
        statistics = statistics.sum(axis=0)

        self.logging.info("Statistics on probes:")
        self.logging.info("{} probes in total.".format(statistics.total))
        self.logging.info("{} probes after GC content and melting temperature filter.".format(statistics.GC_Tm))
        self.logging.info("{} probes after exact match filter.".format(statistics.exact_match))
        self.logging.info("{} probes after blast filter.".format(statistics.blast))

        transcriptome_size = os.path.getsize(os.path.join(self.dir_output_annotations, "transcriptome.fna"))
        self.logging.info("Size of transcriptome: {} MB".format(round(transcriptome_size / (1024 * 1024), 4)))

        blast_folder_size = sum(
            os.path.getsize(os.path.join(self.dir_output_blast, file)) for file in os.listdir(self.dir_output_blast)
        )
        self.logging.info("Size of blast output: {} GB".format(round(blast_folder_size / (1024 * 1024 * 1024), 4)))

        _, _, files = next(os.walk(self.dir_output_probes))
        genes_with_probes = len(files) - 1
        self.logging.info("Number of gene for which probes could be designed: {}".format(genes_with_probes))

        genes_wo_probes = sum(
            1 for line in open(os.path.join(self.dir_output_probes, "genes_with_insufficient_probes.txt"))
        )
        self.logging.info("Number of gene for which no probes could be designed: {}".format(genes_wo_probes))

        if rm_intermediate_files:
            import shutil

            shutil.rmtree(self.dir_output_annotations)
            shutil.rmtree(self.dir_output_blast)
