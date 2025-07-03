############################################
# imports
############################################
import itertools
import multiprocessing
import os
import random

import gtfparse
import pandas as pd
import pyfaidx
import src.utils as utils
from Bio import SeqIO
from Bio.SeqUtils import GC
from Bio.SeqUtils import MeltingTemp as mt


############################################
# data preparation class
############################################


class DataModule:
    """This class is used to download annotations from NCBI or ensemble,
    retrieve the list of genes for which probes should be designed and
    get the list of all possible guides that can be designed for the list of genes,
    based on the transcriptome annotation of those genes.

    :param config: User-defined parameters, where keys are the parameter names and values are the paremeter values.
    :type config: dict
    :param logging: Logger object to store important information.
    :type logging: logging.basicConfig
    :param dir_output: User-defined output directory.
    :type dir_output: string
    """

    def __init__(self, config, logging, dir_output):
        """Constructor method"""
        # set logger
        self.logging = logging

        # set directory
        self.dir_output_annotations = utils.create_dir(dir_output, "annotations")

        # set parameters
        self.number_batchs = config["number_batchs"]

        self.source = config["source"]
        if self.source == "ensemble":
            self.ftp_gene = config["ftp_gene_ensemble"]
            self.ftp_genome = config["ftp_genome_ensemble"]
            self.ftp_chr_mapping = config["ftp_chr_mapping_ensemble"]
        elif self.source == "ncbi":
            self.ftp_gene = config["ftp_gene_ncbi"]
            self.ftp_genome = config["ftp_genome_ncbi"]
            self.ftp_chr_mapping = config["ftp_chr_mapping_ncbi"]
        else:
            raise ValueError('Error: unknown source "{}"'.format(self.source))

        self.file_gene_gtf = config["file_gene_gtf"]
        self.file_genome_fasta = config["file_genome_fasta"]
        self.file_transcriptome_bed = os.path.join(self.dir_output_annotations, "transcriptome.bed")
        self.file_transcriptome_fasta = os.path.join(self.dir_output_annotations, "transcriptome.fna")

        self.file_genes = config["file_genes"]
        self.probe_length = config["probe_length"]
        self.Tm_parameters = utils.get_Tm_parameters(config["Tm_parameters"])
        self.GC_content_min = config["GC_content_min"]
        self.GC_content_max = config["GC_content_max"]
        self.Tm_min = config["Tm_min"]
        self.Tm_max = config["Tm_max"]

        # initialize additional paremeters
        self.batch_size = None
        self.gene_annotation = None
        self.transcriptome_annotation = None
        self.genes = None

    def load_annotations(self):
        """Download all necessary annotation files (gene annotation and genome sequence) from NCBI or ensemble,
        or load provided annotation files. If NCBI is chosen as a source, download chromosome mapping file to map
        chromosome names between GenBank and Ref-Seq accession number.
        """

        def _download_chr_mapping():
            """Download file with mapping of chromosome names between GenBank and Ref-Seq accession number
            from ftp server and create a mapping dictionary.

            :return: Dictionary with mapping of chromsome names from GenBank to Ref-Seq.
            :rtype: dict
            """
            file_mapping = utils.ftp_download(
                self.ftp_chr_mapping["ftp_link"],
                self.ftp_chr_mapping["directory"],
                self.ftp_chr_mapping["file_name"],
                self.dir_output_annotations,
            )

            # skip comment lines but keep last comment line for header
            with open(file_mapping) as handle:
                *_comments, names = itertools.takewhile(lambda line: line.startswith("#"), handle)
                names = names[1:].split()

            assembly_report = pd.read_table(file_mapping, names=names, sep="\t", comment="#")

            mapping_chromosome = assembly_report[assembly_report["Sequence-Role"] == "assembled-molecule"]
            mapping_chromosome = pd.Series(
                mapping_chromosome["Sequence-Name"].values, index=mapping_chromosome["RefSeq-Accn"]
            ).to_dict()

            mapping_scaffolds = assembly_report[assembly_report["Sequence-Role"] != "assembled-molecule"]
            mapping_scaffolds = pd.Series(
                mapping_scaffolds["GenBank-Accn"].values, index=mapping_scaffolds["RefSeq-Accn"]
            ).to_dict()

            mapping = mapping_chromosome
            mapping.update(mapping_scaffolds)

            return mapping

        def _download_gene_gtf(mapping):
            """Download gene annotation in gtf file format from ftp server and unzip file.
            If gene annotation comes from ncbi, map chromosome annotation to Ref-Seq accession number.

            :param mapping: Chromosome mapping dictionary (GenBank to Ref-Seq), only required if source is ncbi.
            :type mapping: dict
            :return: Path to downloaded gene gtf file.
            :rtype: string
            """
            file_gene_gtf_gz = utils.ftp_download(
                self.ftp_gene["ftp_link"],
                self.ftp_gene["directory"],
                self.ftp_gene["file_name"],
                self.dir_output_annotations,
            )
            file_gene_gtf = utils.decompress_gzip(file_gene_gtf_gz)

            if self.source == "ncbi":
                _process_ncbi_gene_gtf(file_gene_gtf, mapping)

            return file_gene_gtf

        def _process_ncbi_gene_gtf(file_gene_gtf, mapping):
            """Process gene annotation file downloaded from NCBI: map chromosome annotation to Ref-Seq.

            :param file_gene_gtf: Path to gtf file with gene annotation.
            :type file_gene_gtf: string
            :param mapping: Chromosome mapping dictionary (GenBank to Ref-Seq).
            :type mapping: dict
            """
            file_tmp = os.path.join(self.dir_output_annotations, "temp.gtf")

            # write comment lines to new file
            with open(file_tmp, "w") as handle_out:
                with open(file_gene_gtf) as handle_in:
                    *_comments, names = itertools.takewhile(lambda line: line.startswith("#"), handle_in)
                    handle_out.write(names)

                # read gtf file without comment lines
                gene_annotation = pd.read_table(
                    file_gene_gtf,
                    names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
                    sep="\t",
                    comment="#",
                )

                # replace ncbi with genbank chromosome annotation
                for accession_number in gene_annotation.seqname.unique():
                    if accession_number in mapping:
                        gene_annotation.loc[gene_annotation.seqname == accession_number, "seqname"] = mapping[
                            accession_number
                        ]
                    else:
                        print("No mapping for accession number: {}".format(accession_number))
                        gene_annotation = gene_annotation[gene_annotation.seqname != accession_number]

                gene_annotation.to_csv(handle_out, sep="\t", header=False, index=False)
            os.replace(file_tmp, file_gene_gtf)

        def _download_genome_fasta(mapping):
            """Download genome sequence in fasta file format from ftp server and unzip file.
            If genome sequence comes from ncbi, map chromosome annotation to Ref-Seq accession number.

            :param mapping: Chromosome mapping dictionary (GenBank to Ref-Seq), only required if source is ncbi.
            :type mapping: dict
            :return: Path to downloaded genome fasta file.
            :rtype: string
            """
            file_genome_fasta_gz = utils.ftp_download(
                self.ftp_genome["ftp_link"],
                self.ftp_genome["directory"],
                self.ftp_genome["file_name"],
                self.dir_output_annotations,
            )
            file_genome_fasta = utils.decompress_gzip(file_genome_fasta_gz)

            if self.source == "ncbi":
                _process_ncbi_genome_fasta(file_genome_fasta, mapping)

            return file_genome_fasta

        def _process_ncbi_genome_fasta(file_genome_fasta, mapping):
            """Process genome sequence file downloaded from NCBI: map chromosome annotation to Ref-Seq.

            :param file_genome_fasta: Path to fasta file with genome sequence.
            :type file_genome_fasta: string
            :param mapping: Chromosome mapping dictionary (GenBank to Ref-Seq).
            :type mapping: dict
            """
            file_tmp = os.path.join(self.dir_output_annotations, "temp.fna")

            with open(file_tmp, "w") as handle:
                for chromosome_sequnece in SeqIO.parse(file_genome_fasta, "fasta"):
                    accession_number = chromosome_sequnece.id
                    if accession_number in mapping:
                        chromosome_sequnece.id = mapping[accession_number]
                        chromosome_sequnece.name = mapping[accession_number]
                        chromosome_sequnece.description = chromosome_sequnece.description.replace(
                            accession_number, mapping[accession_number]
                        )
                        SeqIO.write(chromosome_sequnece, handle, "fasta")
                    else:
                        print("No mapping for accession number: {}".format(accession_number))

            os.replace(file_tmp, file_genome_fasta)

        if self.source == "ncbi":
            # if source ncbi we need a chromosome mapping
            mapping = _download_chr_mapping()
        elif self.source == "ensemble":
            # if source ensemble we don't need a chromosome mapping
            mapping = None

        if self.file_gene_gtf is None:
            self.file_gene_gtf = _download_gene_gtf(mapping)
            self.logging.info(
                "Downloaded gene annotation from {} and save as gene gtf: {}".format(self.source, self.file_gene_gtf)
            )

        if self.file_genome_fasta is None:
            self.file_genome_fasta = _download_genome_fasta(mapping)
            self.logging.info(
                "Downloaded genome annotation from {} and save as genome fasta: {}".format(
                    self.source, self.file_genome_fasta
                )
            )

        print("Annotations downloaded.")

    def load_genes(self):
        """Load list of genes for which probes should be designed."""

        def _get_gene_list_from_file():
            """Read list of genes from text file (gene names must be provided in ensemble or NCBI annotation format).

            :return: List of gene names.
            :rtype: list
            """
            with open(self.file_genes) as handle:
                lines = handle.readlines()
                genes = [line.rstrip() for line in lines]
            random.shuffle(genes)

            return genes

        def _get_gene_list_from_annotation():
            """Retrieve list of genes from NCBI or ensemble gene annotation gtf file.

            :return: List of gene names.
            :rtype: list
            """
            genes = self.gene_annotation.loc[self.gene_annotation["feature"] == "gene"]
            genes = list(genes["gene_id"].unique())
            random.shuffle(genes)  # shuffle gene list for better distribution in batches

            return genes

        # load gene annotation
        self.gene_annotation = gtfparse.read_gtf(self.file_gene_gtf)

        # load list of genes from given file or annotation
        if self.file_genes is None:
            self.genes = _get_gene_list_from_annotation()
            self.logging.info("Loaded gene list from {} annotation.".format(self.source))
        else:
            self.genes = _get_gene_list_from_file()
            self.logging.info("Loaded gene list from {}.".format(self.file_genes))

        self.batch_size = int(len(self.genes) / self.number_batchs) + (len(self.genes) % self.number_batchs > 0)
        self.logging.info(
            "Probes for {} genes will be designed processed in {} parallele batches with {} genes in one batch".format(
                len(self.genes), self.number_batchs, self.batch_size
            )
        )

        print("Gene list loaded.")

    def load_transcriptome(self):
        """Create transcriptome from NCBI or ensemble gene annotation.
        Therefore, retreive all exons and all possible exon junctions from transcript annotation.
        For reference transcriptome, define exon junction region larger than probe length (+5 bp) to allow bulges in
        alignments.
        For probe transcriptome, define exon junction region as probe length - 1, to avoid duplicated probes from
        overlap with exon annotation.
        Merge exon and exon junction annotations that define the exact same region into one annotation entry.
        Save transcriptome annotation in bed12 format that allows split annotations, which are needed to get sequences
        for exon junctions.
        Save transcriptome sequence in fasta format.
        """

        def _load_unique_exons():
            """Merge overlapping exons, which have the same satrt and end coordinates.
            Save transcript information for those exons.

            :return: Dataframe with annotation of unique exons, where overlapping exons are merged.
            :rtype: pandas.DataFrame
            """
            exon_annotation = _load_exon_annotation()

            exon_annotation["region"] = (
                exon_annotation["seqname"]
                + "_"
                + exon_annotation["start"].astype("str")
                + "_"
                + exon_annotation["end"].astype("str")
                + "_"
                + exon_annotation["strand"]
            )

            aggregate_function = {
                "gene_id": "first",
                "transcript_id": ":".join,
                "exon_id": ":".join,
                "seqname": "first",
                "start": "first",
                "end": "first",
                "score": "first",
                "strand": "first",
            }
            merged_exons = exon_annotation.groupby(exon_annotation["region"]).aggregate(aggregate_function)

            merged_exons["score"] = 0
            merged_exons["thickStart"] = merged_exons["start"]
            merged_exons["thickEnd"] = merged_exons["end"]
            merged_exons["itemRgb"] = 0
            merged_exons["blockCount"] = 1
            merged_exons["blockSizes"] = merged_exons["end"] - merged_exons["start"]
            merged_exons["blockStarts"] = 0
            merged_exons["gene_transcript_exon_id"] = (
                merged_exons["gene_id"] + "_tid" + merged_exons["transcript_id"] + "_eid" + merged_exons["exon_id"]
            )
            merged_exons = merged_exons[
                [
                    "gene_id",
                    "gene_transcript_exon_id",
                    "seqname",
                    "start",
                    "end",
                    "score",
                    "strand",
                    "thickStart",
                    "thickEnd",
                    "itemRgb",
                    "blockCount",
                    "blockSizes",
                    "blockStarts",
                ]
            ]

            return merged_exons

        def _merge_containing_exons(unique_exons):

            aggregate_function = {
                "gene_id": "first",
                "gene_transcript_exon_id": ":".join,
                "seqname": "first",
                "start": "min",
                "end": "max",
                "score": "first",
                "strand": "first",
                "thickStart": "min",
                "thickEnd": "max",
                "itemRgb": "first",
                "blockCount": "first",
                "blockSizes": "max",
                "blockStarts": "first",
            }

            unique_exons["region_start"] = (
                unique_exons["seqname"] + "_" + unique_exons["start"].astype("str") + "_" + unique_exons["strand"]
            )
            merged_unique_exons = unique_exons.groupby(unique_exons["region_start"]).aggregate(aggregate_function)

            merged_unique_exons["region_end"] = (
                merged_unique_exons["seqname"]
                + "_"
                + merged_unique_exons["end"].astype("str")
                + "_"
                + merged_unique_exons["strand"]
            )
            merged_unique_exons = merged_unique_exons.groupby(merged_unique_exons["region_end"]).aggregate(
                aggregate_function
            )

            return merged_unique_exons

        def _load_exon_junctions(blockSize):
            """Get all possible exon junctions from the transcript annotation.
            Merge overlapping exons jucntions, which have the same satrt and end coordinates.
            Save transcript information for those exons.

            :param blockSize: Size of the exon junction regions, i.e. <blockSize> bp upstream of first exon
                and <blockSize> bp downstream of second exon.
            :type blockSize: int
            :return: Dataframe with annotation of exons junctions, where overlapping exons junctions are merged.
            :rtype: pandas.DataFrame
            """
            exon_annotation = _load_exon_annotation()
            transcript_exons, transcript_info = _get_transcript_exons_and_info(exon_annotation)
            exon_junction_list = _get_exon_junction_list(blockSize, transcript_exons, transcript_info)

            exon_junctions = pd.DataFrame(
                exon_junction_list,
                columns=[
                    "region",
                    "gene_id",
                    "transcript_id",
                    "exon_id",
                    "seqname",
                    "start",
                    "end",
                    "score",
                    "strand",
                    "thickStart",
                    "thickEnd",
                    "itemRgb",
                    "blockCount",
                    "blockSizes",
                    "blockStarts",
                ],
            )
            aggregate_function = {
                "gene_id": "first",
                "transcript_id": ":".join,
                "exon_id": ":".join,
                "seqname": "first",
                "start": "first",
                "end": "first",
                "score": "first",
                "strand": "first",
                "thickStart": "first",
                "thickEnd": "first",
                "itemRgb": "first",
                "blockCount": "first",
                "blockSizes": "first",
                "blockStarts": "first",
            }
            merged_exon_junctions = exon_junctions.groupby(exon_junctions["region"]).aggregate(aggregate_function)

            merged_exon_junctions["gene_transcript_exon_id"] = (
                merged_exon_junctions["gene_id"]
                + "_tid"
                + merged_exon_junctions["transcript_id"]
                + "_eid"
                + merged_exon_junctions["exon_id"]
            )
            merged_exon_junctions = merged_exon_junctions[
                [
                    "gene_id",
                    "gene_transcript_exon_id",
                    "seqname",
                    "start",
                    "end",
                    "score",
                    "strand",
                    "thickStart",
                    "thickEnd",
                    "itemRgb",
                    "blockCount",
                    "blockSizes",
                    "blockStarts",
                ]
            ]

            return merged_exon_junctions

        def _load_exon_annotation():
            """Retrive exon annotation from loaded gene annotation.

            :return: Exon annotation.
            :rtype: pandas.DataFrame
            """
            exon_annotation = self.gene_annotation.loc[self.gene_annotation["feature"] == "exon"]
            exon_annotation = exon_annotation.assign(source="unknown")

            if "exon_id" not in exon_annotation.columns:
                exon_annotation["exon_id"] = exon_annotation["transcript_id"] + "_exon" + exon_annotation["exon_number"]

            # there are some exon annotations which have the same start and end coordinates and can't be saved as fasta
            # from bedtools
            exon_annotation = exon_annotation[(exon_annotation.end - exon_annotation.start) > 0]
            exon_annotation.reset_index(inplace=True, drop=True)

            return exon_annotation

        def _get_transcript_exons_and_info(exon_annotation):
            """Get list of exons that belong to a transcript.
            Save information about each transcript, i.e. gene ID, chromosome and strand.

            :param exon_annotation: Exon annotation.
            :type exon_annotation: pandas.DataFrame
            :return: Two dictionaries: transcript - exon mapping and transcript information.
            :rtype: dict
            """
            transcripts = self.gene_annotation.loc[self.gene_annotation["feature"] == "transcript"]
            transcripts = sorted(list(transcripts["transcript_id"].unique()))
            transcript_exons = {key: {} for key in transcripts}
            transcript_info = dict()

            for row in exon_annotation.index:
                transcript_id = exon_annotation.transcript_id[row]
                transcript_exons[transcript_id][int(exon_annotation.exon_number[row])] = [
                    exon_annotation.exon_id[row],
                    exon_annotation.start[row],
                    exon_annotation.end[row],
                ]
                transcript_info[transcript_id] = [
                    exon_annotation.gene_id[row],
                    exon_annotation.seqname[row],
                    exon_annotation.strand[row],
                ]

            return transcript_exons, transcript_info

        def _get_exon_junction_list(blockSize, transcript_exons, transcript_info):
            """Get list of all possible exon junctions and save all required information for bed12 file.

            :param blockSize: Size of the exon junction regions, i.e. <blockSize> bp upstream of first exon
                and <blockSize> bp downstream of second exon.
            :type blockSize: int
            :param transcript_exons: Transcript - exon mapping.
            :type transcript_exons: dict
            :param transcript_info: Transcript information.
            :type transcript_info: dict
            :return: List of exon junctions.
            :rtype: list
            """
            exon_junction_list = []

            for transcript, exons in transcript_exons.items():
                gene_id = transcript_info[transcript][0]
                seqname = transcript_info[transcript][1]
                strand = transcript_info[transcript][2]
                exon_upstream = []
                exon_downstream = []

                if strand == "+":
                    exons = sorted(exons.items())
                elif strand == "-":
                    exons = sorted(exons.items(), reverse=True)

                for _exon, attributes in exons:
                    if exon_upstream == []:
                        exon_upstream = attributes
                    else:
                        exon_downstream = attributes
                        start_up = max(exon_upstream[1], (exon_upstream[2] - blockSize))
                        end_down = min((exon_downstream[1] + blockSize), exon_downstream[2])

                        exon_junction_list.append(
                            [
                                "{}_{}_{}_{}".format(seqname, start_up, end_down, strand),
                                gene_id,
                                transcript,
                                "{}_{}".format(exon_upstream[0], exon_downstream[0]),
                                seqname,
                                start_up,
                                end_down,
                                0,
                                strand,
                                start_up,
                                end_down,
                                0,
                                2,
                                "{},{}".format(blockSize, blockSize),
                                "{},{}".format(0, exon_downstream[1] - start_up),
                            ]
                        )
                        exon_upstream = attributes

            return exon_junction_list

        # get annotation of exons and merge exon annotations for the same region
        unique_exons = _load_unique_exons()
        print("{} unique exons loaded.".format(len(unique_exons.index)))

        # get exon junction annotation for probes --> length os probe_length - 1 to continue where exons annotation ends
        exon_junctions_probes = _load_exon_junctions(self.probe_length - 1)
        self.transcriptome_annotation = unique_exons.append(exon_junctions_probes)
        self.transcriptome_annotation = self.transcriptome_annotation.sort_values(by=["gene_id"])
        self.transcriptome_annotation.reset_index(inplace=True, drop=True)
        print("{} exon junctions loaded.".format(len(exon_junctions_probes.index)))

        # get exon junction annotation for reference --> longer than probe length to cover bulges in alignments
        exon_junctions_reference = _load_exon_junctions(self.probe_length + 5)  # to allow bulges in the alignment
        unique_exons_reference = _merge_containing_exons(unique_exons)
        transcriptome_reference = unique_exons_reference.append(exon_junctions_reference)
        transcriptome_reference = transcriptome_reference.sort_values(by=["gene_id"])
        transcriptome_reference.reset_index(inplace=True, drop=True)
        transcriptome_reference[
            [
                "seqname",
                "start",
                "end",
                "gene_id",
                "score",
                "strand",
                "thickStart",
                "thickEnd",
                "itemRgb",
                "blockCount",
                "blockSizes",
                "blockStarts",
            ]
        ].to_csv(self.file_transcriptome_bed, sep="\t", header=False, index=False)

        utils.get_fasta(self.file_transcriptome_bed, self.file_genome_fasta, self.file_transcriptome_fasta, split=True)

    def load_probes(self):
        """Get the fasta sequence of all possible probes with user-defined length for all input genes.
        Generated probes are filtered by undefined nucleotides ('N') in their sequence.
        In addition, generated probes are filtered based on GC content and melting temperature for user-defined
        thresholds
        This process can be executed in a parallele fashion on a user-defined number of threads (defined in config
        file).
        """

        def _get_probes(batch_id, genes_batch):
            """Get the fasta sequence of all possible probes for all genes in the batch.

            :param batch_id: Batch ID.
            :type batch_id: int
            :param genes_batch: List of genes for which probes should be designed.
            :type genes_batch: list
            """
            file_transcriptome_bed_batch = os.path.join(
                self.dir_output_annotations, "transcriptome_batch{}.bed".format(batch_id)
            )
            file_transcriptome_fasta_batch = os.path.join(
                self.dir_output_annotations, "transcriptome_batch{}.fna".format(batch_id)
            )
            file_probe_info_batch = os.path.join(
                self.dir_output_annotations, "probes_info_batch{}.txt".format(batch_id)
            )
            file_probe_sequence_batch = os.path.join(
                self.dir_output_annotations, "probes_sequence_batch{}.txt".format(batch_id)
            )
            batch_logger = os.path.join(self.dir_output_annotations, "logger_batch{}.txt".format(batch_id))

            _get_transcriptome_fasta(genes_batch, file_transcriptome_bed_batch, file_transcriptome_fasta_batch)
            gene_probes = _get_probes_info(genes_batch, file_transcriptome_fasta_batch, batch_logger)
            _write_probes_info(gene_probes, file_probe_info_batch, file_probe_sequence_batch)

            os.remove(file_transcriptome_bed_batch)
            os.remove(file_transcriptome_fasta_batch)

        def _get_transcriptome_fasta(genes_batch, file_transcriptome_bed_batch, file_transcriptome_fasta_batch):
            """Extract transcripts for the current batch and write transcript regions to bed file.
            Get sequence for annotated transcript regions (bed file) from genome sequence (fasta file) and write
            transcriptome sequences to fasta file.

            :param genes_batch: List of genes for which probes should be designed.
            :type genes_batch: list
            :param file_transcriptome_bed_batch: Path to bed transcriptome annotation output file.
            :type file_transcriptome_bed_batch: string
            :param file_transcriptome_fasta_batch: Path to fasta transcriptome sequence output file.
            :type file_transcriptome_fasta_batch: string
            """
            transcriptome_annotation_genes = self.transcriptome_annotation.loc[
                self.transcriptome_annotation["gene_id"].isin(genes_batch)
            ].copy()
            transcriptome_annotation_genes = transcriptome_annotation_genes.sort_values(by=["gene_id"])
            transcriptome_annotation_genes[
                [
                    "seqname",
                    "start",
                    "end",
                    "gene_transcript_exon_id",
                    "score",
                    "strand",
                    "thickStart",
                    "thickEnd",
                    "itemRgb",
                    "blockCount",
                    "blockSizes",
                    "blockStarts",
                ]
            ].to_csv(file_transcriptome_bed_batch, sep="\t", header=False, index=False)

            # get sequence for exons
            utils.get_fasta(
                file_transcriptome_bed_batch, self.file_genome_fasta, file_transcriptome_fasta_batch, split=True
            )

        def _parse_header(header):
            """Helper function to parse the header of each exon fasta entry.

            :param header: Header of fasta entry.
            :type header: string
            :return: Parsed header information, i.e.
                gene_id, transcript_id, exon_id and position (chromosome, start, strand)
            :rtype: string
            """
            identifier = header.split("::")[0]
            gene_id = identifier.split("_tid")[0]
            transcript_id = identifier.split("_tid")[1].split("_eid")[0]
            exon_id = identifier.split("_eid")[1]

            coordinates = header.split("::")[1]
            chrom = coordinates.split(":")[0]
            start = int(coordinates.split(":")[1].split("-")[0])
            strand = coordinates.split("(")[1].split(")")[0]

            return gene_id, transcript_id, exon_id, chrom, start, strand

        def _get_probes_info(genes_batch, file_transcriptome_fasta_batch, batch_logger):
            """Merge all probes with identical sequence that come from the same gene into one fasta entry.
            Filter all probes based on GC content and melting temperature for user-defined thresholds.
            Collect additional information about each probe.

            :param genes_batch: List of genes for which probes should be designed.
            :type genes_batch: list
            :param file_transcriptome_fasta_batch: Path to fasta transcriptome sequence output file.
            :type file_transcriptome_fasta_batch: string
            :param batch_logger: Path to logger file for probe statistics.
            :type batch_logger: string
            :return: Mapping of probes to corresponding genes with additional information about each probe, i.e.
                position (chromosome, start, end, strand), gene_id, transcript_id, exon_id, melting temp. and GC content
            :rtype: dict
            """
            gene_probes = {key: {} for key in genes_batch}
            total_probes = 0
            loaded_probes = 0

            # parse the exon fasta sequence file
            for exon in SeqIO.parse(file_transcriptome_fasta_batch, "fasta"):
                sequence = exon.seq

                if len(sequence) > self.probe_length:
                    number_probes = len(sequence) - (self.probe_length - 1)
                    probes_sequence = [sequence[i: i + self.probe_length] for i in range(number_probes)]

                    for i in range(number_probes):
                        total_probes += 1
                        probe_sequence = probes_sequence[i]

                        if "N" not in probe_sequence:
                            gc_content = round(GC(probe_sequence), 2)

                            if self.GC_content_min < gc_content < self.GC_content_max:
                                Tm = round(mt.Tm_NN(probe_sequence, **self.Tm_parameters), 2)

                                if self.Tm_min < Tm < self.Tm_max:
                                    gene_id, transcript_id, exon_id, chrom, start, strand = _parse_header(exon.id)
                                    probe_start = start + i
                                    probe_end = start + i + self.probe_length

                                    tmp = gene_probes[gene_id]

                                    if probe_sequence in tmp:
                                        tmp[probe_sequence]["transcript_id"].append(transcript_id)
                                        tmp[probe_sequence]["exon_id"].append(exon_id)
                                        tmp[probe_sequence]["start"].append(probe_start)
                                        tmp[probe_sequence]["end"].append(probe_end)
                                    else:
                                        loaded_probes += 1
                                        tmp[probe_sequence] = {
                                            "transcript_id": [transcript_id],
                                            "exon_id": [exon_id],
                                            "chr": chrom,
                                            "start": [probe_start],
                                            "end": [probe_end],
                                            "strand": strand,
                                            "gc": gc_content,
                                            "Tm": Tm,
                                        }
                                    gene_probes[gene_id] = tmp

            with open(batch_logger, "w") as handle:
                handle.write("{}\n".format(total_probes))
                handle.write("{}\n".format(loaded_probes))

            return gene_probes

        def _write_probes_info(gene_probes, file_probe_info_batch, file_probe_sequence_batch):
            """Save additional probe information in tsv file.
            Additionally, save all probe sequences of this batch in a seperate text file.

            :param gene_probes: Mapping of probes to corresponding genes with additional information about each probe,
                i.e. position (chromosome, start, end, strand), gene_id, transcript_id, exon_id, melting temp. and GC
                content
            :type gene_probes: dict
            :param file_probe_info_batch: Path to probe info output file.
            :type file_probe_info_batch: string
            :param file_probe_sequence_batch: Path to probe sequence output file.
            :type file_probe_sequence_batch: string
            """
            with open(file_probe_info_batch, "w") as handle_probeinfo:
                handle_probeinfo.write(
                    "gene_id\ttranscript_id\texon_id\tprobe_sequence\tchromosome\tstart\tend\tstrand\tGC_content\t"
                    "melting_temperature\n"
                )

                with open(file_probe_sequence_batch, "w") as handle_sequences:
                    for gene_id, probes in gene_probes.items():
                        for probe_sequence, probe_attributes in probes.items():
                            output1 = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                gene_id,
                                ";".join(probe_attributes["transcript_id"]),
                                ";".join(probe_attributes["exon_id"]),
                                probe_sequence,
                                probe_attributes["chr"],
                                ";".join(str(s) for s in probe_attributes["start"]),
                                ";".join(str(e) for e in probe_attributes["end"]),
                                probe_attributes["strand"],
                                probe_attributes["gc"],
                                probe_attributes["Tm"],
                            )
                            handle_probeinfo.write(output1)

                            output2 = "{}\n".format(probe_sequence)
                            handle_sequences.write(output2)

        # create index file
        pyfaidx.Fasta(self.file_genome_fasta)

        # create probes in parallele
        jobs = []
        for batch_id in range(self.number_batchs):
            genes_batch = self.genes[
                (self.batch_size * batch_id): (min(self.batch_size * (batch_id + 1), len(self.genes) + 1))
            ]

            proc = multiprocessing.Process(
                target=_get_probes,
                args=(
                    batch_id,
                    genes_batch,
                ),
            )
            jobs.append(proc)
            proc.start()

        print("\n {} \n".format(jobs))

        for job in jobs:
            job.join()

        # remove index file
        os.remove("{}.fai".format(self.file_genome_fasta))
        print("All probes created.")
