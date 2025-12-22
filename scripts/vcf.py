#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys


def run(cmd):
    """Run a command and exit on error."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def file_exists(path):
    """Check if a file exists."""
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser(
        description="Combine sample FASTAs, align to a reference, and call variants to produce a VCF.")
    
    parser.add_argument(
        "--ref", help="Path to the reference FASTA file")
    parser.add_argument(
        "--samples", nargs='+', help="Paths to one or more sample FASTA files to combine and align")
    parser.add_argument(
        "--outdir", help="Directory to write output files into")
    
    args = parser.parse_args()

    ref = args.ref
    samples = args.samples
    outdir = args.outdir

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Combine sample FASTAs
    combined_fasta = os.path.join(outdir, "combined_samples.fasta")
    if not file_exists(combined_fasta):
        print(f"Combining {len(samples)} FASTA files into {combined_fasta}")
        with open(combined_fasta, 'w') as outfile:
            for fasta in samples:
                if not file_exists(fasta):
                    print(f"Error: sample file {fasta} not found.")
                    sys.exit(1)
                with open(fasta) as infile:
                    outfile.write(infile.read())
    else:
        print(f"{combined_fasta} exists. Skipping combination.")

    # Use combined FASTA as sample
    sample_base = os.path.splitext(os.path.basename(combined_fasta))[0]

    # 1. Index reference with BWA
    ref_prefix = os.path.splitext(ref)[0]
    bwa_index_files = [f"{ref_prefix}.{ext}" for ext in ["amb", "ann", "bwt", "pac", "sa"]]
    if not all(file_exists(f) for f in bwa_index_files):
        run(["bwa", "index", ref])
    else:
        print("Reference already indexed. Skipping bwa index.")

    # 2. Align combined samples to reference (SAM)
    sam_path = os.path.join(outdir, sample_base + ".sam")
    if not file_exists(sam_path):
        run(["bwa", "mem", ref, combined_fasta, "-o", sam_path])
    else:
        print(f"{sam_path} exists. Skipping alignment.")

    # 3. Convert SAM to unsorted BAM
    bam_path = os.path.join(outdir, sample_base + ".bam")
    if not file_exists(bam_path):
        run(["samtools", "view", "-bS", sam_path, "-o", bam_path])
    else:
        print(f"{bam_path} exists. Skipping SAM->BAM conversion.")

    # 4. Sort BAM
    sorted_bam = os.path.join(outdir, sample_base + ".sorted.bam")
    if not file_exists(sorted_bam):
        run(["samtools", "sort", bam_path, "-o", sorted_bam])
    else:
        print(f"{sorted_bam} exists. Skipping BAM sorting.")

    # 5. Index sorted BAM
    bam_index = sorted_bam + ".bai"
    if not file_exists(bam_index):
        run(["samtools", "index", sorted_bam])
    else:
        print(f"{bam_index} exists. Skipping BAM indexing.")

    # 6. Generate BCF via mpileup
    bcf_path = os.path.join(outdir, sample_base + ".bcf")
    if not file_exists(bcf_path):
        run(["bcftools", "mpileup", "-f", ref, sorted_bam, "-Ob", "-o", bcf_path])
    else:
        print(f"{bcf_path} exists. Skipping mpileup.")

    # 7. Call variants to VCF
    vcf_path = os.path.join(outdir, sample_base + ".vcf")
    if not file_exists(vcf_path):
        run(["bcftools", "call", "-mv", "-Ov", "-o", vcf_path, bcf_path])
    else:
        print(f"{vcf_path} exists. Skipping variant calling.")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
