#!/usr/bin/env python3
import os
import csv
import subprocess
from pathlib import Path
from Bio import Entrez

########################################################################
# CONFIG
########################################################################

Entrez.email = "milkmanmahler1017@gmail.com"   # REQUIRED by NCBI

# Path to your reference genome FASTA (must be bwa-indexed and faidx'd)
REF_FASTA = "reference/genome.fa"

# Threads for mapping / QC / calling
THREADS = "8"

# Base output directory
BASE_OUTDIR = Path("data/genomic_data")

# Whether to run fastp QC
USE_FASTP = True


########################################################################
# Helper to run commands
########################################################################

def run_cmd(cmd, cwd=None, use_shell=False):
    """
    Run a shell command and crash if it fails.
    cmd: list[str] if use_shell=False, or str if use_shell=True.
    """
    if use_shell:
        print("Running (shell):", cmd)
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    else:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=cwd)


########################################################################
# 1. Fetch SRR runs for an SRX
########################################################################

def get_run_accessions(srx):
    """
    Given an SRX accession, return all SRR run accessions.
    """
    handle = Entrez.esearch(db="sra", term=srx)
    record = Entrez.read(handle)
    handle.close()

    if not record["IdList"]:
        raise ValueError(f"No SRA record found for {srx}")

    sra_id = record["IdList"][0]

    # Try runinfo first
    handle = Entrez.efetch(db="sra", id=sra_id, rettype="runinfo", retmode="text")
    raw = handle.read()
    handle.close()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    lines = raw.strip().splitlines()
    if len(lines) > 1:
        header = lines[0].split(",")
        if "Run" in header:
            idx = header.index("Run")
            runs = [row.split(",")[idx] for row in lines[1:] if row.strip()]
            if runs:
                return runs

    # Fallback: esummary
    handle = Entrez.esummary(db="sra", id=sra_id, retmode="xml")
    summary = Entrez.read(handle)
    handle.close()

    runs = []
    exp = summary[0]
    if "Runs" in exp:
        for r in exp["Runs"].split(","):
            r = r.strip()
            if r.startswith("SRR"):
                runs.append(r)

    if not runs:
        raise RuntimeError(f"No SRR runs found for {srx}")

    return runs


########################################################################
# 2. Download FASTQs using SRA Toolkit (per SRR)
########################################################################

def download_fastq(run_accession, outdir):
    """
    Download one SRR using prefetch + fasterq-dump (paired-end assumed).
    Output: <run>_1.fastq, <run>_2.fastq in outdir.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading {run_accession} ===")
    run_cmd(["prefetch", run_accession])
    run_cmd([
        "fasterq-dump",
        "--split-files",
        "--outdir", str(outdir),
        run_accession
    ])

    fq1 = outdir / f"{run_accession}_1.fastq"
    fq2 = outdir / f"{run_accession}_2.fastq"

    if not fq1.exists():
        raise FileNotFoundError(f"Missing {fq1}")
    if not fq2.exists():
        raise FileNotFoundError(f"Missing {fq2}")

    return fq1, fq2


########################################################################
# 3. Optional QC using fastp
########################################################################

def qc_fastqs(run_accession, fq1, fq2, outdir):
    """
    Run fastp on paired-end reads.
    Output: <run>.clean_1.fastq, <run>.clean_2.fastq
    """
    outdir = Path(outdir)
    clean1 = outdir / f"{run_accession}.clean_1.fastq"
    clean2 = outdir / f"{run_accession}.clean_2.fastq"

    json_report = outdir / f"{run_accession}.fastp.json"
    html_report = outdir / f"{run_accession}.fastp.html"

    print(f"=== QC with fastp: {run_accession} ===")
    run_cmd([
        "fastp",
        "-i", str(fq1),
        "-I", str(fq2),
        "-o", str(clean1),
        "-O", str(clean2),
        "-w", THREADS,
        "-j", str(json_report),
        "-h", str(html_report)
    ])

    return clean1, clean2


########################################################################
# 4. Align to reference with bwa mem + samtools sort/index
########################################################################

def align_and_sort(sample_id, fq1, fq2, outdir, ref_fasta=REF_FASTA):
    """
    Map paired reads to reference and produce sorted, indexed BAM.

    Output:
      <outdir>/<sample_id>.sorted.bam
      <outdir>/<sample_id>.sorted.bam.bai
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bam_path = outdir / f"{sample_id}.sorted.bam"

    print(f"=== Aligning {sample_id} to reference ===")

    # Use a shell pipeline: bwa mem -> samtools sort -> BAM
    cmd = (
        f"bwa mem -t {THREADS} {ref_fasta} {fq1} {fq2} "
        f"| samtools sort -@ {THREADS} -o {bam_path} -"
    )
    run_cmd(cmd, use_shell=True)

    # Index BAM
    run_cmd(["samtools", "index", str(bam_path)])

    return bam_path


########################################################################
# 5. Per-sample FASTQ cleanup
########################################################################

def cleanup_fastqs_for_run(run_accession, outdir, used_fastp):
    """
    Delete FASTQ files for one run to save disk space.
    Removes:
      <run>_1.fastq, <run>_2.fastq
      and if fastp used:
      <run>.clean_1.fastq, <run>.clean_2.fastq
    """
    outdir = Path(outdir)
    print(f"=== Cleaning up FASTQs for {run_accession} ===")

    paths_to_delete = [
        outdir / f"{run_accession}_1.fastq",
        outdir / f"{run_accession}_2.fastq",
    ]

    if used_fastp:
        paths_to_delete.extend([
            outdir / f"{run_accession}.clean_1.fastq",
            outdir / f"{run_accession}.clean_2.fastq",
        ])

    for p in paths_to_delete:
        if p.exists():
            print(f"  deleting {p}")
            p.unlink()
        else:
            print(f"  (not found, skipping) {p}")


########################################################################
# 6. Joint variant calling with bcftools (all BAMs at once)
########################################################################

def joint_variant_calling(ref_fasta, bam_paths, out_vcf_gz):
    """
    Run bcftools mpileup + bcftools call on all BAMs jointly.
    Output: compressed VCF (.vcf.gz) + index (.csi)
    """
    bam_paths = [str(p) for p in bam_paths]
    out_vcf_gz = Path(out_vcf_gz)

    print("\n=== Joint variant calling with bcftools ===")
    mpileup_cmd = [
        "bcftools", "mpileup", "-Ou",
        "-f", ref_fasta
    ] + bam_paths

    call_cmd = [
        "bcftools", "call", "-mv",
        "-Oz", "-o", str(out_vcf_gz)
    ]

    # Pipe: mpileup -> call in one shell pipeline
    joined_cmd = (
        " ".join(mpileup_cmd) +
        " | " +
        " ".join(call_cmd)
    )
    run_cmd(joined_cmd, use_shell=True)

    # Index VCF
    run_cmd(["bcftools", "index", str(out_vcf_gz)])

    print(f"Joint VCF written to: {out_vcf_gz}")


########################################################################
# 7. Main workflow — process CSV
########################################################################

def process_csv(csv_path):
    csv_path = Path(csv_path)

    # Make sure reference exists
    if not Path(REF_FASTA).exists():
        raise FileNotFoundError(
            f"Reference FASTA {REF_FASTA} not found. "
            f"Set REF_FASTA at the top of this script."
        )

    # Mapping from ecotype -> list of SRX
    ecotype_to_srx = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            srx = row["Experiment Accession"]
            library_name = row["Library Name"]
            # Same assumption as before: ecotype prefix before first '-'
            ecotype = library_name.split("-")[0]
            ecotype_to_srx.setdefault(ecotype, []).append(srx)

    all_bams = []
    metadata_rows = []

    for ecotype, srx_list in ecotype_to_srx.items():
        print("\n===================================================")
        print(f"PROCESSING ECOTYPE: {ecotype}")
        print(f"  SRXs: {srx_list}")
        print("===================================================\n")

        ecotype_dir = BASE_OUTDIR / ecotype
        ecotype_dir.mkdir(parents=True, exist_ok=True)

        for srx in srx_list:
            print(f"\n--- SRX: {srx} ---")
            runs = get_run_accessions(srx)
            print(f"  SRR runs: {runs}")

            for run in runs:
                # Treat each SRR run as one individual
                sample_id = run

                print(f"\n### SAMPLE {sample_id} (ecotype {ecotype}) ###")

                # 1. Download FASTQs for this run
                fq1, fq2 = download_fastq(run, ecotype_dir)

                # 2. Optional QC
                if USE_FASTP:
                    fq1_clean, fq2_clean = qc_fastqs(run, fq1, fq2, ecotype_dir)
                else:
                    fq1_clean, fq2_clean = fq1, fq2

                # 3. Align + sort/index -> BAM
                bam_path = align_and_sort(
                    sample_id=sample_id,
                    fq1=fq1_clean,
                    fq2=fq2_clean,
                    outdir=ecotype_dir
                )
                all_bams.append(bam_path)

                # 4. Immediately clean up FASTQs for this run
                cleanup_fastqs_for_run(run, ecotype_dir, used_fastp=USE_FASTP)

                # 5. Record metadata
                metadata_rows.append({
                    "sample_id": sample_id,
                    "ecotype": ecotype,
                    "SRR": run,
                    "SRX": srx
                })

    # Write metadata table
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)
    meta_path = BASE_OUTDIR / "sample_metadata.tsv"
    print(f"\nWriting sample metadata to: {meta_path}")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "ecotype", "SRR", "SRX"],
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Joint variant calling at the end
    if all_bams:
        joint_vcf = BASE_OUTDIR / "all_samples.vcf.gz"
        joint_variant_calling(REF_FASTA, all_bams, joint_vcf)
    else:
        print("No BAMs produced — nothing to call variants on.")


########################################################################
# 8. Script entry point
########################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download GBS/RAD data per individual, "
            "align to a single reference, and joint-call SNPs."
        )
    )
    parser.add_argument(
        "csv_file",
        help="Input CSV with columns 'Experiment Accession' and 'Library Name'"
    )
    args = parser.parse_args()

    process_csv(args.csv_file)
