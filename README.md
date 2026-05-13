```markdown
# CISC 886 — Biomedical Research Chatbot

> An end-to-end cloud-based biomedical question-answering chatbot built on AWS —
> from raw PubMed data ingestion through Spark preprocessing, LoRA fine-tuning of TinyLlama,
> to GGUF model deployment on EC2 via Ollama and Open WebUI.

---

## Introduction

This project delivers a **production-style, cloud-native biomedical chatbot** developed as part of
CISC 886 (Cloud Computing) by **Team 19**. The system ingests up to **1,000,000 PubMed research
abstracts** from HuggingFace, preprocesses them at scale using **Apache Spark on AWS EMR**, and
fine-tunes **TinyLlama-1.1B-Chat** with **LoRA (PEFT)** via the Unsloth framework in Google Colab.
The resulting quantized `GGUF` model is stored in **Amazon S3** and deployed on an
**EC2 m5.xlarge** instance, served through **Ollama** and exposed to end users via
**Open WebUI** — all provisioned using **Terraform**.

| Property           | Value                                                                        |
|--------------------|------------------------------------------------------------------------------|
| **Course**         | CISC 886 — Cloud Computing                                                   |
| **Team**           | Team 19                                                                      |
| **Goal**           | End-to-end cloud biomedical chatbot pipeline on AWS                          |
| **Base Model**     | TinyLlama/TinyLlama-1.1B-Chat-v1.0                                          |
| **Dataset**        | ncbi/pubmed (HuggingFace streaming, up to 1M records)                        |
| **Deployment**     | EC2 m5.xlarge + Ollama + Open WebUI (Docker)                                 |
| **IaC Tool**       | Terraform (VPC, subnets, security groups, us-east-1)                         |
| **Estimated Cost** | ~$148.16/month (covered by AWS student credits — actual: **$0**)            |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
   - [Local Tools](#local-tools)
   - [Accounts & Access](#accounts--access)
   - [Environment Requirements](#environment-requirements)
4. [Phase 1 — Data Ingestion](#phase-1--data-ingestion)
5. [Phase 2 — Spark Preprocessing](#phase-2--spark-preprocessing)
6. [Phase 3 — Model Fine-Tuning](#phase-3--model-fine-tuning)
7. [Phase 4 — Infrastructure & Deployment](#phase-4--infrastructure--deployment)
8. [Configuration Reference](#configuration-reference)
9. [Cost Summary](#cost-summary)
10. [Troubleshooting](#troubleshooting)
11. [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements a **four-phase MLOps pipeline** entirely on AWS,
treating each stage as an independent, reproducible unit.

| Property                | Detail                                                                        |
|-------------------------|-------------------------------------------------------------------------------|
| **Pipeline Stages**     | Ingestion -> Preprocessing -> Fine-Tuning -> Deployment                       |
| **Data Source**         | `ncbi/pubmed` via HuggingFace Hub (streaming)                                 |
| **Raw Storage**         | `s3://q1abc-cisc886-bucket/raw/pubmed/`                                       |
| **Processed Storage**   | `s3://q1abc-cisc886-bucket/processed/train/`                                  |
| **Model Storage**       | `s3://q1abc-cisc886-bucket/fine-tuned-models/tinyllama-q4km.gguf`            |
| **Training Platform**   | Google Colab (GPU runtime)                                                    |
| **Inference Platform**  | AWS EC2 m5.xlarge (Amazon Linux 2)                                            |
| **User Interface**      | Open WebUI (Docker, port 8080)                                                |
| **Model Format**        | GGUF q4_k_m (4-bit quantized)                                                 |
| **Trainable Params**    | 4,505,600 / 1,104,553,984 (0.41%)                                             |
| **IaC Tool**            | Terraform (VPC, subnets, security groups, EC2)                                |

### Architecture Overview

```
HuggingFace PubMed Stream
        |
        v
  [Phase 1]  download_pubmed.py
  Snappy Parquet --> S3 raw/
        |
        v
  [Phase 2]  spark_preprocess.py  (AWS EMR)
  Cleaned & formatted Parquet --> S3 processed/
        |
        v
  [Phase 3]  finetune_tinyllama.ipynb  (Google Colab GPU)
  LoRA fine-tuning --> GGUF q4_k_m --> S3 fine-tuned-models/
        |
        v
  [Phase 4]  setup_ec2.sh  (Terraform + EC2 m5.xlarge)
  Ollama (port 11434) + Open WebUI Docker (port 8080)
        |
        v
       Biomedical Research Chatbot
```

---

## Repository Structure

```
CloudProject-Team19/
|
|-- README.md                        # You are here
|-- .gitignore                       # Python, Terraform, Jupyter ignore rules
|-- requirements.txt                 # Python dependencies for local scripts
|
|-- terraform/                       # Infrastructure as Code
|   |-- main.tf                      # VPC, subnets, security groups, EC2 instance
|   |-- variables.tf                 # Input variable declarations
|   |-- outputs.tf                   # EC2 public IP, instance ID outputs
|   `-- terraform.tfvars.example     # Example variable values (copy -> .tfvars)
|
|-- spark/                           # Data pipeline scripts
|   |-- download_pubmed.py           # Phase 1: Stream PubMed -> S3 raw Parquet
|   `-- spark_preprocess.py          # Phase 2: EMR Spark preprocessing -> S3 processed
|
|-- finetuning/                      # Model training
|   |-- finetune_tinyllama.ipynb     # Phase 3: LoRA fine-tuning notebook (Colab)
|   `-- README.md                    # Colab-specific setup and usage instructions
|
|-- deployment/                      # EC2 deployment automation
|   `-- setup_ec2.sh                 # Phase 4: Full EC2 bootstrap script
|
`-- docs/                            # Supplementary documentation
    |-- architecture.md              # Detailed architecture diagrams & decisions
    `-- troubleshooting.md           # Extended troubleshooting guide
```

---

## Prerequisites

### Local Tools

Ensure the following tools are installed on your local machine before running any phase:

| Tool              | Recommended Version | Purpose                                          |
|-------------------|---------------------|--------------------------------------------------|
| Python            | 3.10.x              | Data ingestion scripts, local testing            |
| pip               | 23.x+               | Python package management                        |
| AWS CLI           | 2.x                 | S3 uploads, EMR job submission, EC2 interaction  |
| Terraform         | 1.7.x+              | Provision AWS infrastructure                     |
| Git               | 2.x+                | Clone and version control                        |
| Docker (optional) | 24.x+               | Local Open WebUI testing                         |
| Ollama (optional) | 0.1.x+              | Local model serving / testing                    |

> **Note:** The fine-tuning notebook (`finetuning/finetune_tinyllama.ipynb`) is designed for
> **Google Colab with a GPU runtime** (T4 or better). Running it locally requires a CUDA-capable
> GPU with at least 12 GB VRAM.

---

### Accounts & Access

| Service          | Requirement                                                              |
|------------------|--------------------------------------------------------------------------|
| AWS Account      | IAM user/role with S3, EC2, and EMR permissions                          |
| AWS S3           | A bucket named `q1abc-cisc886-bucket` (or update paths in all scripts)   |
| AWS EMR          | EMR cluster with PySpark 3.x (Hadoop 3.x) in `us-east-1`                |
| Google Account   | Required for Google Colab GPU runtime access                             |
| HuggingFace Hub  | No token required; `ncbi/pubmed` is publicly accessible (streaming)      |

---

### Environment Requirements

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/CloudProject-Team19.git
cd CloudProject-Team19

# 2. Create and activate a Python virtual environment
python3.10 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# 5. Verify S3 bucket access
aws s3 ls s3://q1abc-cisc886-bucket/
```

> **Security Warning:** Never commit `terraform.tfvars`, `.env` files, or AWS credentials
> to version control. The `.gitignore` excludes these by default. Rotate any accidentally
> exposed keys immediately.

---

<div align="center">

Made with dedication by **Team 19** — CISC 886 Cloud Computing  
Queen's University · 2026

</div>
```
