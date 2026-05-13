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
