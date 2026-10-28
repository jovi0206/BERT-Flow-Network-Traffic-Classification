# BERT Flow Network Traffic Classification

A simple BERT-based network traffic classification project using VPN and NonVPN datasets.

## Project Overview

This project was developed as part of the **High Performance Internet Architecture and Design** course at National Cheng Kung University (NCKU).

The objective was to explore BERT-based network traffic classification through two independent experiments using VPN and NonVPN traffic datasets.

## Workflow

1. PCAP packet parsing
2. Flow construction and feature extraction
3. Dataset preprocessing and JSON generation
4. BERT-based model training using PyTorch
5. Model evaluation and confusion matrix visualization
6. Comparison of VPN and NonVPN experimental results

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Scapy
- Scikit-learn
- CUDA GPU Training

## Experimental Results

| Metric | VPN | NonVPN |
|---|---:|---:|
| Accuracy | 77.46% | 37.09% |
| Weighted F1-score | 76.34% | 20.07% |

The VPN experiment demonstrated better classification performance, while the NonVPN experiment showed predictions concentrated in a single class.

These results highlight the importance of analyzing class distribution, model limitations, and evaluation metrics beyond accuracy.

## Project Structure

- `VPN/` — VPN traffic classification experiment
- `NonVPN/` — NonVPN traffic classification experiment

Each directory contains preprocessing, model training, evaluation, and utility scripts.

## Notes

This repository documents a course-based experiment rather than a production-ready traffic classification system.

The BERT-based model was initialized from configuration and trained on flow features, rather than using pretrained BERT language-model weights.

The experiments were based on a framework and preprocessing scripts provided by the course teaching assistant. The work included environment configuration, model training, debugging, evaluation, and experimental analysis.