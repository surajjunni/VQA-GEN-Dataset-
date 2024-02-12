# VQA-GEN: A Visual Question Answering Benchmark for Domain Generalization

## Overview

This project contains the code and data for the research paper "VQA-GEN: A Visual Question Answering Benchmark for Domain Generalization".

## What's Included

- **code/:** Python programs for making the VQA-GEN dataset
  - `Artistic_Style_Shifts.py:` Applies artistic filters to images
  - `Conversational_shift.py:` Generates different versions of questions
  - `Noise_Injection_Shifts.py:` Adds noise effects to images
  - `Translational_Shift.py:` Translates questions into other languages and back to English

- **data/:**
  - `VQA-GEN.csv:` The generated dataset
  - `model/:` Pre-trained visual question answering models

- `best_model.pth.filepart:` A trained VQA model checkpoint
- `README.md:` This file

## Required Software

The code needs these Python libraries:

- PyTorch
- torchvision
- numpy
- pandas

## How to Use

### To make the VQA-GEN dataset:

```bash
python Noise_Injection_Shifts.py
python Artistic_Style_Shifts.py
python Translational_Shift.py 
python Conversational_shift.py

## Usage

The output `VQA-GEN.csv` file will contain the generated image-question pairs.

To test models on VQA-GEN:

```bash
python evaluate.py --model {ViLT, MAC, RelNet} --data VQA-GEN.csv

## Citing this Work

If you use this dataset, please cite the paper:

```bibtex
@article{VQA-GEN,
  title={VQA-GEN: A Visual Question Answering Benchmark for Domain Generalization},
  author={Unni, Suraj Jyothi and Moraffah, Raha and Liu, Huan},
  journal={arXiv preprint arXiv:2311.00807},
  year={2023}
}
