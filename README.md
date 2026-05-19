# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

wandb report Link : https://api.wandb.ai/links/cs25m010-iitmaana/frqu0qkk
Github link : https://github.com/Anirudh796/da6401_assignment_3

Note regarding W&B plots and run filters(IMPORTANT)

I have applied section-wise run filters in the W&B report so that each section shows only the runs relevant to that experiment. The plots appear correctly when I open the report link while logged into my W&B account.

However, when I open the same report link in incognito mode, it seems that the run filters may not be preserved, and plots from all runs sometimes appear together in each section. To avoid any ambiguity during evaluation, please consider only the following runs for each section:

Section 2.1:
Please include only the runs part2_1_overlay, fixed_lr, and noam. The remaining runs may be disabled using the eye icon next to each run.

Section 2.2:
Please include only the runs part2_2_overlay, scaling_off, and scaling_on. The remaining runs may be disabled.

Section 2.3:
Please include only the run part2_3_attention_rollout. The remaining runs may be disabled.

Section 2.4:
Please include only the runs part2_4_overlay, learned_pos, and sinusoidal. The remaining runs may be disabled.

Section 2.5:
Please include only the runs part2_5_overlay, ls_0_0, and ls_0_1. The remaining runs may be disabled.

I apologize for the inconvenience. I am adding this note only to make sure the intended plots for each section are clear during evaluation.

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
