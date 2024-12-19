# RNN and GRU Implementations

This repository contains implementations of recurrent neural network models and dynamic programming-based solutions for speech-to-phoneme mapping. The repository is divided into two main sections, each focusing on a specific aspect of sequence modeling.

---

## 1. RNN and GRU Cell Implementations

This section demonstrates the implementation of foundational recurrent neural network components from scratch, including:

- **RNN Cells**: Forward and backward pass for sequence modeling.
- **GRU Cells**: Efficient gate-based mechanisms for handling long-term dependencies.
- **CTC Loss**: A specialized loss function for sequence-to-sequence problems.
- **Decoding Strategies**:
  - **Greedy Search**: Simplistic approach to sequence decoding.
  - **Beam Search**: Optimized decoding using probabilistic selection.

### Features
- Custom implementations without relying on autodiff libraries.
- Vectorized operations using NumPy for optimized performance.
- Step-by-step implementations of forward and backward passes.

---

## 2. Speech-to-Phoneme Mapping with CTC Loss

This section focuses on mapping audio utterances to their phonetic representations using RNN-based architectures.

### Dataset
The dataset comprises:
- Feature data includes Mel-Frequency Cepstral Coefficients (MFCCs) with 28 band frequencies per frame.
- Transcripts contain phoneme sequences, including 40 phonemes and a `BLANK` symbol, for alignment tasks.

### Highlights
- **Neural Network Design**:
  - Encoder-decoder architecture utilizing CNNs, BiLSTMs, and pBLSTMs for feature extraction and sequence modeling.
  - Dynamic programming for time-asynchronous output alignment.
- **Decoding Techniques**:
  - Greedy and Beam Search decoding for phoneme sequence generation.
- **Training**:
  - CTC Loss used for alignment-based training.
  - Implementations include both Viterbi and all-alignments training for optimal performance.

### Results
- Achieved an average Levenshtein distance of 5.4, demonstrating high accuracy in phoneme prediction.


---

This repository serves as a comprehensive resource for understanding and implementing sequence modeling and dynamic programming techniques for speech processing tasks.
