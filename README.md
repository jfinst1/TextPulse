# TextPulse

**TextPulse** is an advanced text file recovery tool powered by a hybrid quantum-classical neural network (eQ-DIMON). It processes corrupted or incomplete text files, reconstructing them chunk-by-chunk using entropy-based corruption detection, n-gram context modeling, and user feedback. Designed for real-world chaos, it supports full byte range (0–255), adaptive chunk sizing, and persistent learning via a memory playbook.

## Features

- **Real File Recovery**: Processes actual corrupted text files—no simulations.
- **Entropy-Based Corruption Detection**: Identifies damaged regions using local entropy analysis.
- **Full Byte Range**: Handles ASCII (0–255), including control characters and basic UTF-8 resilience.
- **User Feedback**: Interactive mask refinement for precise corruption marking.
- **Adaptive Chunking**: Scales chunk size (32×32 to 128×128) based on file length.
- **Anomaly Visualization**: Generates heatmaps of inferred corruption.
- **Memory Playbook**: Retains knowledge across runs for improved future recoveries.

## Installation

### Prerequisites

- Python 3.8+

### Required Libraries

```bash
pip install torch pennylane numpy scipy matplotlib python-Levenshtein
```

### Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/textpulse.git
cd textpulse
```

Install dependencies:

```bash
pip install -r requirements.txt
```

*(Create a `requirements.txt` file with the required libraries listed above if needed.)*

## Usage

### Running TextPulse

1. Prepare a corrupted text file (e.g., `corrupted.txt`).
2. Run the script:

```bash
python textpulse.py
```

3. Enter the file path when prompted (e.g., `corrupted.txt`).

### For Each Chunk:

- View the chunk and inferred corruption mask.
- Respond to prompts:
  - **`y/n`** to adjust the mask.
  - Enter good/bad region indices (e.g., `0-10` for good, `15-25` for bad, or `none`).

### Check Outputs:

- `recovered_text.txt`: The reconstructed text.
- `memory/anomaly_user_input.png`: Corruption heatmap for the first chunk.

## Example

```bash
Enter the path to your text file: corrupted.txt
INFO: Loading and chunking user file...
INFO: File split into 3 chunks with N=64.
INFO: Training on user-input chunks...
Chunk 0: Adjust mask? (y/n): y
Enter good region indices (e.g., '0-10,20-30' or 'none'): 0-20
Enter bad region indices (e.g., '15-25' or 'none'): none
...
INFO: Recovered text saved to recovered_text.txt
```

## How It Works

- **Chunking**: Splits the file into overlapping chunks (e.g., 64×64), sized adaptively.
- **Corruption Detection**: Uses entropy to flag low-variability or extreme-value regions.
- **Training**: Fine-tunes a quantum-enhanced neural network (EnhancedMIONet) on each chunk.
- **Prediction**: Reconstructs chunks, blending overlaps for seamless output.
- **Memory**: Stores models and playbook in `memory/` for future tasks.

## Future Enhancements

- Advanced corruption classifiers (e.g., ML-based anomaly detection).
- Support for binary file recovery beyond text.
- GUI for easier user feedback.

## License

This project is licensed under my craziness. See the `LICENSE` file for details.
