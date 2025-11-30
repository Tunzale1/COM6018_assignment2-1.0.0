# COM6018 Assignment 2

## The Materials

The directory contains the following files:

```bash
├── assignment2.v1_0.md  # Assignment brief in markdown format
├── assignment2.v1_0.pdf # Assignment brief in pdf format
├── COM6018_latex_template.v1_0.zip  # LaTeX report template
├── data/   # Directory in which to store the data files
├── models/  
|   ├── baseline_fbank/  # Baseline models trained on filterbank features  
│   └── baseline_signal/  # Baseline models trained on raw signal
├── pyproject.toml  # Packages that you are allowed to use 
├── README.md
└── src
    ├── __init__.py   
    ├── baseline_fbank/    # Code for training baseline (from filterbanks)
    ├── baseline_signal/  # Code for training baseline (from signals)
    ├── demo.ipynb  # A demo notebook showing how to use the data
    └── evaluate.py  # Code for evaluating a model on the test set
```

## Installation

### The Assignment Data

The data can be downloaded from:

<https://drive.google.com/drive/folders/1Z_ZcrIbshgAFdj0wy8ou8bKlvi6BXHXe>

Copy the data files into the directory called `data` in the root of the assignment directory.

### The Python Environment

Make a separate uv environment for the assignment, e.g., using uv:

```bash
uv sync
```

And then activate the environment on mac/linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

## The Assignment Details

For full instructions, please read  the assignment brief in `assignment2.v1_0.md` and `assignment2.v1_0.pdf`.
