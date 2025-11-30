# COM6018 Assignment 2 <!-- omit from toc -->

*Copyright &copy; 2025 Jon Barker, University of Sheffield. All rights reserved*.

21st November 2025. v1.0.0

## Detecting Speed and Tempo Alterations in Speech Recordings <!-- omit from toc -->

## Due: 15:00, Wednesday 17th December 2025 <!-- omit from toc -->

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. The Data](#2-the-data)
- [3. The Task](#3-the-task)
- [4. Additional Rules](#4-additional-rules)
- [5. Assessment](#5-assessment)
- [6. Submission](#6-submission)

## 1. Introduction

This assignment focuses on the development and evaluation of machine-learning systems for detecting temporal modifications in speech recordings. We consider two forms of manipulation:

1. **Speed modification** (using the `sox speed` command), which uniformly changes both pitch and duration.
2. **Tempo modification** (using the `sox tempo` command), which changes duration while preserving pitch.

Your objective is to classify each utterance into one of five rate categories: very slow, slow, normal, fast, or very fast. You will first construct a classifier for speed-modified speech, and then adapt or extend your approach to build a classifier for tempo-modified speech.

The aim is to practise:

- supervised classification with scikit‑learn,
- model selection and hyperparameter optimisation,
- training data augmentation techniques,
- designing robust evaluation,
- clear scientific reporting.

Further details on the data, task, assessment criteria, and submission process are provided below.

## 2. The Data

You are provided with training and evaluation datasets that have been pre-constructed for you using recordings of a single female talker taken from the CMU Arctic Speech Database [Kominek and Black, 2004](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.pdf). There are separate datasets for speed modification and tempo modification. Each contains utterances that have been modified to five different rates: very slow, slow, normal, fast, and very fast. The speech samples are all just 1 second long.

The modifications have used the following speed/tempo factors:

| Category    | Speed Factor | Tempo Factor |
|-------------|--------------|--------------|
| Very Slow   | 0.90         | 0.60         |
| Slow        | 0.95         | 0.80         |
| Normal      | 1.00         | 1.00         |
| Fast        | 1.05         | 1.20         |
| Very Fast   | 1.10         | 1.40         |

A larger range has been used for tempo modification, as this is a more subtle effect.

To avoid the need for any signal processing expertise, the data has been pre-processed into filterbank features that can be used directly with machine learning algorithms. These filterbank features represent the speech as a 2-D time-frequency image, where time is along the horizontal axis and frequency along the vertical axis. The filterbank features are computed using 64 frequency channels and a 10 ms time resolution, so each 1-second speech sample forms a 64x101 'pixel' representation. Each pixel in the image represents the log-amplitude of a particular frequency band at a particular time frame.

Examples of filterbank features for speech segments are shown below.

<img src="latex/speech.png" alt="Log‑Mel spectrogram of a speech segment" width="480" />

The data has also been provided as raw audio signals for those of you who wish to experiment with their own feature extraction methods. In this case, the data is provided as 16 kHz single-channel waveforms. So one second of audio corresponds to 16,000 samples.

<div class="page"/>

### 2.1. Downloading the Data

You can download the data from the following link:

<https://drive.google.com/drive/folders/1Z_ZcrIbshgAFdj0wy8ou8bKlvi6BXHXe>

Download all the files and store them in the assignment project directory in the folder called `data/`.

### 2.2. Training data

There are separate training datasets for speed modification and tempo modification. Each set consists of 4,155 samples (831 speech segments x 5 different speed/tempo categories). Each sample has a corresponding label in [0, 1, 2, 3, 4] representing the speed/tempo category: very slow, slow, normal, fast, very fast.

The data is stored in joblib files called:

- 'fbank_speed.train.joblib' - filterbank features for speed modification
- 'fbank_tempo.train.joblib' - filterbank features for tempo modification.

Or if you wish to use the raw audio signals:

- 'signal_speed.train.joblib' - raw audio signals for speed modification.
- 'signal_tempo.train.joblib' - raw audio signals for tempo modification.

These can be loaded using joblib as follows:

```python
import joblib

data = joblib.load('fbank_speed.train.joblib')
```

The data variable is a dictionary with the following structure:

```python
data = {
    'data': <A 2-D numpy array storing the features>,
    'target': <A 1-D numpy array storing the integer class labels>,
    'filenames': <A list of the original filenames>
}
```

For the filterbank features, the 'data' entry is a 4155 x 6464 numpy array. Each row of the data represents the filterbank features for a single speech segment. The 6464 elements in the row represent a flattened version of a filterbank feature matrix of size 64 (frequency bins) x 101 (time frames).

<div class="page"/>

To view the filterbank features as an image, you can reshape them into a 2D array. For example, to visualise the filterbank features for the first training example, you can do:

```python
import numpy as np
import matplotlib.pyplot as plt
import joblib
data = joblib.load('fbank_speed.train.joblib')
features = data['data'][0].reshape(64, 101)  # Reshape back to 2D
plt.imshow(features, aspect='auto', origin='lower')
plt.show()
```

### 2.3. Evaluation data

The evaluation set consists of 500 examples (100 speech segments x 5 different speed/tempo categories). The data is stored in the following joblib files:

- 'fbank_speed.test1.joblib' - filterbank features for speed modification
- 'fbank_tempo.test1.joblib' - filterbank features for tempo modification.

Or if you wish to use the raw audio signals:

- 'signal_speed.test1.joblib' - raw audio signals for speed modification.
- 'signal_tempo.test1.joblib' - raw audio signals for tempo modification.

Note that the evaluation set uses speech data from the same talker, but none of the utterances in the evaluation set appear in the training set.

The evaluation data can be loaded similarly to the training data:

```python
import joblib

data = joblib.load('fbank_speed.test1.joblib')
```

The data variable is a dictionary with the following structure:

```python
data = {
    'data': <2-D array where each row is a flattened filterbank feature matrix>,
    'target': <1-D array of class labels>,
    'filenames': <A list of the original filenames>
}
```

The target class labels have the same interpretation as for the training data.

There is a second test set, 'test2', which is similar in structure to 'test1' but which is not being released. Your final model will be evaluated on this hidden test set to determine your system's performance.

### 2.4. The Baseline Models

You are provided with an example model that uses the filterbank features. The classifier averages the filterbank frequency channels over time to produce a 64-element vector and then applies a 1-nearest neighbour classifier.

This approach will work fairly well on the speed-modification task but is not expected to perform well on the tempo-modification task, as averaging over time removes all temporal information.

The baseline approach is intentionally simple to allow you to focus on improving the system through better feature engineering, classifier design, and hyperparameter tuning.

## 3. The Task

### 3.1 Developing a Classification Model

You are asked to develop two classification models: one for the speed modification problem and one for the tempo modification problem.  The model training code should be submitted as two scripts called `train_speed.py` and `train_tempo.py`, respectively. The training scripts should save the trained models to joblib files called `model.speed.joblib` and `model.tempo.joblib`, which will also be submitted.

Your training scripts can use either the filterbank features or the raw audio signals as input. We recommend using the filterbank features, unless you have a specific reason to work with the raw audio signals.

The code should run from the command line as follows:

```bash
python train_speed.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME>
```

So, for example, to train your models from the FBANK features you would run:

```bash
python train_speed.py data/fbank_speed.train.joblib model.speed.joblib
python train_tempo.py data/fbank_tempo.train.joblib model.tempo.joblib
```

Example code for training the baseline models is provided in the `src/baseline_fbank` and `src/baseline_signal` directories. You can use this code as a starting point for your own models, but you are free to design your own training scripts from scratch if you prefer.

For example, to train the baseline speed model from the FBANK features, you would run:

```bash
# For speed
python src/baseline_fbank/train_speed.py data/fbank_speed.train.joblib models/baseline_fbank/model.speed.joblib

# For tempo
python src/baseline_fbank/train_tempo.py data/fbank_tempo.train.joblib models/baseline_fbank/model.tempo.joblib
```

Or, if training from the raw audio signals, you would run:

```bash
# For speed
python src/baseline_signal/train_speed.py data/signal_speed.train.joblib models/baseline_signal/model.speed.joblib

# For tempo
python src/baseline_signal/train_tempo.py data/signal_tempo.train.joblib models/baseline_signal/model.tempo.joblib
```

### 3.2 Evaluating Your Models

You are provided with a script `evaluate.py`, which you can use to check that your models work. This script will also be used to assess your system.

To run the evaluation script:

```bash
python src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME>
```

Where

- `<SRC_DIR>` is the source directory containing `train_speed.py` and `train_tempo.py`,
- `<MODEL_FILE_NAME>` is the path to your trained model file
- `<TEST_DATA_FILE_NAME>` is the path to the evaluation data file.

For example, to test the baseline models trained on the FBANK features, you would run:

```bash
# For speed
python src/evaluate.py src/baseline_fbank models/baseline_fbank/model.speed.joblib data/fbank_speed.test1.joblib

# For tempo
python src/evaluate.py src/baseline_fbank models/baseline_fbank/model.tempo.joblib data/fbank_tempo.test1.joblib
```

Or to test the baseline models trained on the audio signals, you would run:

```bash
# For speed
python src/evaluate.py src/baseline_signal models/baseline_signal/model.speed.joblib data/signal_speed.test1.joblib

# For tempo
python src/evaluate.py src/baseline_signal models/baseline_signal/model.tempo.joblib data/signal_tempo.test1.joblib
```

The evaluation script will print out the classification accuracy of your model and display a confusion matrix.

### 3.3. Writing a Report

You are also required to write a report, which you will submit as a PDF file named `report.pdf`. The report should be no more than two pages long and include the following sections:

- **Introduction**: A brief description of the speed and tempo modification detection problem.
- **System Description**:  A complete description of your classification system pipeline, highlighting critical hyperparameters to optimise.
- **Experiments**: A description of the experiments you conducted to tune your system's hyperparameters, including the construction of your training dataset. Be explicit about which hyperparameters you experimented with and how they influenced the performance of your model. Include results that justify your final choices.
- **Results and Analysis**: Provide an analysis of the performance of your final system on the evaluation data, including a comparison to the baseline system. Include a table that reports the accuracy of your model. Additionally, provide a brief discussion of any observed trends or insights, highlighting factors that may have influenced the results.
- **Conclusions**: A summary of your work, including suggestions for further improvements.
- **References**: A list of any references cited in your report.

The report should be written clearly and concisely, using LaTeX. A LaTeX template has been provided. We recommend using Overleaf (<https://www.overleaf.com/>) to write your report, as it simplifies the process of compiling LaTeX documents.

Note, we will not accept reports that exceed two pages in length or that have not used the LaTeX template.

## 4. Additional Rules

Some additional rules have been set to ensure that the assignment is fair for everyone. Please read carefully.

- Neither of your model files must exceed 80 MB in size.
- You can only train your model using the provided training data (but augmentation is allowed).
- You cannot use any pre-trained models, e.g., models from HuggingFace or elsewhere.
- You may only use the standard Python libraries and the packages in the provided pyproject.toml environment.

<div class="page"/>

## 5. Assessment

This assignment is worth 60% of the module mark and will be graded out of 60.

### Evaluation Tips

Here are some tips to help you succeed in this assignment:

- Training Data Augmentation: Consider augmenting your training data by making additional copies with small modifications applied. This can help your model generalise better and avoid overfitting.
- Feature Engineering: Try different ways to preprocess the features, e.g. normalising or scaling them, or averaging over time or frequency.
- Classifier: Decide on a basic approach early and focus on getting the approach working as well as possible. Do not try to implement and report on many different approaches. Depth is better than breadth.
- Hyperparameters: Experiment with different hyperparameters, but be careful not to overfit to the training data. Cross-validation can help you assess generalisability.

The final mark will be based on the following criteria:

- The quality and clarity of your code (20/60)
- The quality and clarity of the written report (30/60)
- The performance of your classifier on a hidden evaluation dataset (10/60)

Further details of the assessment criteria are given below.

### 5.1. Code (20/60)

Marks will be awarded based on the following rubric:

| Score | Description |
| -- | -- |
| Fail (0-9)| The code lacks clarity, organisation, and proper documentation. It may have significant errors, making it challenging to understand or run. Essential machine learning concepts are not demonstrated.|
| Pass (10-11)| The code meets the minimum requirements and demonstrates a basic understanding of scikit-learn. However, the organisation and documentation could be improved for better clarity.|
| Merit (12-13)| The code meets all requirements and shows a good understanding of scikit-learn. It is well-organised, with clear documentation that aids in understanding the implementation.|
| Distinction (14-15)| The code exceeds expectations, demonstrating a thorough understanding of scikit-learn. It is well-organised, follows best practices, and includes comprehensive documentation that enhances clarity.|
| Distinction (16-17)| The code is of exceptional quality, showcasing a high level of mastery. It not only meets all requirements but does so with exceptional clarity, well-structured organisation, and detailed documentation.|
| Exceptional (18-20)| The code is flawless. It demonstrates outstanding clarity, organisation, and documentation. The implementation goes beyond the requirements, showcasing creativity and innovation in classifier design.|

### 5.2. Report (30/60)

Marks will be awarded based on the following rubric:

| Score | Description|
| -- | -- |
| Fail (0-15) | The report lacks a clear and coherent description of the classification system. It does not provide sufficient details for someone to reproduce the work. There are significant issues with the presentation of figures, tables, and referencing. |
| Pass (16-18) | The report offers a basic description of the classification system, but it lacks clarity and coherence. There are areas that could be more detailed. Reproduction of the work is possible with effort. LaTeX is used, but there are some issues with figures, tables, or referencing. |
| Merit (19-21) | The report effectively describes the classification system with clarity and organisation. The details provided make it reasonably easy to reproduce the work. LaTeX is used appropriately, but there may be some minor issues with figures, tables, or referencing. |
| Distinction (22-24) | The report exceeds expectations in clarity and detail, making it easy to reproduce the work. The document has well-designed figures, tables, and accurate referencing. |
| Distinction (25-27) | The report is of exceptional quality, providing a clear and highly detailed description of the classification system. Reproducing the work is straightforward. The document has well-designed figures, tables, and accurate referencing.  |
| Exceptional (28-30) | The report is flawless. It excels in clarity, detail, and reproducibility. The document has well-designed figures, tables, and accurate referencing.  The document is of a publishable quality. |

### 5.3. Performance (10/60)

Marks will be awarded based on the following rubric.

| Score | Description|
| -- | -- |
| Fail (0-4)| The system does not run |
| Pass (5) | The system runs and produces results.|
| Merit (6) | The system runs and produces results that are better than the baseline system.|
| Distinction (7) | The system runs and produces results that are better than the baseline system and you have clearly documented some results showing hyperparameter tuning.|
| Distinction (8) | The system runs and produces results that are better than the baseline system and you have thoroughly investigated the effect of hyperparameter tuning.|
| Exceptional (9 or 10) | As above but with bonus marks for exceptional performance (e.g., top 2 or 3 scores in the class) and/or particularly thorough experimentation.|

## 6. Submission

You will need to submit the following files:

- **`report.pdf`**:  PDF report describing your systems and experiments. Ensure all sections are included and the report is formatted according to LaTeX standards.
- **`train_speed.py`** and **`train_tempo.py`**: The Python scripts that train your classifiers for the speed and tempo problems respectively. Include clear comments explaining the training process and key steps.
- **`model.speed.joblib`** and **`model.tempo.joblib`**: Joblib files containing your trained model for the speed and tempo problems, respectively. These should be the outputs of your trained models and neither must exceed 80 MB.

A submission page will appear on the module's Blackboard site where you will be able to upload the above five files.

**The assignment is due by 15:00 on Wednesday, 17th December 2025. Standard lateness penalties apply**.

## <font color="red">:warning:</font> Academic Misconduct <!-- omit from toc -->

The University takes academic misconduct very seriously. Academic misconduct includes plagiarism, collusion, fabrication of data and inappropriate use of generative AI. You should read the University's guidance on academic integrity to make sure you understand what is and what is not acceptable <https://sheffield.ac.uk/study-skills/assessment/academic-integrity/academic-integrity>. In particular, this is an individual assignment and any suspected collusion will be investigated. Do not share your code with anyone else.

---

*Copyright &copy; 2025 Jon Barker, University of Sheffield. All rights reserved*.
