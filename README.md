# Extreme Weather Event Classification with Transformers and LoRA

An end-to-end Natural Language Processing project that classifies NOAA Storm Events narratives into extreme-weather categories using traditional machine learning and parameter-efficient Transformer fine-tuning.

The project compares a **TF-IDF + Logistic Regression baseline** against **DistilBERT fine-tuned with Low-Rank Adaptation (LoRA)** and evaluates both approaches on a held-out test set.

## Project Overview

Weather agencies produce large volumes of unstructured event narratives describing floods, droughts, tornadoes, wildfires, severe storms, and other hazards. Manually categorizing these reports can be time-consuming when working with large historical archives.

This project develops an automated NLP pipeline that predicts the weather-event category directly from textual NOAA event descriptions.

Example input:

```text
Several hours of intense rainfall caused streams to rise rapidly.
Multiple roads became impassable and low-lying areas were flooded.
```

Example output:

```text
Flash Flood: 91.4%
Flood: 7.3%
Thunderstorm Wind: 0.7%
```

## Objectives

The project was designed to demonstrate a complete applied machine-learning workflow rather than only Transformer fine-tuning.

The main objectives are to:

* ingest multi-year NOAA Storm Events data;
* preprocess unstructured weather-event narratives;
* construct a multi-class NLP classification dataset;
* reduce train/test leakage using episode-based grouped splitting;
* establish a TF-IDF + Logistic Regression baseline;
* fine-tune DistilBERT using parameter-efficient LoRA;
* evaluate models using accuracy, precision, recall, macro-F1, and class-level metrics;
* analyze classification errors using a confusion matrix;
* save the trained LoRA model for future inference and deployment; and
* generate probabilistic predictions for new weather-event narratives.

## Weather Event Classes

The initial model predicts eight event categories:

1. Drought
2. Flash Flood
3. Flood
4. Hail
5. Thunderstorm Wind
6. Tornado
7. Wildfire
8. Winter Storm

The classes can be modified in the project configuration.

## Data

The project uses the **NOAA/NCEI Storm Events Database**, which contains historical records of significant weather events reported across the United States.

The main fields used are:

* `EVENT_TYPE` — target weather-event category;
* `EVENT_NARRATIVE` — description of the individual event;
* `EPISODE_NARRATIVE` — description of the larger meteorological episode;
* `EPISODE_ID` — identifier used for leakage-safe grouped splitting; and
* `EVENT_ID` — unique event identifier.

The script automatically downloads the configured years of NOAA data and stores the raw files locally.

By default:

```python
YEARS = list(range(2018, 2025))
```

## Machine Learning Pipeline

```text
NOAA Storm Events
        |
        v
Automated Data Download
        |
        v
Text Cleaning and Preprocessing
        |
        v
Class Balancing
        |
        v
Episode-Based Train / Validation / Test Split
        |
        +---------------------------+
        |                           |
        v                           v
TF-IDF                       DistilBERT
        |                           |
Logistic Regression               LoRA
        |                           |
        +-------------+-------------+
                      |
                      v
              Model Evaluation
                      |
                      v
       Accuracy / Precision / Recall
              Macro-F1 / Confusion Matrix
                      |
                      v
              Probabilistic Inference
```

## Why Episode-Based Splitting?

A single NOAA meteorological episode can contain multiple related event records.

A standard random row-level split could place similar narratives from the same storm episode in both training and testing data, producing overly optimistic performance estimates.

This project instead uses:

```python
GroupShuffleSplit
```

with `EPISODE_ID`.

All records associated with an episode are therefore assigned entirely to the training, validation, or test dataset.

This provides a more realistic evaluation of model generalization.

## Models

### 1. TF-IDF + Logistic Regression

A traditional NLP baseline is created using:

* unigram and bigram TF-IDF features;
* up to 50,000 text features; and
* Logistic Regression classification.

The baseline establishes how well a relatively simple NLP method performs before introducing a Transformer.

### 2. DistilBERT + LoRA

The advanced model uses:

```text
distilbert-base-uncased
```

DistilBERT is adapted to the weather-event classification task using **Low-Rank Adaptation (LoRA)** through Hugging Face PEFT.

LoRA adapters are applied to the Transformer attention:

```python
target_modules = [
    "q_lin",
    "v_lin"
]
```

This enables parameter-efficient fine-tuning without updating all DistilBERT parameters.

Default LoRA configuration:

```python
r = 8
lora_alpha = 16
lora_dropout = 0.05
```

## Model Evaluation

The project evaluates models using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Class-specific Precision
* Class-specific Recall
* Class-specific F1-score
* Confusion Matrix

**Macro-F1 is used as the primary model-selection metric** because each weather-event category should contribute equally to evaluation.

The best Transformer checkpoint is selected according to:

```python
metric_for_best_model="f1_macro"
```

## Project Structure

```text
extreme-weather-nlp/
|
├── weather_event_classifier.py
├── README.md
├── requirements.txt
|
└── weather_event_project/
    |
    ├── data/
    |   |
    |   ├── raw/
    |   |   └── NOAA Storm Events files
    |   |
    |   ├── train.csv
    |   ├── validation.csv
    |   └── test.csv
    |
    ├── models/
    |   |
    |   └── distilbert_lora/
    |       ├── adapter_config.json
    |       ├── adapter_model.safetensors
    |       └── tokenizer files
    |
    └── results/
        |
        ├── baseline_metrics.json
        ├── transformer_metrics.json
        ├── transformer_classification_report.csv
        ├── model_comparison.csv
        └── confusion_matrix.png
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd extreme-weather-nlp
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n weather-nlp python=3.10
conda activate weather-nlp
```

Alternatively, using Python `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

Run:

```bash
python weather_event_classifier.py
```

The script will automatically:

1. locate NOAA Storm Events files;
2. download the configured years;
3. preprocess the event narratives;
4. balance event classes;
5. generate train, validation, and test datasets;
6. train the TF-IDF baseline;
7. tokenize the narratives using DistilBERT;
8. fine-tune DistilBERT using LoRA;
9. evaluate the final model;
10. generate a confusion matrix;
11. compare the Transformer against the baseline;
12. save the trained model; and
13. generate example predictions.

## GPU Training

The project automatically detects CUDA:

```python
torch.cuda.is_available()
```

When a CUDA-compatible NVIDIA GPU is available, mixed-precision training is enabled automatically.

Example output:

```text
CUDA available: YES
GPU: NVIDIA ...
```

The model can also run on CPU, although Transformer training will be slower.

## Key Training Configuration

The main hyperparameters can be modified near the beginning of the Python script.

```python
MODEL_CHECKPOINT = "distilbert-base-uncased"

MAX_LENGTH = 256

LEARNING_RATE = 2e-4

BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

NUM_EPOCHS = 3
```

The maximum observations per weather-event category can be changed with:

```python
MAX_SAMPLES_PER_CLASS = 4000
```

This is useful for controlling computational requirements during experimentation.

## Example Inference

After training, the model can classify previously unseen narratives:

```python
text = """
Several hours of intense rainfall caused streams to rise rapidly.
Roads became impassable and water entered multiple low-lying areas.
"""

predictions = predict_event(
    text,
    trainer.model,
    tokenizer,
    top_k=3
)
```

Example output:

```text
Flash Flood              91.42%
Flood                     7.31%
Thunderstorm Wind         0.68%
```

Predicted probabilities should be interpreted as model confidence scores rather than guarantees of correctness.

## Output Files

### `model_comparison.csv`

Compares the traditional baseline against the Transformer.

Example structure:

| Model                        | Accuracy | Precision Macro | Recall Macro | F1 Macro |
| ---------------------------- | -------: | --------------: | -----------: | -------: |
| TF-IDF + Logistic Regression |      ... |             ... |          ... |      ... |
| DistilBERT + LoRA            |      ... |             ... |          ... |      ... |

### `transformer_classification_report.csv`

Contains class-level evaluation results for each extreme-weather event.

### `confusion_matrix.png`

Visualizes which event categories are most frequently confused by the Transformer.

Potentially important classification relationships include:

```text
Flood <-> Flash Flood

Winter Storm <-> other winter-related narratives

Thunderstorm Wind <-> Tornado
```

These errors can be investigated through additional qualitative error analysis.

## Technical Skills Demonstrated

This project demonstrates experience with:

* Natural Language Processing
* Transformer architectures
* Transfer learning
* Parameter-efficient fine-tuning
* LoRA / PEFT
* Hugging Face Transformers
* PyTorch
* GPU training
* Multi-class classification
* Feature engineering
* TF-IDF
* Logistic Regression
* Data preprocessing
* Data leakage prevention
* Grouped data splitting
* Model validation
* Model comparison
* Precision / Recall / F1 analysis
* Confusion-matrix analysis
* Probabilistic inference
* Model serialization
* Reproducible ML experimentation

## Potential Extensions

Future versions of the project can include:

* FastAPI REST inference service;
* Docker containerization;
* cloud deployment using AWS;
* MLflow experiment tracking;
* hyperparameter optimization;
* class-weighted Transformer training;
* temporal generalization experiments;
* geographically independent validation;
* alternative pretrained language models;
* model explainability using SHAP or attention analysis; and
* integration with real-time weather-event processing pipelines.

A particularly useful extension would be **temporal validation**, where earlier years are used for training and later years are used for testing. This would evaluate whether the model can generalize to future event reports rather than randomly held-out historical observations.

## Reproducibility

A fixed random seed is used throughout the project:

```python
SEED = 42
```

Random seeds are applied to:

* Python;
* NumPy; and
* PyTorch.

The same seed is also supplied to Hugging Face training and dataset splitting where applicable.

Exact GPU-level reproducibility can still vary depending on CUDA and hardware implementations.

## Disclaimer

This project is intended for machine-learning research, portfolio development, and educational analysis.

Predictions produced by the model should not be used as official weather warnings, emergency guidance, or replacements for information issued by NOAA, the National Weather Service, or emergency-management authorities.

## Author

**Juan Rojas**

Ph.D. Student in Computing — Data Science Emphasis
Research interests include machine learning, deep learning, geospatial artificial intelligence, remote sensing, environmental data science, and spatiotemporal modeling.
