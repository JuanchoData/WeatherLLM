# ================================================================
# EXTREME WEATHER EVENT CLASSIFICATION WITH TRANSFORMERS + LoRA
# ================================================================
#
# Project:
# Automatically classify NOAA Storm Event narratives into
# extreme-weather categories using NLP.
#
# Models:
#   1. TF-IDF + Logistic Regression baseline
#   2. DistilBERT + LoRA Transformer
#
# Main technologies:
#   Python
#   Pandas
#   Scikit-learn
#   PyTorch
#   Hugging Face Transformers
#   Hugging Face Datasets
#   PEFT / LoRA
#
# Author:
#   Juan Rojas
#
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import re
import json
import random
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from datasets import Dataset, DatasetDict

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel
)


# ================================================================
# 2. PROJECT CONFIGURATION
# ================================================================

# ------------------------------------------------
# Random seed
# ------------------------------------------------
# Using a fixed seed makes the experiment more reproducible.

SEED = 42


# ------------------------------------------------
# Years of NOAA data
# ------------------------------------------------
# Start with 2018-2024.
#
# This gives us several years of real observations while keeping
# the training dataset manageable.
#
# You can later expand this range.

YEARS = list(range(2018, 2025))


# ------------------------------------------------
# Weather-event classes
# ------------------------------------------------
# We convert the original NOAA problem into an
# 8-class text-classification problem.

SELECTED_EVENTS = [
    "Drought",
    "Flash Flood",
    "Flood",
    "Hail",
    "Thunderstorm Wind",
    "Tornado",
    "Wildfire",
    "Winter Storm"
]


# ------------------------------------------------
# Maximum samples per class
# ------------------------------------------------
# Some NOAA event types occur much more frequently than others.
#
# We limit the number of observations per class to:
#
#     4,000
#
# The code will automatically use the size of the smallest class
# if it has fewer than 4,000 observations.

MAX_SAMPLES_PER_CLASS = 4000


# ------------------------------------------------
# Minimum text length
# ------------------------------------------------
# Very short narratives usually contain little useful information.

MIN_TEXT_LENGTH = 40


# ------------------------------------------------
# Transformer configuration
# ------------------------------------------------

MODEL_CHECKPOINT = "distilbert-base-uncased"

MAX_LENGTH = 256

LEARNING_RATE = 2e-4

BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

NUM_EPOCHS = 3


# ================================================================
# 3. PROJECT DIRECTORIES
# ================================================================

PROJECT_DIR = Path("weather_event_project")

DATA_DIR = PROJECT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR = PROJECT_DIR / "models" / "distilbert_lora"


# Create folders if they do not already exist.

for directory in [
    PROJECT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ================================================================
# 4. REPRODUCIBILITY
# ================================================================

def set_seed(seed=42):

    """
    Set random seeds for Python, NumPy, and PyTorch.

    This helps make experiments reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ================================================================
# 5. CHECK GPU
# ================================================================

print("\n================================================")
print("COMPUTING DEVICE")
print("================================================")

if torch.cuda.is_available():

    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))

else:

    print("CUDA available: NO")
    print("Training will use CPU.")


# ================================================================
# 6. NOAA DATA LOCATION
# ================================================================
#
# NOAA periodically updates the filenames.
#
# Instead of hard-coding something such as:
#
# StormEvents_details....2024....csv.gz
#
# we inspect the NOAA directory and automatically identify the
# newest version for each year.
#
# Splitting the host and path also makes this easier to modify.

NOAA_HOST = "www.ncei.noaa.gov"

NOAA_PATH = "/pub/data/swdi/stormevents/csvfiles/"

NOAA_BASE = "https://" + NOAA_HOST + NOAA_PATH


# ================================================================
# 7. FIND THE LATEST NOAA FILE FOR EACH YEAR
# ================================================================

def find_noaa_files(years):

    """
    Find the newest NOAA Storm Events details file for each year.

    NOAA filenames contain two important parts:

        dYYYY = data year
        cYYYYMMDD = date NOAA created/updated the file

    Example conceptually:

        StormEvents_details_..._d2024_c2026xxxx.csv.gz

    We always choose the newest available version.
    """

    print("\n================================================")
    print("SEARCHING NOAA DATA ARCHIVE")
    print("================================================")

    response = requests.get(
        NOAA_BASE,
        timeout=60
    )

    response.raise_for_status()

    html = response.text

    files = {}

    for year in years:

        pattern = (
            rf"StormEvents_details-ftp_v1\.0_"
            rf"d{year}_c\d+\.csv\.gz"
        )

        matches = re.findall(
            pattern,
            html
        )

        matches = sorted(
            set(matches)
        )

        if len(matches) == 0:

            raise FileNotFoundError(
                f"No NOAA file found for {year}"
            )

        # The last filename corresponds to the latest creation date.
        latest_file = matches[-1]

        files[year] = latest_file

        print(
            f"{year}: {latest_file}"
        )

    return files


# ================================================================
# 8. DOWNLOAD NOAA DATA
# ================================================================

def download_file(filename):

    """
    Download a NOAA file only if it has not already been downloaded.

    This creates a local cache.

    Therefore, when the script is executed again, it does not need
    to download all files again.
    """

    local_path = RAW_DATA_DIR / filename

    if local_path.exists():

        print(
            f"Already downloaded: {filename}"
        )

        return local_path

    file_url = NOAA_BASE + filename

    print(
        f"Downloading: {filename}"
    )

    response = requests.get(
        file_url,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    with open(
        local_path,
        "wb"
    ) as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)

    return local_path


# ================================================================
# 9. LOAD NOAA DATA
# ================================================================

def load_noaa_data(years):

    """
    Download and combine NOAA Storm Events data.

    We only load the columns needed for this NLP project.

    EVENT_TYPE:
        Classification target.

    EVENT_NARRATIVE:
        Description of the individual event.

    EPISODE_NARRATIVE:
        Description of the larger weather episode.

    EPISODE_ID:
        Used later to prevent data leakage.

    EVENT_ID:
        Unique NOAA event identifier.
    """

    files = find_noaa_files(years)

    dataframes = []

    columns = [
        "EVENT_ID",
        "EPISODE_ID",
        "EVENT_TYPE",
        "EPISODE_NARRATIVE",
        "EVENT_NARRATIVE"
    ]

    print("\n================================================")
    print("READING NOAA DATA")
    print("================================================")

    for year, filename in files.items():

        path = download_file(filename)

        df_year = pd.read_csv(
            path,
            compression="gzip",
            usecols=columns,
            low_memory=False
        )

        df_year["YEAR"] = year

        dataframes.append(
            df_year
        )

        print(
            year,
            "rows:",
            len(df_year)
        )

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("\nTotal NOAA observations:", len(df))

    return df


# ================================================================
# 10. PREPROCESS TEXT
# ================================================================

def preprocess_data(df):

    """
    Prepare NOAA text for machine learning.

    Steps:

    1. Keep selected event types.
    2. Handle missing narratives.
    3. Combine episode and event narratives.
    4. Remove very short observations.
    5. Remove exact duplicates.
    6. Create a group ID for leakage-safe splitting.
    """

    print("\n================================================")
    print("PREPROCESSING DATA")
    print("================================================")

    # ------------------------------------------------------------
    # Keep only selected weather hazards
    # ------------------------------------------------------------

    df = df[
        df["EVENT_TYPE"].isin(
            SELECTED_EVENTS
        )
    ].copy()


    # ------------------------------------------------------------
    # Replace missing narratives with empty strings
    # ------------------------------------------------------------

    df["EPISODE_NARRATIVE"] = (
        df["EPISODE_NARRATIVE"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["EVENT_NARRATIVE"] = (
        df["EVENT_NARRATIVE"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    # ------------------------------------------------------------
    # Build model input
    # ------------------------------------------------------------
    #
    # We combine:
    #
    #     episode context
    #
    # with:
    #
    #     individual event description
    #
    # This gives the Transformer more context.

    df["text"] = (
        "Episode context: "
        + df["EPISODE_NARRATIVE"]
        + " Event report: "
        + df["EVENT_NARRATIVE"]
    )


    # Clean repeated whitespace.

    df["text"] = (
        df["text"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


    # ------------------------------------------------------------
    # Remove very short descriptions
    # ------------------------------------------------------------

    df = df[
        df["text"].str.len() >= MIN_TEXT_LENGTH
    ].copy()


    # ------------------------------------------------------------
    # Remove exact duplicates
    # ------------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "text",
            "EVENT_TYPE"
        ]
    )


    # ------------------------------------------------------------
    # Create group ID
    # ------------------------------------------------------------
    #
    # This is VERY IMPORTANT.
    #
    # NOAA can contain multiple events belonging to the same larger
    # meteorological episode.
    #
    # If related events appear in both training and testing data,
    # our accuracy could be artificially optimistic.
    #
    # We therefore split using EPISODE_ID.
    #
    # If EPISODE_ID is missing, EVENT_ID becomes the group.

    episode_group = (
        "episode_"
        + df["EPISODE_ID"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    event_group = (
        "event_"
        + df["EVENT_ID"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    df["group_id"] = np.where(
        df["EPISODE_ID"].notna(),
        episode_group,
        event_group
    )


    print("\nAvailable observations by class:")

    print(
        df["EVENT_TYPE"]
        .value_counts()
        .sort_index()
    )

    return df


# ================================================================
# 11. BALANCE THE DATASET
# ================================================================

def balance_dataset(df):

    """
    Build a balanced classification dataset.

    Without this step, very common hazards such as Hail or
    Thunderstorm Wind could dominate training.

    We use the same number of observations for every class.
    """

    counts = (
        df["EVENT_TYPE"]
        .value_counts()
    )

    # Determine the smallest class.

    smallest_class = counts[
        SELECTED_EVENTS
    ].min()

    samples_per_class = min(
        smallest_class,
        MAX_SAMPLES_PER_CLASS
    )

    samples_per_class = int(
        samples_per_class
    )

    print("\n================================================")
    print("BALANCING DATA")
    print("================================================")

    print(
        "Samples per class:",
        samples_per_class
    )

    balanced_parts = []

    for event in SELECTED_EVENTS:

        subset = df[
            df["EVENT_TYPE"] == event
        ]

        subset = subset.sample(
            n=samples_per_class,
            random_state=SEED
        )

        balanced_parts.append(
            subset
        )

    df_balanced = pd.concat(
        balanced_parts,
        ignore_index=True
    )


    # Shuffle dataset.

    df_balanced = df_balanced.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


    print(
        "\nBalanced dataset size:",
        len(df_balanced)
    )

    print(
        df_balanced["EVENT_TYPE"]
        .value_counts()
    )

    return df_balanced


# ================================================================
# 12. CREATE TRAIN / VALIDATION / TEST SPLITS
# ================================================================

def split_dataset(df):

    """
    Create leakage-safe splits.

    Desired approximate split:

        Training     = 70%
        Validation   = 10%
        Test         = 20%

    IMPORTANT:

    We split by NOAA EPISODE rather than by individual row.

    This helps prevent related narratives from the same storm
    episode from appearing in both training and testing data.
    """

    print("\n================================================")
    print("CREATING GROUPED DATA SPLITS")
    print("================================================")


    # ------------------------------------------------------------
    # First split:
    #
    # 80% train + validation
    # 20% test
    # ------------------------------------------------------------

    splitter_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_val_idx, test_idx = next(
        splitter_test.split(
            df,
            groups=df["group_id"]
        )
    )

    train_val = df.iloc[
        train_val_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()


    # ------------------------------------------------------------
    # Second split
    #
    # Validation is 12.5% of the remaining 80%.
    #
    # 0.125 × 0.80 = 0.10
    #
    # Therefore:
    #
    # train ≈ 70%
    # validation ≈ 10%
    # test ≈ 20%
    # ------------------------------------------------------------

    splitter_val = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=SEED
    )

    train_idx, val_idx = next(
        splitter_val.split(
            train_val,
            groups=train_val["group_id"]
        )
    )

    train = train_val.iloc[
        train_idx
    ].copy()

    validation = train_val.iloc[
        val_idx
    ].copy()


    print("Training:", len(train))

    print("Validation:", len(validation))

    print("Testing:", len(test))


    # ------------------------------------------------------------
    # Check for episode leakage
    # ------------------------------------------------------------

    train_groups = set(
        train["group_id"]
    )

    val_groups = set(
        validation["group_id"]
    )

    test_groups = set(
        test["group_id"]
    )

    assert train_groups.isdisjoint(
        val_groups
    )

    assert train_groups.isdisjoint(
        test_groups
    )

    assert val_groups.isdisjoint(
        test_groups
    )

    print("\nEpisode leakage check: PASSED")


    print("\nTraining distribution:")

    print(
        train["EVENT_TYPE"]
        .value_counts()
    )


    print("\nValidation distribution:")

    print(
        validation["EVENT_TYPE"]
        .value_counts()
    )


    print("\nTest distribution:")

    print(
        test["EVENT_TYPE"]
        .value_counts()
    )

    return train, validation, test


# ================================================================
# 13. CREATE LABEL MAPPINGS
# ================================================================

label2id = {
    label: i
    for i, label in enumerate(
        SELECTED_EVENTS
    )
}

id2label = {
    i: label
    for label, i in label2id.items()
}


# ================================================================
# 14. ASSIGN NUMERIC LABELS
# ================================================================

def add_labels(
    train,
    validation,
    test
):

    for df in [
        train,
        validation,
        test
    ]:

        df["labels"] = (
            df["EVENT_TYPE"]
            .map(label2id)
        )

    return (
        train,
        validation,
        test
    )


# ================================================================
# 15. BASELINE MODEL
# ================================================================
#
# Before using a Transformer, we should establish a traditional
# machine-learning baseline.
#
# This is important because:
#
#     "Transformer accuracy = 90%"
#
# means much less without knowing whether a simple method
# already obtains 89%.
#
# Our baseline:
#
#     TF-IDF
#       +
#     Logistic Regression
#

def train_baseline(
    train,
    test
):

    print("\n================================================")
    print("BASELINE: TF-IDF + LOGISTIC REGRESSION")
    print("================================================")


    # ------------------------------------------------------------
    # Convert text into TF-IDF features
    # ------------------------------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


    X_train = vectorizer.fit_transform(
        train["text"]
    )

    X_test = vectorizer.transform(
        test["text"]
    )


    y_train = train["labels"].values

    y_test = test["labels"].values


    # ------------------------------------------------------------
    # Train Logistic Regression
    # ------------------------------------------------------------

    baseline_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    baseline_model.fit(
        X_train,
        y_train
    )


    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    predictions = baseline_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    baseline_results = {

        "model":
            "TF-IDF + Logistic Regression",

        "accuracy":
            float(accuracy),

        "precision_macro":
            float(precision),

        "recall_macro":
            float(recall),

        "f1_macro":
            float(f1)
    }


    print("\nBaseline results:")

    for metric, value in baseline_results.items():

        print(
            metric,
            ":",
            value
        )


    # Save baseline results.

    with open(
        RESULTS_DIR / "baseline_metrics.json",
        "w"
    ) as file:

        json.dump(
            baseline_results,
            file,
            indent=4
        )


    return baseline_results


# ================================================================
# 16. CONVERT PANDAS TO HUGGING FACE DATASET
# ================================================================

def create_huggingface_dataset(
    train,
    validation,
    test
):

    """
    Convert Pandas DataFrames into Hugging Face Datasets.
    """

    columns = [
        "text",
        "labels"
    ]


    dataset = DatasetDict({

        "train":
            Dataset.from_pandas(
                train[columns],
                preserve_index=False
            ),

        "validation":
            Dataset.from_pandas(
                validation[columns],
                preserve_index=False
            ),

        "test":
            Dataset.from_pandas(
                test[columns],
                preserve_index=False
            )
    })

    return dataset


# ================================================================
# 17. LOAD TOKENIZER
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT
)


# ================================================================
# 18. TOKENIZATION FUNCTION
# ================================================================

def tokenize_function(examples):

    """
    Convert text into DistilBERT tokens.

    We use truncation because some NOAA narratives are long.

    We DO NOT manually pad here.

    Padding is performed dynamically by DataCollatorWithPadding.
    """

    return tokenizer(

        examples["text"],

        truncation=True,

        max_length=MAX_LENGTH
    )


# ================================================================
# 19. TOKENIZE DATASET
# ================================================================

def tokenize_dataset(dataset):

    print("\n================================================")
    print("TOKENIZING DATA")
    print("================================================")

    tokenized_dataset = dataset.map(

        tokenize_function,

        batched=True,

        remove_columns=[
            "text"
        ]
    )

    return tokenized_dataset


# ================================================================
# 20. DYNAMIC PADDING
# ================================================================
#
# Sentences have different lengths.
#
# Instead of padding every example to 256 tokens,
# we only pad to the longest sequence in each batch.
#
# This reduces wasted GPU memory.

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ================================================================
# 21. EVALUATION FUNCTION
# ================================================================

def compute_metrics(eval_pred):

    """
    Calculate classification metrics during training.

    Macro-F1 is the primary metric because every weather class
    receives equal importance.
    """

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )


    accuracy = accuracy_score(
        labels,
        predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            labels,

            predictions,

            average="macro",

            zero_division=0
        )
    )


    return {

        "accuracy":
            accuracy,

        "precision_macro":
            precision,

        "recall_macro":
            recall,

        "f1_macro":
            f1
    }


# ================================================================
# 22. CREATE DISTILBERT MODEL
# ================================================================

def create_model():

    """
    Load pretrained DistilBERT and configure it for 8-class
    sequence classification.
    """

    base_model = (
        AutoModelForSequenceClassification
        .from_pretrained(

            MODEL_CHECKPOINT,

            num_labels=len(
                SELECTED_EVENTS
            ),

            id2label=id2label,

            label2id=label2id
        )
    )

    return base_model


# ================================================================
# 23. CONFIGURE LoRA
# ================================================================

def apply_lora(model):

    """
    Apply Low-Rank Adaptation to DistilBERT.

    Instead of updating all ~66 million parameters,
    LoRA trains small low-rank matrices attached to selected
    Transformer attention layers.

    q_lin:
        Query projection.

    v_lin:
        Value projection.

    r:
        Rank of the LoRA matrices.

    lora_alpha:
        Controls the magnitude of the LoRA update.

    modules_to_save:
        Classification layers that also need to be trained and saved.
    """

    peft_config = LoraConfig(

        task_type=TaskType.SEQ_CLS,

        inference_mode=False,

        r=8,

        lora_alpha=16,

        lora_dropout=0.05,

        target_modules=[
            "q_lin",
            "v_lin"
        ],

        bias="none",

        modules_to_save=[
            "pre_classifier",
            "classifier"
        ]
    )


    model = get_peft_model(
        model,
        peft_config
    )


    print("\n================================================")
    print("LoRA TRAINABLE PARAMETERS")
    print("================================================")

    model.print_trainable_parameters()


    return model


# ================================================================
# 24. TRAIN TRANSFORMER
# ================================================================

def train_transformer(
    tokenized_dataset
):

    print("\n================================================")
    print("DISTILBERT + LoRA TRAINING")
    print("================================================")


    model = create_model()

    model = apply_lora(
        model
    )


    # ------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------
    #
    # If CUDA is available, FP16 usually makes Transformer
    # training faster and reduces GPU memory usage.

    use_fp16 = torch.cuda.is_available()


    training_args = TrainingArguments(

        # Where checkpoints are stored
        output_dir=str(
            MODEL_DIR
        ),

        # LoRA can generally use a higher LR than full fine-tuning.
        learning_rate=LEARNING_RATE,

        # Batch size
        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        # Simulates a larger effective batch size.
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # Training epochs
        num_train_epochs=NUM_EPOCHS,

        # Regularization
        weight_decay=0.01,

        # Learning-rate warmup
        warmup_steps=0.10,

        # Evaluate once per epoch
        eval_strategy="epoch",

        # Save once per epoch
        save_strategy="epoch",

        # Keep only the most useful checkpoints
        save_total_limit=2,

        # Restore best model based on validation Macro-F1
        load_best_model_at_end=True,

        metric_for_best_model="f1_macro",

        greater_is_better=True,

        # GPU mixed precision
        fp16=use_fp16,

        # Log training progress
        logging_steps=50,

        # Prevent integrations such as WandB from starting
        report_to="none",

        # Reproducibility
        seed=SEED,

        data_seed=SEED,

        # Windows-safe setting
        dataloader_num_workers=0,

        # Important when PEFT wraps the original model
        label_names=[
            "labels"
        ]
    )


    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset["validation"]
        ),

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )


    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------

    trainer.train()


    # ------------------------------------------------------------
    # Save LoRA adapter
    # ------------------------------------------------------------

    model.save_pretrained(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )


    print(
        "\nModel saved to:",
        MODEL_DIR
    )

    return trainer


# ================================================================
# 25. FINAL TEST SET EVALUATION
# ================================================================

def evaluate_transformer(
    trainer,
    tokenized_dataset
):

    print("\n================================================")
    print("FINAL TRANSFORMER TEST EVALUATION")
    print("================================================")


    output = trainer.predict(
        tokenized_dataset["test"]
    )


    logits = output.predictions

    labels = output.label_ids


    predictions = np.argmax(
        logits,
        axis=-1
    )


    # ------------------------------------------------------------
    # Basic metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            labels,

            predictions,

            average="macro",

            zero_division=0
        )
    )


    transformer_results = {

        "model":
            "DistilBERT + LoRA",

        "accuracy":
            float(accuracy),

        "precision_macro":
            float(precision),

        "recall_macro":
            float(recall),

        "f1_macro":
            float(f1)
    }


    print("\nTransformer results:")

    for metric, value in transformer_results.items():

        print(
            metric,
            ":",
            value
        )


    # ------------------------------------------------------------
    # Detailed classification report
    # ------------------------------------------------------------

    print("\n================================================")
    print("CLASSIFICATION REPORT")
    print("================================================")


    report = classification_report(

        labels,

        predictions,

        labels=list(
            range(
                len(SELECTED_EVENTS)
            )
        ),

        target_names=SELECTED_EVENTS,

        zero_division=0,

        output_dict=True
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    print(
        report_df
    )


    # Save report.

    report_df.to_csv(
        RESULTS_DIR /
        "transformer_classification_report.csv"
    )


    # Save overall metrics.

    with open(

        RESULTS_DIR /
        "transformer_metrics.json",

        "w"

    ) as file:

        json.dump(

            transformer_results,

            file,

            indent=4
        )


    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )


    ConfusionMatrixDisplay.from_predictions(

        labels,

        predictions,

        display_labels=SELECTED_EVENTS,

        xticks_rotation=45,

        ax=ax
    )


    ax.set_title(
        "DistilBERT + LoRA Weather Event Classification"
    )


    plt.tight_layout()


    confusion_path = (
        RESULTS_DIR /
        "confusion_matrix.png"
    )


    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nConfusion matrix saved:",
        confusion_path
    )


    return transformer_results


# ================================================================
# 26. COMPARE BASELINE VS TRANSFORMER
# ================================================================

def compare_models(
    baseline_results,
    transformer_results
):

    comparison = pd.DataFrame([

        baseline_results,

        transformer_results

    ])


    print("\n================================================")
    print("MODEL COMPARISON")
    print("================================================")

    print(
        comparison
    )


    comparison.to_csv(

        RESULTS_DIR /
        "model_comparison.csv",

        index=False
    )


# ================================================================
# 27. PREDICT A NEW WEATHER NARRATIVE
# ================================================================

def predict_event(
    text,
    model,
    tokenizer,
    top_k=3
):

    """
    Classify a new weather narrative.

    Returns the TOP-K most probable classes rather than only
    the most likely class.

    Example:

        Flash Flood       0.91
        Flood             0.07
        Thunderstorm Wind 0.01
    """

    model.eval()


    # Find the device where the model currently lives.

    device = next(
        model.parameters()
    ).device


    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_LENGTH,

        padding=True
    )


    # Move input tensors to CPU/GPU.

    inputs = {

        key:
            value.to(device)

        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    # Convert logits to probabilities.

    probabilities = torch.softmax(

        outputs.logits,

        dim=-1

    )[0]


    # Get top K classes.

    top_probabilities, top_indices = torch.topk(

        probabilities,

        k=top_k
    )


    results = []


    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        class_id = index.item()

        results.append({

            "event_type":
                id2label[class_id],

            "confidence":
                probability.item()
        })


    return results


# ================================================================
# 28. MAIN PROJECT PIPELINE
# ================================================================

def main():

    print("\n================================================")
    print("EXTREME WEATHER NLP PROJECT")
    print("================================================")


    # ------------------------------------------------------------
    # STEP 1
    # Download NOAA data
    # ------------------------------------------------------------

    df = load_noaa_data(
        YEARS
    )


    # ------------------------------------------------------------
    # STEP 2
    # Clean narratives
    # ------------------------------------------------------------

    df = preprocess_data(
        df
    )


    # ------------------------------------------------------------
    # STEP 3
    # Balance classes
    # ------------------------------------------------------------

    df = balance_dataset(
        df
    )


    # ------------------------------------------------------------
    # STEP 4
    # Train / validation / test
    # ------------------------------------------------------------

    train_df, validation_df, test_df = (
        split_dataset(
            df
        )
    )


    # ------------------------------------------------------------
    # STEP 5
    # Convert class names to numbers
    # ------------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df

    ) = add_labels(

        train_df,
        validation_df,
        test_df
    )


    # ------------------------------------------------------------
    # Save processed datasets
    # ------------------------------------------------------------

    train_df.to_csv(
        DATA_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        DATA_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        DATA_DIR / "test.csv",
        index=False
    )


    # ------------------------------------------------------------
    # STEP 6
    # Traditional ML baseline
    # ------------------------------------------------------------

    baseline_results = train_baseline(

        train_df,

        test_df
    )


    # ------------------------------------------------------------
    # STEP 7
    # Convert to Hugging Face datasets
    # ------------------------------------------------------------

    dataset = create_huggingface_dataset(

        train_df,

        validation_df,

        test_df
    )


    print("\nHugging Face dataset:")

    print(
        dataset
    )


    # ------------------------------------------------------------
    # STEP 8
    # Tokenize
    # ------------------------------------------------------------

    tokenized_dataset = tokenize_dataset(
        dataset
    )


    # ------------------------------------------------------------
    # STEP 9
    # Fine-tune DistilBERT with LoRA
    # ------------------------------------------------------------

    trainer = train_transformer(
        tokenized_dataset
    )


    # ------------------------------------------------------------
    # STEP 10
    # Evaluate on completely unseen test data
    # ------------------------------------------------------------

    transformer_results = evaluate_transformer(

        trainer,

        tokenized_dataset
    )


    # ------------------------------------------------------------
    # STEP 11
    # Compare traditional ML vs Transformer
    # ------------------------------------------------------------

    compare_models(

        baseline_results,

        transformer_results
    )


    # ------------------------------------------------------------
    # STEP 12
    # Real inference examples
    # ------------------------------------------------------------

    print("\n================================================")
    print("NEW EVENT PREDICTIONS")
    print("================================================")


    examples = [

        """
        Several hours of intense rainfall caused streams to rise
        rapidly. Roads became impassable and water entered several
        low-lying areas.
        """,

        """
        A rotating storm moved across the county during the
        afternoon and produced a narrow path of structural damage
        and downed trees.
        """,

        """
        Strong winds associated with severe thunderstorms damaged
        roofs and knocked down numerous trees and power lines.
        """
    ]


    for text in examples:

        print("\nNarrative:")

        print(
            text.strip()
        )


        predictions = predict_event(

            text,

            trainer.model,

            tokenizer,

            top_k=3
        )


        print("\nPredictions:")


        for prediction in predictions:

            print(

                f"{prediction['event_type']:25s} "
                f"{prediction['confidence']:.2%}"

            )


        print(
            "-" * 60
        )


# ================================================================
# 29. RUN PROJECT
# ================================================================

if __name__ == "__main__":

    main()