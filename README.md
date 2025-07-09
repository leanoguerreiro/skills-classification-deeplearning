
# Brain Tumor Classification with Deep Learning

This repository contains a deep learning pipeline for classifying brain tumors from MRI images. The pipeline is designed for experimentation with various pre-trained models, data augmentation techniques (including Gaussian noise), and robustness testing on noisy validation sets using k-fold cross-validation.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- timm
- scikit-learn
- pandas
- matplotlib
- seaborn
- requests (for Telegram notifications)

You can install the required Python packages using pip:

```bash
pip install torch torchvision timm scikit-learn pandas matplotlib seaborn requests
````

### 📁 Data Preparation

The pipeline expects the dataset to be organized in a specific directory structure. Place your brain tumor MRI images in `../data/braintumor` relative to where you run the `main.py` script. The structure should follow the ImageFolder convention:

```
data/
└── braintumor/
    ├── glioma_tumor/
    │   ├── gg (1).jpg
    │   └── ...
    ├── meningioma_tumor/
    │   ├── m (1).jpg
    │   └── ...
    ├── no_tumor/
    │   ├── nt (1).jpg
    │   └── ...
    └── pituitary_tumor/
        ├── p (1).jpg
        └── ...
```

As shown above, the dataset contains four classes: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, and `pituitary_tumor`.

### 🔐 Environment Variables

For Telegram notifications, set the following environment variables:

* `TELEGRAM_BOT_TOKEN`: Your Telegram Bot API Token.
* `TELEGRAM_CHAT_ID`: The chat ID where you want to receive notifications.

---

## 📦 Project Structure

```
main.py           # Core training/validation and cross-validation logic
Experimento1.py   # Experiments without Gaussian noise
Experimento2.py   # Training with Gaussian noise as augmentation
Experimento3.py   # Validation with Gaussian noise for robustness testing
reports/          # Automatically generated confusion matrices and reports
```

---

## 🧠 Core Functionality (main.py)

The `main.py` script encapsulates the following key features:

* **Config Class**: Manages hyperparameters and settings: `DATA_PATH`, `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `N_SPLITS`, and device (CUDA/CPU).
* **Gaussian Noise Module**: Adds configurable Gaussian noise to image tensors.
* **Dynamic Data Transformations**: Controlled via `get_transforms` based on training/validation phase and flags.
* **Robustness Testing**: Use `APPLY_GAUSSIAN_NOISE_VAL_TEST` to apply noise only to validation.
* **TransformedSubset**: Enables applying transforms dynamically for k-fold subsets.
* **Training and Validation Loop**: Mixed-precision training, early stopping, model saving.
* **Early Stopping**: Stops training if validation accuracy doesn’t improve for `EARLY_STOPPING_PATIENCE` epochs.
* **Best Model Saving**: Saves the best model for each fold.
* **K-Fold Cross-Validation**: Uses `StratifiedKFold` to maintain class distribution.
* **Model Selection**: Supports various models via `timm.create_model`.
* **Reporting**: Generates and saves classification reports and confusion matrices.
* **Command Line Arguments**: Configure training via CLI (model, batch size, learning rate, noise, etc.).

---

## 🔬 Experiment Orchestration

The `ExperimentoX.py` scripts automate experiments using `main.py` with various configurations. Telegram notifications update you on progress.

Each experiment defines a dictionary with:

* `model_name`: Architecture (e.g., `'convnext_base'`, `'mixer_s16_224'`, etc.)
* `batch_size`
* `learning_rate`
* `patience`: Early stopping patience
* `save_best`: Save best weights
* `no_data_augmentation`: Disable standard augmentations
* `apply_gaussian_noise_train`: Use Gaussian noise in training
* `gaussian_noise_std_train`: Std dev for training noise
* `apply_gaussian_noise_val_test`: Use Gaussian noise in validation
* `test_noise_std_val`: Std dev for validation noise

### 🧪 Experimento1.py: Baseline (No Noise)

No Gaussian noise is applied.

```python
experimentos = [
    {
        'model_name': 'convnext_base',
        'batch_size': 32,
        'learning_rate': 0.0001,
        'patience': 10,
        'save_best': True,
        'no_data_augmentation': False,
        'apply_gaussian_noise_train': False,
        'gaussian_noise_std_train': 0.0,
        'apply_gaussian_noise_val_test': False,
        'test_noise_std_val': 0.0
    },
    # ... other models
]
```

### 🧪 Experimento2.py: Training with Gaussian Noise

Applies Gaussian noise (`std=0.05`) during training.

```python
experimentos = [
    {
        'model_name': 'convnext_base',
        'batch_size': 32,
        'learning_rate': 0.0001,
        'patience': 10,
        'save_best': True,
        'no_data_augmentation': False,
        'apply_gaussian_noise_train': True,
        'gaussian_noise_std_train': 0.05,
        'apply_gaussian_noise_val_test': False,
        'test_noise_std_val': 0.0
    },
    # ... other models
]
```

### 🧪 Experimento3.py: Validation Noise for Robustness Testing

Applies Gaussian noise (`std=0.07`) only to the validation set.

```python
experimentos = [
    {
        'model_name': 'convnext_base',
        'batch_size': 32,
        'learning_rate': 0.0001,
        'patience': 10,
        'save_best': True,
        'no_data_augmentation': False,
        'apply_gaussian_noise_train': False,
        'gaussian_noise_std_train': 0.0,
        'apply_gaussian_noise_val_test': True,
        'test_noise_std_val': 0.07
    },
    # ... other models
]
```

---

## 📲 Telegram Notifications

Each experiment script uses `send_telegram_message` to notify progress, success, or failure directly to your Telegram chat.

---

## 🏃 How to Run Experiments

Run any experiment script like so:

```bash
python Experimento1.py
# or
python Experimento2.py
# or
python Experimento3.py
```

Scripts automatically iterate through the experiment list, running `main.py` for each. A 3-minute delay is included between experiments.

---

## 📊 Results

After running experiments, detailed outputs are saved to `reports/`, including:

* Classification reports
* Confusion matrices (PNG)
* Text reports with summary results

---

## 🤝 Contributing

Feel free to fork this repository, open issues, and submit pull requests.

---

## 📄 License

[MIT](/License)
