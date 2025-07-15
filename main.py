from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from timm import create_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datetime import datetime
import warnings
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
from sklearn.calibration import calibration_curve # NEW: For Calibration Plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time

TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

warnings.filterwarnings("ignore")

def send_telegram_message(message):
    """Envia uma mensagem para o Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML' # Permite formatação em HTML (negrito, itálico, etc.)
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Lança uma exceção para códigos de status HTTP de erro
        # print(f"Notificação Telegram enviada: {message}") # Opcional: para depuração
    except requests.exceptions.RequestException as e:
        print(f"Erro ao enviar notificação Telegram: {e}")


# --- 1. CLASSE DE CONFIGURAÇÃO ---
class Config:
    """Armazena todas as configurações para o experimento de treinamento."""
    DATA_PATH = '../data/braintumor'
    IMG_SIZE = (224, 224)

    MODEL_NAME = 'resnet18'  # Nome do modelo padrão, será sobrescrito pelo argumento

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    N_SPLITS = 5
    NUM_WORKERS = 8

    # --- NOVAS CONFIGURAÇÕES PARA AUMENTO DE DADOS ---
    APPLY_DATA_AUGMENTATION = True  # Ativar/desativar aumento de dados no treino
    APPLY_GAUSSIAN_NOISE_TRAIN = False  # Aplicar ruído gaussiano no treino
    GAUSSIAN_NOISE_MEAN = 0.0
    GAUSSIAN_NOISE_STD = 0.07
    GAUSSIAN_NOISE_CLIP = True

    # --- Configurações de teste com ruído na validação (para robustez) ---
    APPLY_GAUSSIAN_NOISE_VAL_TEST = False  # Aplicar ruído gaussiano *apenas para testar a robustez na validação*
    # Isso NÃO é aumento de dados tradicional
    TEST_NOISE_MEAN = 0.0
    TEST_NOISE_STD = 0.07
    TEST_NOISE_CLIP = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- NOVAS CONFIGURAÇÕES PARA EARLY STOPPING ---
    EARLY_STOPPING_PATIENCE = 5  # Número de épocas para esperar sem melhora
    SAVE_BEST_MODEL = True  # Se deve salvar o melhor modelo por fold


# --- 2. CLASSES E FUNÇÕES AUXILIARES ---
class GaussianNoise:
    def __init__(self, mean=0., std=1., clip=True):
        self.std = std
        self.mean = mean
        self.clip = clip

    def __call__(self, img_tensor):
        noise = torch.randn_like(img_tensor) * self.std + self.mean
        noisy_img = img_tensor + noise
        if self.clip:
            noisy_img = torch.clamp(noisy_img, -1., 1.)  # Assume imagem normalizada entre -1 e 1
        return noisy_img

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, clip={self.clip})"


# --- NOVO: Função para obter transformações ---
def get_transforms(config, is_train):
    base_transforms = [
        transforms.Resize(config.IMG_SIZE),
        #transforms.ToTensor(), # ToTensor já é aplicado no load_data agora
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalização para [-1, 1]
    ]

    if is_train and config.APPLY_DATA_AUGMENTATION:
        # Adicionar transformações de aumento de dados para o treinamento
        augmentation_transforms = [
            transforms.RandomRotation(15),  # Rotação aleatória
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),  # Recorte e redimensionamento aleatório
            transforms.RandomHorizontalFlip(),  # Inversão horizontal
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Para imagens RGB
            # Adicionar ruído Gaussiano se configurado para treino
            *([] if not config.APPLY_GAUSSIAN_NOISE_TRAIN else [
                GaussianNoise(config.GAUSSIAN_NOISE_MEAN, config.GAUSSIAN_NOISE_STD, config.GAUSSIAN_NOISE_CLIP)
            ])
        ]
        return transforms.Compose(augmentation_transforms + base_transforms)
    else:
        # Apenas transformações de pré-processamento para validação
        return transforms.Compose(base_transforms)


def load_data(data_path):
    # As transformações serão aplicadas na hora de criar os DataLoaders para train_subset e val_subset
    # Aqui, apenas carregamos o ImageFolder sem transformações iniciais
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"O diretório do dataset não foi encontrado em: {data_path}")
    # Usamos uma transformação mínima apenas para carregar o PIL Image
    dataset = datasets.ImageFolder(data_path, transform=transforms.ToTensor())
    return dataset, len(dataset.classes), dataset.classes  # Return class_names as well


# --- NOVO: Wrapper para aplicar transformações ao Subset dinamicamente ---
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        # A imagem já vem como tensor por conta do `load_data` que usa ToTensor()
        # Se for um ImageFolder puro, idx retornaria (img_pil, label)
        # Como o load_data já usa ToTensor, subset[idx] retorna (img_tensor, label)
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# --- 3. FUNÇÕES PRINCIPAIS DE TREINAMENTO E VALIDAÇÃO (ajustadas) ---

def train_and_validate(model, train_loader, val_loader, config, fold_num, current_timestamp):
    model_filename = f"best_model_{config.MODEL_NAME}_fold_{fold_num}_{current_timestamp}.pth"
    os.makedirs('models', exist_ok=True) # Garante que o diretório 'models' existe
    model_save_path = os.path.join('models', model_filename)
    model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()

    # NEW: Lists to store metrics for learning curves
    train_losses_history = []
    val_losses_history = []
    val_accuracies_history = []

    best_val_acc = -1.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS} [Train]")
        send_telegram_message(f"Epoch {epoch + 1}/{config.EPOCHS} [Train]")
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print progress every 10 batches
                print(f"  Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses_history.append(avg_train_loss) # NEW: Store train loss

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        print(f"Epoch {epoch + 1}/{config.EPOCHS} [Validation]")
        send_telegram_message(f"Epoch {epoch + 1}/{config.EPOCHS} [Validation]")
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += torch.eq(predicted, labels).sum().item()
                total += labels.size(0)
                if (i + 1) % 50 == 0:  # Print progress every 50 batches
                    print(f"  Validation Batch {i + 1}/{len(val_loader)}")

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses_history.append(avg_val_loss) # NEW: Store val loss
        val_accuracies_history.append(val_acc) # NEW: Store val accuracy

        summary_line = (
            f"Epoch [{epoch + 1}/{config.EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        print(summary_line)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            if config.SAVE_BEST_MODEL:
                best_model_state = model.state_dict()
                print(f"  --> Nova melhor acurácia de validação. Salvando modelo para o Fold {fold_num}.")
        else:
            epochs_no_improve += 1
            print(f"  --> Acurácia de validação não melhorou por {epochs_no_improve} época(s).")
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n=== Early stopping acionado no Fold {fold_num} após {epoch + 1} épocas! ===")
                break

    if best_model_state is not None and config.SAVE_BEST_MODEL:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), model_save_path)
        print(f"Modelo carregado com a melhor acurácia de validação ({best_val_acc:.2f}%) para o Fold {fold_num}.")

    print('-' * 20, f"Fold {fold_num} training finished", '-' * 20)
    # NEW: Return histories for plotting learning curves
    return best_val_acc, train_losses_history, val_losses_history, val_accuracies_history


# NEW FUNCTION: For plotting learning curves
def plot_learning_curves(train_losses, val_losses, val_accuracies, config, fold_num, current_timestamp):
    reports_dir = 'reports'
    learning_curve_path = os.path.join(reports_dir, 'learning_curves')
    os.makedirs(learning_curve_path, exist_ok=True)

    plt.figure(figsize=(14, 6))

    # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    plt.title(f'Loss Curves (Fold {fold_num} - {config.MODEL_NAME})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Validation Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
    plt.title(f'Validation Accuracy (Fold {fold_num} - {config.MODEL_NAME})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filename = os.path.join(learning_curve_path, f'{current_timestamp}_{config.MODEL_NAME}_learning_curves_fold_{fold_num}.png')
    plt.savefig(filename)
    print(f"Learning curves saved to '{os.path.abspath(filename)}'")
    send_telegram_message(f"Learning curves saved to '{os.path.abspath(filename)}'")
    plt.close()


def generate_report(model, data_loader, device, model_name, class_names, fold_num, timestamp):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = [] # Store probabilities for ROC/PRC

    base_reports_dir = 'reports'
    classification_report_path = os.path.join(base_reports_dir, 'classification')
    confusion_report_path = os.path.join(base_reports_dir, 'confusion_matrix')
    roc_curve_path = os.path.join(base_reports_dir, 'roc_curves')
    pr_curve_path = os.path.join(base_reports_dir, 'precision_recall_curves')
    per_class_metrics_path = os.path.join(base_reports_dir, 'per_class_metrics') # NEW
    calibration_plots_path = os.path.join(base_reports_dir, 'calibration_plots') # NEW

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1) # Get probabilities
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_scores_np = np.array(y_scores)

    # --- Classification Report ---
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\n--- Classification Report (Fold {fold_num}) ---")
    print(report_str)

    # Garantir que o diretório 'reports/classification/' exista
    os.makedirs(classification_report_path, exist_ok=True)
    report_filename = os.path.join(classification_report_path, f'{timestamp}_{model_name}_classification_report_fold_{fold_num}.txt')
    with open(report_filename, 'w') as f:
        f.write(f"--- Classification Report (Fold {fold_num}) ---\n")
        f.write(report_str)
    print(f"Classification report saved to '{os.path.abspath(report_filename)}'")
    send_telegram_message(f"Classification report saved to '{os.path.abspath(report_filename)}'")

    print(f"\n--- Confusion Matrix (Fold {fold_num} {model_name}) ---")
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix (Fold {fold_num} {model_name})')
    # Garantir que o diretório 'reports/confusion_matrix/' exista
    os.makedirs(confusion_report_path, exist_ok=True)
    report_path = os.path.join(confusion_report_path, f'{timestamp}_{model_name}_confusion_matrix_fold_{fold_num}.png')
    plt.savefig(report_path)
    print(f"Confusion matrix saved as '{os.path.abspath(report_path)}'")
    plt.close()  # Fechar a figura para evitar sobrecarga de memória
    send_telegram_message(f"Confusion matrix saved as '{os.path.abspath(report_path)}'")

    # --- ROC Curve and AUC --- 
    os.makedirs(roc_curve_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_np == i, y_scores_np[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve of class {class_name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - Fold {fold_num} {model_name}')
    plt.legend(loc='lower right')
    roc_path = os.path.join(roc_curve_path, f'{timestamp}_{model_name}_roc_curve_fold_{fold_num}.png')
    plt.savefig(roc_path)
    print(f"ROC curve saved as '{os.path.abspath(roc_path)}'")
    plt.close()
    send_telegram_message(f"ROC curve saved as '{os.path.abspath(roc_path)}'")

    # --- Precision-Recall Curve ---
    os.makedirs(pr_curve_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_np == i, y_scores_np[:, i])
        plt.plot(recall, precision, label=f'PR curve of class {class_name}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold_num} {model_name}')
    plt.legend(loc='lower left')
    pr_path = os.path.join(pr_curve_path, f'{timestamp}_{model_name}_precision_recall_curve_fold_{fold_num}.png')
    plt.savefig(pr_path)
    print(f"Precision-Recall curve saved as '{os.path.abspath(pr_path)}'")
    plt.close()
    send_telegram_message(f"Precision-Recall curve saved as '{os.path.abspath(pr_path)}'")

    # NEW PLOT: Per-Class Metrics Bar Chart
    print(f"\n--- Generating Per-Class Metrics Bar Chart (Fold {fold_num} {model_name}) ---")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average=None, labels=np.arange(len(class_names)))

    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })

    metrics_df_melted = metrics_df.melt(id_vars='Class', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Class', y='Score', hue='Metric', data=metrics_df_melted, palette='viridis')
    plt.title(f'Per-Class Metrics - Fold {fold_num} {model_name}')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.ylim(0, 1.05) # Ensure full range visible
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(per_class_metrics_path, exist_ok=True)
    metrics_plot_filename = os.path.join(per_class_metrics_path, f'{timestamp}_{model_name}_per_class_metrics_fold_{fold_num}.png')
    plt.savefig(metrics_plot_filename)
    print(f"Per-class metrics bar chart saved to '{os.path.abspath(metrics_plot_filename)}'")
    send_telegram_message(f"Per-class metrics bar chart saved to '{os.path.abspath(metrics_plot_filename)}'")
    plt.close()

    # NEW PLOT: Calibration Plot (Reliability Diagram)
    print(f"\n--- Generating Calibration Plot (Fold {fold_num} {model_name}) ---")
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')

    # Plot one curve per class (one-vs-rest)
    for i, class_name in enumerate(class_names):
        prob_pos, actual_prob_pos = calibration_curve(y_true_np == i, y_scores_np[:, i], n_bins=10)
        plt.plot(prob_pos, actual_prob_pos, marker='o', label=f'Class {class_name}')

    plt.title(f'Calibration Plot (Reliability Diagram) - Fold {fold_num} {model_name}')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend(loc='upper left')
    plt.grid(True)

    os.makedirs(calibration_plots_path, exist_ok=True)
    calibration_plot_filename = os.path.join(calibration_plots_path, f'{timestamp}_{model_name}_calibration_plot_fold_{fold_num}.png')
    plt.savefig(calibration_plot_filename)
    print(f"Calibration plot saved to '{os.path.abspath(calibration_plot_filename)}'")
    send_telegram_message(f"Calibration plot saved to '{os.path.abspath(calibration_plot_filename)}'")
    plt.close()


def run_cross_validation(config, dataset, num_classes, class_names):
    kf = StratifiedKFold(n_splits=config.N_SPLITS)
    fold_accuracies = []

    print(f'\n- - - Starting Cross-Validation ({config.MODEL_NAME}) - - -')
    send_telegram_message(f"Starting Cross-Validation ({config.MODEL_NAME}) - - ")
    # --- GERAR TIMESTAMP AQUI ---
    # Isso criará um timestamp único para esta execução de cross-validation
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp da execução: {current_timestamp}")
    # --- FIM DO TIMESTAMP ---

    # `dataset.targets` funciona bem com ImageFolder
    targets = dataset.targets

    start_time_fold = time.time()


    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), targets)):
        fold_num = fold + 1
        send_telegram_message(f"\n{'=' * 5}Iniciando Fold {fold_num}/{config.N_SPLITS} {'=' * 5}")
        print(f"\n{'=' * 25} Fold {fold_num}/{config.N_SPLITS} {'=' * 25}")

        # Criar Subsets
        train_subset_raw = Subset(dataset, train_idx)
        val_subset_raw = Subset(dataset, val_idx)

        # Obter transformações específicas para treino e validação
        train_transforms = get_transforms(config, is_train=True)
        val_transforms = get_transforms(config, is_train=False)

        # Aplicar ruído gaussiano para testar robustez NA VALIDAÇÃO (se configurado)
        # Note: isso não é aumento de dados de treino, é para testar o modelo treinado
        # em condições ruidosas.
        if config.APPLY_GAUSSIAN_NOISE_VAL_TEST:
            print(f"Aplicando ruído gaussiano para teste de robustez no Fold {fold_num} (validação)...")
            val_noise_transform = transforms.Compose([
                val_transforms,  # Aplica as transformações de validação normais primeiro
                GaussianNoise(config.TEST_NOISE_MEAN, config.TEST_NOISE_STD, config.TEST_NOISE_CLIP)
            ])
            train_dataset = TransformedSubset(train_subset_raw, transform=train_transforms)
            val_dataset = TransformedSubset(val_subset_raw, transform=val_noise_transform)
        else:
            train_dataset = TransformedSubset(train_subset_raw, transform=train_transforms)
            val_dataset = TransformedSubset(val_subset_raw, transform=val_transforms)

        print(f"Training Images: {len(train_dataset)} | Validation Images: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, pin_memory=True)

        model = create_model(config.MODEL_NAME, pretrained=True, num_classes=num_classes)

        # NEW: Get history of losses and accuracies
        best_fold_acc, train_losses, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, config, fold_num, current_timestamp)
        fold_accuracies.append(best_fold_acc)
        print(f"Best Fold Accuracy [{fold_num}]: {best_fold_acc:.2f}%")

        # NEW: Plot learning curves
        print(f"\nGenerating learning curves for Fold {fold_num}...")
        plot_learning_curves(train_losses, val_losses, val_accuracies, config, fold_num, current_timestamp)

        print(f"\nGenerating report for Fold {fold_num} validation set...")
        generate_report(model, val_loader, config.DEVICE, config.MODEL_NAME, class_names, fold_num, current_timestamp)
        end_time_fold = time.time()
        duration_fold = end_time_fold - start_time_fold
        duration_fold_formatted = time.strftime("%H:%M:%S", time.gmtime(duration_fold))
        send_telegram_message(
            f"📊 Fold `{fold_num}` finalizado\n"
            f"Duração: `{duration_fold_formatted}`\n"
            f"Progresso Total: `{fold + 1}/{config.N_SPLITS}`\n"
        )
    return fold_accuracies

def log_final_results(config, accuracies):
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print("\n" + "=" * 25 + " Final Results " + "=" * 25)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Final Mean Accuracy ({config.N_SPLITS} folds): {mean_acc:.2f}%")
    print(f"Accuracy Standard Deviation: {std_acc:.4f}")
    print("=" * 69)

    # Garantir que o diretório 'reports' exista
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(reports_dir, f'report_{config.MODEL_NAME}_{timestamp}.txt')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"--- Experiment Report: {timestamp} ---\n")
        f.write(f"Model: {config.MODEL_NAME}\n\n")
        f.write("== Cross-Validation Results ==\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}%\n")
        f.write(f"Standard Deviation: {std_acc:.4f}\n\n")
        f.write("== Configurations Used ==\n")
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")

    print(f"Final report saved to: {os.path.abspath(filename)}")


# --- 4. FUNÇÃO DE EXECUÇÃO PRINCIPAL ---

def main():
    parser = argparse.ArgumentParser(description="Trains an image classification model with cross-validation.")
    parser.add_argument('--model_name', type=str, required=True, default='convnext_tiny',
                        help="Name of the TIMM model to be trained (e.g., 'resnet18', 'convnext_atto').")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size. Adjust for your GPU VRAM.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train.")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        help="Image size as two integers (height width).")

    # --- Argumentos para Data Augmentation no TREINO ---
    parser.add_argument('--no_data_augmentation', action='store_false', dest='apply_data_augmentation',
                        help="Disable data augmentation for training.")
    parser.add_argument('--apply_gaussian_noise_train', action='store_true',
                        help="Apply Gaussian noise during training (as data augmentation).")
    parser.add_argument('--gaussian_noise_std', type=float, default=0.02,
                        help="Standard deviation of Gaussian noise for training augmentation.")

    # --- Argumentos para Ruído Gaussiano na VALIDAÇÃO (teste de robustez) ---
    parser.add_argument('--apply_gaussian_noise_val_test', action='store_true',
                        help="Apply Gaussian noise to validation data to test model robustness.")
    parser.add_argument('--test_noise_std', type=float, default=0.02,
                        help="Standard deviation of Gaussian noise for validation robustness test.")

    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of workers for data loading.")
    parser.add_argument('--patience', type=int, default=5,
                        help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument('--save_best', action='store_true',
                        help="Save the best performing model (based on validation accuracy) for each fold.")

    args = parser.parse_args()

    config = Config()
    config.MODEL_NAME = args.model_name
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.NUM_WORKERS = args.num_workers
    config.IMG_SIZE = args.img_size

    # Atribui argumentos de aumento de dados e ruído
    config.APPLY_DATA_AUGMENTATION = args.apply_data_augmentation
    config.APPLY_GAUSSIAN_NOISE_TRAIN = args.apply_gaussian_noise_train
    config.GAUSSIAN_NOISE_STD = args.gaussian_noise_std
    config.APPLY_GAUSSIAN_NOISE_VAL_TEST = args.apply_gaussian_noise_val_test
    config.TEST_NOISE_STD = args.test_noise_std

    # Atribui argumentos de Early Stopping e salvar modelo
    config.EARLY_STOPPING_PATIENCE = args.patience
    config.SAVE_BEST_MODEL = args.save_best

    print('- - - Loading data - - -')
    # load_data agora retorna o ImageFolder base, sem transformações aplicadas ainda
    dataset, num_classes, class_names = load_data(config.DATA_PATH)
    print(f"Dataset loaded with {len(dataset)} images and {num_classes} classes.")
    print(f"Class names: {class_names}")

    class_counts = Counter(dataset.targets)
    print("\n--- Class Distribution ---")
    for class_idx, count in class_counts.items():
        print(f" Class'{class_names[class_idx]}: {count} images")
    print("--------------------------")

    print(f"Using device: {config.DEVICE}")

    accuracies = run_cross_validation(config, dataset, num_classes, class_names)

    log_final_results(config, accuracies)


if __name__ == "__main__":
    main()