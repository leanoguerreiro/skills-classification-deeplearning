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
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


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
        #transforms.ToTensor(),
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

def train_and_validate(model, train_loader, val_loader, config, fold_num):
    model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()

    epoch_accuracies = []

    best_val_acc = -1.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS} [Train]")
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

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        print(f"Epoch {epoch + 1}/{config.EPOCHS} [Validation]")
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
        epoch_accuracies.append(val_acc)

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
        print(f"Modelo carregado com a melhor acurácia de validação ({best_val_acc:.2f}%) para o Fold {fold_num}.")

    print('-' * 20, f"Fold {fold_num} training finished", '-' * 20)
    return best_val_acc


def generate_report(model, data_loader, device, model_name, class_names, fold_num, timestamp):
    model.eval()
    y_true = []
    y_pred = []
    
    base_reports_dir = 'reports'
    classification_report_path = os.path.join(base_reports_dir, 'classification')
    confusion_report_path = os.path.join(base_reports_dir, 'confusion_matrix')

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

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


def run_cross_validation(config, dataset, num_classes, class_names):
    kf = StratifiedKFold(n_splits=config.N_SPLITS)
    fold_accuracies = []

    print(f'\n- - - Starting Cross-Validation ({config.MODEL_NAME}) - - -')

    # --- GERAR TIMESTAMP AQUI ---
    # Isso criará um timestamp único para esta execução de cross-validation
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp da execução: {current_timestamp}")
    # --- FIM DO TIMESTAMP ---
    
    # `dataset.targets` funciona bem com ImageFolder
    targets = dataset.targets

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), targets)):
        fold_num = fold + 1
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

        best_fold_acc = train_and_validate(model, train_loader, val_loader, config, fold_num)
        fold_accuracies.append(best_fold_acc)
        print(f"Best Fold Accuracy [{fold_num}]: {best_fold_acc:.2f}%")

        print(f"\nGenerating report for Fold {fold_num} validation set...")
        generate_report(model, val_loader, config.DEVICE, config.MODEL_NAME, class_names, fold_num, current_timestamp)

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
    print(f"Using device: {config.DEVICE}")

    accuracies = run_cross_validation(config, dataset, num_classes, class_names)

    log_final_results(config, accuracies)


if __name__ == "__main__":
    main()
