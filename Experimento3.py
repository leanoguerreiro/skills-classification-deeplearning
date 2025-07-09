import subprocess
import time
import sys
import os
import requests

TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

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
# Caminho para o seu script principal (main.py)
main_script_path = 'main.py'  # Assumindo que main.py está na mesma pasta

# Lista de dicionários, onde cada dicionário representa um experimento
# e contém o nome do modelo e o batch_size a ser usado.
# Adicione os novos parâmetros 'patience' e 'save_best' conforme sua necessidade.
experimentos = [
    # Experimento 3: Ruído na VALIDAÇÃO (para testar robustez)
    {'model_name': 'convnext_base', 'batch_size': 32, 'learning_rate': 0.0001, 'patience': 10,
     'save_best': True, 'no_data_augmentation': False,
     'apply_gaussian_noise_train': False, 'gaussian_noise_std_train': 0.0,
     'apply_gaussian_noise_val_test': True, 'test_noise_std_val': 0.07}, # Aplicar ruído na validação com std 0.07

    {'model_name': 'mixer_s16_224', 'batch_size': 32, 'learning_rate': 0.0001, 'patience': 10,
     'save_best': True, 'no_data_augmentation': False,
     'apply_gaussian_noise_train': False, 'gaussian_noise_std_train': 0.0,
     'apply_gaussian_noise_val_test': True, 'test_noise_std_val': 0.07}, # Aplicar ruído na validação com std 0.07
    {'model_name': 'efficientnet_b0', 'batch_size': 32, 'learning_rate': 0.0001, 'patience': 10,
     'save_best': True, 'no_data_augmentation': False,
     'apply_gaussian_noise_train': False, 'gaussian_noise_std_train': 0.0,
     'apply_gaussian_noise_val_test': True, 'test_noise_std_val': 0.07}, # Aplicar ruído na validação com std 0.07
    {'model_name': 'mobilevit_s', 'batch_size': 32, 'learning_rate': 0.0001, 'patience': 10,
     'save_best': True, 'no_data_augmentation': False,
     'apply_gaussian_noise_train': False, 'gaussian_noise_std_train': 0.0,
     'apply_gaussian_noise_val_test': True, 'test_noise_std_val': 0.07}, # Aplicar ruído na validação com std 0.07
    {'model_name': 'densenet121', 'batch_size': 32, 'learning_rate': 0.0001, 'patience': 10,
     'save_best': True, 'no_data_augmentation': False,
     'apply_gaussian_noise_train': False, 'gaussian_noise_std_train': 0.0,
     'apply_gaussian_noise_val_test': True, 'test_noise_std_val': 0.07} # Aplicar ruído na validação com std 0.07
]

start_time_global = time.time()
send_telegram_message("🤖 Iniciando todos os experimentos...")

print("========================================")
print("INICIANDO TODOS OS EXPERIMENTOS.")
print("========================================")

# Itera sobre a lista de experimentos e executa cada um
all_experiments_successful = True # Nova flag para rastrear o sucesso geral
for i, experimento in enumerate(experimentos):
    model_name = experimento['model_name']
    batch_size = experimento['batch_size']
    learning_rate = experimento['learning_rate']
    patience = experimento.get('patience', 5)
    save_best = experimento.get('save_best', True)
    disable_data_augmentation_flag = experimento.get('no_data_augmentation', False)

    apply_gaussian_noise_train = experimento.get('apply_gaussian_noise_train', False)
    gaussian_noise_std_train = experimento.get('gaussian_noise_std_train', 0.0)
    apply_gaussian_noise_val_test = experimento.get('apply_gaussian_noise_val_test', False)
    test_noise_std_val = experimento.get('test_noise_std_val', 0.0)

    command = [
        sys.executable,
        main_script_path,
        '--model_name', model_name,
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate),
        '--patience', str(patience),
    ]

    if save_best:
        command.append('--save_best')

    if disable_data_augmentation_flag:
        command.append('--no_data_augmentation')

    if apply_gaussian_noise_train:
        command.append('--apply_gaussian_noise_train')
        if gaussian_noise_std_train > 0:
            command.extend(['--gaussian_noise_std', str(gaussian_noise_std_train)])

    if apply_gaussian_noise_val_test:
        command.append('--apply_gaussian_noise_val_test')
        if test_noise_std_val > 0:
            command.extend(['--test_noise_std', str(test_noise_std_val)])

    print(f"\n========================================")
    print(f"EXECUTANDO EXPERIMENTO {i + 1}/{len(experimentos)}:")
    print(f"  Modelo: {model_name}")
    print(f"  Batch Size: {batch_size}")
    print(f"  LR: {learning_rate}")
    print(f"  Paciência ES: {patience}")
    print(f"  Salvar Melhor: {save_best}")
    print(f"  Aumento de Dados Desabilitado: {disable_data_augmentation_flag}")
    print(f"  Ruído Treino: {'Sim' if apply_gaussian_noise_train else 'Não'} (std={gaussian_noise_std_train})")
    print(f"  Ruído Validação (Teste Robustez): {'Sim' if apply_gaussian_noise_val_test else 'Não'} (std={test_noise_std_val})")
    print(f"Comando: {' '.join(command)}")
    print(f"========================================")

    send_telegram_message(
        f"🚀 **Iniciando Experimento {i + 1}/{len(experimentos)}:**\n"
        f"Modelo: `{model_name}`\n"
        f"Batch Size: `{batch_size}`\n"
        f"LR: `{learning_rate}`\n"
        f"Aumento Dados: `{'Não' if disable_data_augmentation_flag else 'Sim'}`\n"
        f"Ruído Treino: `{'Sim' if apply_gaussian_noise_train else 'Não'}` (std={gaussian_noise_std_train})\n"
        f"Ruído Val: `{'Sim' if apply_gaussian_noise_val_test else 'Não'}` (std={test_noise_std_val})"
    )

    start_time_experiment = time.time()
    current_experiment_successful = False # Flag para o sucesso do experimento atual

    try:
        subprocess.run(command, check=True)
        current_experiment_successful = True
        message_status = "✅ **CONCLUÍDO COM SUCESSO!**"
        print(f"----------------------------------------")
        print(f"EXPERIMENTO COM {model_name} CONCLUÍDO COM SUCESSO.")
        print(f"----------------------------------------\n")

    except subprocess.CalledProcessError as e:
        message_status = f"❌ **FALHA NO EXPERIMENTO!** (Modelo: `{model_name}`, Erro: Código {e.returncode})"
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERRO ao executar o experimento com {model_name}: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        all_experiments_successful = False # Marca que pelo menos um experimento falhou
    except FileNotFoundError:
        message_status = f"⚠️ **ERRO FATAL: main.py não encontrado!** (Modelo: `{model_name}`)"
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERRO: O arquivo {main_script_path} não foi encontrado.")
        print(f"Verifique se 'main.py' está no caminho correto.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        all_experiments_successful = False # Marca que pelo menos um experimento falhou

    # --- Este bloco SEMPRE será executado, independentemente de sucesso ou falha ---
    end_time_experiment = time.time()
    duration_experiment = end_time_experiment - start_time_experiment
    duration_formatted = time.strftime("%Hh %Mm %Ss", time.gmtime(duration_experiment))

    send_telegram_message(
        f"📊 Experimento de `{model_name}` {message_status}\n"
        f"Duração: `{duration_formatted}`\n"
        f"Progresso Total: `{i + 1}/{len(experimentos)}`"
    )
    # --- Fim do bloco que SEMPRE será executado ---

    # Se o experimento atual falhou e você quer PARAR TUDO, use este if
    if not current_experiment_successful and 'break' in locals(): # Verifica se a exceção fez um 'break'
        print("Interrompendo a execução de experimentos restantes devido a uma falha.")
        break # Sai do loop for

    # Pausa APENAS se não for o ÚLTIMO experimento da lista
    if (i < len(experimentos) - 1) and current_experiment_successful:
        print("Pausa de 3 minutos para resfriamento do sistema e liberação de VRAM...")
        send_telegram_message(f"😴 Pausa de 3 minutos para resfriamento...")
        time.sleep(180)

# --- Mensagem de conclusão global ---
end_time_global = time.time()
total_duration = end_time_global - start_time_global
total_duration_formatted = time.strftime("%Hh %Mm %Ss", time.gmtime(total_duration))

if all_experiments_successful:
    final_message = f"🏁 **TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO!**\n" \
                    f"Tempo Total: `{total_duration_formatted}`"
else:
    final_message = f"🚨 **EXECUÇÃO DE EXPERIMENTOS FINALIZADA COM FALHAS!**\n" \
                    f"Verifique os logs acima para detalhes. Tempo Total: `{total_duration_formatted}`"

send_telegram_message(final_message)
print("========================================")
print("TODOS OS EXPERIMENTOS FORAM EXECUTADOS.")
print("========================================")