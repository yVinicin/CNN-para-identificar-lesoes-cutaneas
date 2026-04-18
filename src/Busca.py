import json
import os

# Suprime a mensagem informativa do oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# =============================
# 1. Carregar conjunto de dados
# =============================
caminho_arquivo = "hmnist_28_28_RGB.csv"
conjunto_dados = pd.read_csv(caminho_arquivo)

rotulo = conjunto_dados["label"]
dados = conjunto_dados.drop(columns=["label"])

# =============================
# 2. Oversampling
# =============================
sobreamostragem = RandomOverSampler()
dados, rotulo = sobreamostragem.fit_resample(dados, rotulo)
dados = np.array(dados).reshape(-1, 28, 28, 3)
rotulo = np.array(rotulo)

# =============================
# 3. Split: treino/val/teste
# =============================
X_treino, X_teste, y_treino, y_teste = train_test_split(dados, rotulo, test_size=0.20, random_state=49)
X_val, X_teste, y_val, y_teste = train_test_split(X_teste, y_teste, test_size=0.5, random_state=49)

# =============================
# 4. One-hot encoding
# =============================
y_treino = to_categorical(y_treino, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)

# =============================
# 5. Normalização
# =============================
X_treino_norm = X_treino / 255.0
X_val = X_val / 255.0

# =============================
# 6. Função modelo CNN
# =============================
def criar_modelo(num_filtros_conv1=16, num_filtros_conv2=32, num_filtros_conv3=64, num_filtros_conv4=128,
                 num_neuronios_dense1=64, num_neuronios_dense2=32, l2_reg=0.0001,
                 dropout_rate=0.1, optimizer='adam' ):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 3)),

        # Bloco 1
        layers.Conv2D(num_filtros_conv1, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Bloco 2
        layers.Conv2D(num_filtros_conv2, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Bloco 3
        layers.Conv2D(num_filtros_conv3, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Bloco 4
        layers.Conv2D(num_filtros_conv4, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Densas
        layers.Flatten(),
        layers.Dense(num_neuronios_dense1, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_neuronios_dense2, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(7, activation='softmax', name='classificador')
    ])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =============================
# 7. Busca de hiperparâmetros
# =============================
param_grid = {
    'num_filtros_conv1': [16, 32, 64],
    'num_filtros_conv2': [32, 64, 128],
    'num_filtros_conv3': [64, 128, 256],
    'num_filtros_conv4': [128, 256, 512],
    'num_neuronios_dense1': [64, 128, 256],
    'num_neuronios_dense2': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'l2_reg': [0.0001, 0.001, 0.01],
    'optimizer': ['adam', 'sgd']
}
max_combinacoes = 13122
combinacoes_testadas = 0
melhor_acuracia = 0.0
melhores_params = None

# Callbacks para tuning
reducao_taxa_aprendizado = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6, verbose=0)
parada_antecipada = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

for params in tqdm(list(ParameterGrid(param_grid))[:max_combinacoes], desc="Buscando hiperparâmetros"):
    modelo_teste = criar_modelo(**params)
    historico_teste = modelo_teste.fit(
        X_treino_norm, y_treino,
        epochs=5, batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[reducao_taxa_aprendizado, parada_antecipada],
        verbose=0 # Silencia o output do Keras para não poluir a barra de progresso
    )

    acuracia_val = max(historico_teste.history['val_accuracy'])
    if acuracia_val > melhor_acuracia:
        melhor_acuracia = acuracia_val
        melhores_params = params

    combinacoes_testadas += 1
    print(f"Combinação {combinacoes_testadas}: {params}, Acurácia Val = {acuracia_val:.4f}")

print(f"\nMelhores parâmetros: {melhores_params}")
print(f"Melhor acurácia de validação: {melhor_acuracia:.4f}")

# Salvar os melhores parâmetros em TXT
with open('melhores_hiperparametros.txt', 'w') as f:
    json.dump(melhores_params, f, indent=4)

print("Melhores parâmetros salvos em 'melhores_hiperparametros.txt'")
