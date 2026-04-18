import json
import os

# Suprime a mensagem informativa do oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Equipe: Matheus Bueno, Thalita, Vinícius Mattos
# Código para treinamento da CNN sem busca de hiperparâmetros (removido o GridSearchCV)

# --- 1. Carregar o conjunto de dados ---
caminho_arquivo = "hmnist_28_28_RGB.csv"  # Ajuste o caminho se necessário
conjunto_dados = pd.read_csv(caminho_arquivo)

rotulo = conjunto_dados["label"]
dados = conjunto_dados.drop(columns=["label"])

# --- 2. Balancear as classes com oversampling ---
sobreamostragem = RandomOverSampler()
dados, rotulo = sobreamostragem.fit_resample(dados, rotulo)
dados = np.array(dados).reshape(-1, 28, 28, 3)
print('Formato dos dados após oversampling:', dados.shape)

# --- 3. Mapeamento das classes ---
# Ordem: 0:akiec, 1:bcc, 2:bkl, 3:df, 4:nv, 5:vasc, 6:mel
nomes_curtos = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
nomes_classes = {
    0: 'Ceratoses Actínicas e Carcinoma Intraepitelial',
    1: 'Carcinoma Basocelular',
    2: 'Lesões Benignas Semelhantes a Ceratose',
    3: 'Dermatofibroma',
    4: 'Nevos Melanocíticos',
    5: 'Lesões Vasculares (Granulomas e Hemorragia)',
    6: 'Melanoma'
}
rotulo = np.array(rotulo)

# --- 4. Dividir o conjunto de dados (Treino, Validação, Teste) ---
X_treino, X_teste, y_treino, y_teste = train_test_split(dados, rotulo, test_size=0.20, random_state=49)
X_val, X_teste, y_val, y_teste = train_test_split(X_teste, y_teste, test_size=0.5, random_state=49)

print("Formato X_treino:", X_treino.shape)
print("Formato y_treino:", y_treino.shape)
print("Formato X_val:", X_val.shape)
print("Formato y_val:", y_val.shape)
print("Formato X_teste:", X_teste.shape)
print("Formato y_teste:", y_teste.shape)

# --- 5. Codificar os rótulos em one-hot ---
y_treino = to_categorical(y_treino, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)
y_teste = to_categorical(y_teste, num_classes=7)

print("\nFormato y_treino (one-hot):", y_treino.shape)
print("Formato y_val (one-hot):", y_val.shape)
print("Formato y_teste (one-hot):", y_teste.shape)

# --- 6. Calcular pesos de classe para balanceamento ---
y_treino_classes = np.argmax(y_treino, axis=1)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_treino_classes),
    y=y_treino_classes
)
class_weight_dict = dict(zip(np.unique(y_treino_classes), class_weights))
print("\nPesos de classe:", class_weight_dict)

# --- 7. Normalizar dados e Augmentation ---
# Normalizar dados de validação e teste manualmente
X_val = X_val / 255.0
X_teste = X_teste / 255.0

# DataGenerator para augmentação no treino
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normaliza para [0,1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- 8. Função para criar o modelo ---
def criar_modelo(filters_conv1=16, filters_conv2=32, filters_conv3=64, filters_conv4=128,
                 dropout1=0.1, dropout2=0.2, dropout3=0.2, dropout4=0.1, dropout_dense=0.3, optimizer='adam'):
    model = models.Sequential()

    # Camada de entrada explícita para seguir - Keras
    model.add(layers.Input(shape=(28, 28, 3)))

    # Bloco 1
    model.add(layers.Conv2D(filters_conv1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout1))

    # Bloco 2
    model.add(layers.Conv2D(filters_conv2, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout2))

    # Bloco 3
    model.add(layers.Conv2D(filters_conv3, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout3))

    # Bloco 4
    model.add(layers.Conv2D(filters_conv4, (3, 3), activation='relu', padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout4))

    # Camadas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(dropout_dense))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(7, activation='softmax', name='classificador'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Criar generator para treino
batch_size = 16
train_generator = train_datagen.flow(X_treino, y_treino, batch_size=batch_size)

# --- 9. Cria o modelo final com parâmetros padrão ou definidos ---
modelo = criar_modelo()
modelo.summary()

# --- 10. Callbacks ---
reducao_taxa_aprendizado = ReduceLROnPlateau(
    monitor='val_loss',
    patience=10,
    verbose=1,
    factor=0.5,
    min_lr=1e-6
)

parada_antecipada = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# --- 11. Treinar o modelo ---
steps_per_epoch = len(X_treino) // batch_size
historico = modelo.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    validation_data=(X_val, y_val),
    callbacks=[reducao_taxa_aprendizado, parada_antecipada],
    verbose=1,
    class_weight=class_weight_dict # Adicionado pesos de classe
)

# Salvar o modelo
modelo.save('meu_modelo_melhorado.h5')

# --- 12. Visualizar Curvas de Aprendizado ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historico.history["accuracy"], color='blue', label="Acurácia no Treinamento")
plt.plot(historico.history["val_accuracy"], color='red', label="Acurácia na Validação")
plt.legend()
plt.title("Evolução da Acurácia")
plt.xlabel("Época")
plt.ylabel("Acurácia")

plt.subplot(1, 2, 2)
plt.plot(historico.history["loss"], color='blue', label="Perda no Treinamento")
plt.plot(historico.history["val_loss"], color='red', label="Perda na Validação")
plt.legend()
plt.title("Evolução da Perda")
plt.xlabel("Época")
plt.ylabel("Perda")

plt.tight_layout()
plt.savefig('evolucao_treinamento.png', dpi=300)
plt.show()

# --- 13. Avaliar o modelo ---
test_loss, test_accuracy = modelo.evaluate(X_teste, y_teste, verbose=0)
print(f"\nAcurácia no teste: {test_accuracy:.4f}")

# Predições
y_pred = modelo.predict(X_teste)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_teste, axis=1)

# Relatório de Classificação
print("\nRelatório de Classificação:")
relatorio = classification_report(y_true, y_pred_classes, target_names=nomes_curtos, output_dict=True)
print(classification_report(y_true, y_pred_classes, target_names=nomes_curtos))
print(f"F1-Score Médio (macro): {relatorio['macro avg']['f1-score']:.4f}")

print("\nMapeamento das classes:")
for i, nome in nomes_classes.items():
    print(f"{i}: {nomes_curtos[i]} - {nome}")

# Matriz de Confusão
matriz_confusao = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
            xticklabels=nomes_curtos, yticklabels=nomes_curtos)
plt.title('Matriz de Confusão')
plt.xlabel('Rótulos Preditos')
plt.ylabel('Rótulos Reais')
plt.savefig('matriz_confusao_melhorada.png', dpi=300)
plt.show()

# ROC-AUC
roc_auc = roc_auc_score(y_teste, y_pred, multi_class='ovr')
print(f"ROC-AUC (OvR): {roc_auc:.4f}")

# --- 14. Salvar informações em JSON ---
# Note: Removed best_hyperparams since GridSearchCV was removed
info_modelo = {
    "pesos_classe": {str(k): v for k, v in class_weight_dict.items()},
    "acuracia_teste": float(test_accuracy),
    "acuracia_validacao_final": float(historico.history['val_accuracy'][-1]),
    "f1_macro": float(relatorio['macro avg']['f1-score']),
    "roc_auc": float(roc_auc),
    "matriz_confusao": matriz_confusao.tolist(),
    "relatorio_classificacao": relatorio,
    "mapeamento_classes": {
        str(i): {'curto': nomes_curtos[i], 'completo': nomes_classes[i]} for i in range(7)
    }
}
with open('informacoes_modelo_melhorado.json', 'w') as f:
    json.dump(info_modelo, f, indent=4)

# --- 15. Inferência em exemplos de validação ---
exemplos = list(range(10))  # Primeiros 10 da validação

# Predições para o conjunto de validação
y_pred_val = modelo.predict(X_val)
y_pred_classes_val = np.argmax(y_pred_val, axis=1)
y_true_val = np.argmax(y_val, axis=1)

plt.figure(figsize=(15, 6))
plt.suptitle("Exemplos de Inferência (Validação)")
for i, idx in enumerate(exemplos):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_val[idx])  # X_val já está normalizado, imshow lida bem com float [0,1]
    real = nomes_curtos[y_true_val[idx]]
    predito = nomes_curtos[y_pred_classes_val[idx]]
    plt.title(f"Real: {real}\nPred: {predito}")
    plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('inferencia_exemplos.png', dpi=300)
plt.show()

for idx in exemplos:
    real = y_true_val[idx]
    pred = y_pred_classes_val[idx]
    print(f"Índice {idx}: Real={nomes_curtos[real]}, Predito={nomes_curtos[pred]}")
