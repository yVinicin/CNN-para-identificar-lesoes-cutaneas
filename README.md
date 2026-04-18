# 🔬 CNN para Identificação de Lesões Cutâneas

> Implementação de uma Rede Neural Convolucional (CNN) focada na classificação e identificação de lesões na pele, desenvolvida para a disciplina de Inteligência Artificial.

![Badge Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![Badge AI](https://img.shields.io/badge/Topic-Artificial%20Intelligence-orange?logo=openai&logoColor=white)
![Badge Deep Learning](https://img.shields.io/badge/Topic-Deep%20Learning-red?logo=keras&logoColor=white)
![Badge Academic](https://img.shields.io/badge/Type-Academic%20Project-blue)

## 🏫 Sobre o Projeto

Este projeto consiste no 2º Trabalho de Inteligência Artificial (Redes Neurais Artificiais - RNAs). 

O objetivo principal é treinar e utilizar um modelo de **Rede Neural Convolucional (CNN)** capaz de analisar imagens de lesões cutâneas e classificá-las de forma automatizada. Projetos desse tipo são fundamentais para auxiliar diagnósticos médicos preliminares utilizando Visão Computacional.

## 📂 Estrutura do Projeto

O repositório está organizado para separar o código fonte, a documentação e os resultados obtidos pelo modelo:

```bash
CNN-para-identificar-lesoes-cutaneas/
├── src/                          # Códigos fonte em Python (arquitetura da CNN e predição)
├── Resultados/                   # Gráficos de treinamento, matrizes de confusão e métricas
├── Artigo.pdf                    # Artigo científico detalhando a metodologia e conclusões
├── manual_usuario.pdf            # Manual passo a passo para utilização da ferramenta
├── instruções_treinamento.txt    # Guia para replicar o treinamento do modelo do zero
├── TrabalhoRNAs2025.pdf          # Enunciado e especificações originais do projeto
├── requirements.txt              # Bibliotecas necessárias para rodar o projeto
└── README.md                     # Esta documentação
```

## 🚀 Como Executar

Certifique-se de ter o **Python 3** instalado. É altamente recomendado o uso de um ambiente virtual (como `venv` ou `conda`) para gerenciar as dependências de Machine Learning.

### Passo a passo

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/yVinicin/CNN-para-identificar-lesoes-cutaneas.git
    cd CNN-para-identificar-lesoes-cutaneas
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Utilização:**
    * Para aprender a utilizar a interface ou realizar predições, consulte o **`manual_usuario.pdf`**.
    * Se deseja treinar o modelo novamente com o seu próprio hardware, siga as etapas descritas no arquivo **`instruções_treinamento.txt`**.

## 📊 Análise e Resultados

Toda a fundamentação teórica, a arquitetura das camadas de convolução (tamanho dos filtros, funções de ativação, *pooling*, etc.) e a discussão sobre acurácia e perda (*loss*) estão documentadas no **`Artigo.pdf`**. Os gráficos gerados durante a época de testes encontram-se na pasta `Resultados/`.
