import streamlit as st
import numpy as np
from PIL import Image as PILImage
import pandas as pd
import json

# Configuração da página
st.set_page_config(
    page_title="Classificador de Câncer de Pele 🩺",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Interface simplificada

# Carregar o modelo e informações
@st.cache_resource
def load_trained_model():
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelEncoder
    model = load_model('meu_modelo_melhorado.h5')
    with open('informacoes_modelo_melhorado.json', 'r') as f:
        model_info = json.load(f)
    le = LabelEncoder()
    # Extrai apenas os nomes completos das classes do dicionário aninhado
    nomes_das_classes = [v['completo'] for k, v in model_info['mapeamento_classes'].items()]
    le.classes_ = np.array(nomes_das_classes)
    return model, model_info, le

def preprocess_image(image, target_size=(28, 28)):
    """Pré-processa a imagem para o modelo."""
    img = image.convert('RGB').resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalização para [0, 1]
    return np.expand_dims(img_array, axis=0)

def main():
    # Sidebar
    with st.sidebar:
        st.title("Sobre o Sistema")
        st.write("""
        Este sistema utiliza Inteligência Artificial para classificar tipos de câncer de pele através de imagens de lesões cutâneas.

        **Como usar:**
        1. Faça upload de uma imagem de lesão cutânea
        2. Clique em "Classificar" para obter o resultado
        3. Analise as probabilidades por classe

        **Aviso:** Este sistema é apenas uma ferramenta auxiliar e não substitui o diagnóstico médico profissional.
        """)

        st.write("---")
        st.write("**Desenvolvido com:**")
        st.write("- TensorFlow/Keras")
        st.write("- CNN Personalizada")
        st.write("- Streamlit")

    # Conteúdo principal
    st.title("Sistema de Classificação de Câncer de Pele")
    st.write("Faça upload de uma imagem de lesão cutânea para classificação usando Inteligência Artificial.")

    # Carregar modelo
    try:
        model, model_info, le = load_trained_model()
    except FileNotFoundError:
        st.error("Arquivo do modelo ou informações não encontrado. Execute o treinamento primeiro.")
        return

    # Upload da imagem
    uploaded_file = st.file_uploader(
        "Selecione uma imagem de lesão cutânea",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = PILImage.open(uploaded_file)
        st.image(image, caption='Imagem enviada', width=300)

        # Botão de classificação
        if st.button('Classificar Lesão'):
            with st.spinner('Analisando imagem...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]
                class_name = le.inverse_transform([predicted_class])[0]
                confidence = np.max(prediction) * 100

            # Resultados
            st.header("Resultados da Classificação")

            st.write(f"**Classe Predita:** {class_name}")
            st.write(f"**Confiança:** {confidence:.1f}%")

            # Probabilidades detalhadas
            st.subheader("Probabilidades por Classe")
            classes = [v['curto'] for v in model_info['mapeamento_classes'].values()]

            probs_data = []
            for i, prob in enumerate(prediction[0]):
                probs_data.append({
                    'Classe': classes[i],
                    'Probabilidade': f"{prob*100:.1f}%"
                })

            # Ordenar por probabilidade decrescente
            probs_data.sort(key=lambda x: float(x['Probabilidade'].rstrip('%')), reverse=True)

            # Mostrar como tabela
            st.table(probs_data)

    # Footer
    st.write("---")
    st.write("**Aviso Importante:** Este sistema é uma ferramenta auxiliar e não substitui o diagnóstico médico profissional. Consulte sempre um dermatologista qualificado.")
    st.write("Desenvolvido com Streamlit e TensorFlow.")

if __name__ == "__main__":
    main()
