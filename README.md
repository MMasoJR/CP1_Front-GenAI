# 🩻 VAE PneumoniaMNIST - Triagem Inteligente

Este projeto faz parte do ecossistema acadêmico e consiste em um sistema de triagem para auxílio diagnóstico de pneumonia utilizando **Variational Autoencoders (VAE)**. O sistema analisa radiografias de tórax e utiliza o erro de reconstrução para identificar anomalias, é válido informar que o projeto não está em nivel de produção e deve ser usado única e exclusivamente para fins de estudo.

## 🚀 Funcionalidades

* **Análise por Reconstrução:** Utiliza um modelo VAE treinado no dataset PneumoniaMNIST para detectar divergências em raios-x.
* **Pipeline Visual:** Interface interativa com Streamlit que simula o fluxo de processamento médico.
* **Human-in-the-Loop:** Aba dedicada para que especialistas validem ou corrijam as predições da IA, alimentando um log de auditoria.
* **Monitoramento de Degradação:** Gráficos em tempo real que alertam se a acurácia percebida cair abaixo dos limites de segurança.

## 🛠️ Tecnologias Utilizadas

* **Python 3.12**
* **TensorFlow / Keras 3**
* **Streamlit** (Interface Web)
* **Plotly** (Monitoramento e Histórico)
* **Pandas** (Gestão de Dados)

## 📁 Estrutura do Repositório

* `app.py`: Arquivo principal contendo a lógica do Streamlit e a arquitetura do modelo.
* `.weights.h5`: Pasta contendo os pesos treinados
*  `config.json`: O arquivo de configuração .
* `requirements.txt`: Lista de dependências para deploy em nuvem.

## ⚙️ Como Executar Localmente

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/MMasoJR/CP1_GenAi-Front.git](https://github.com/MMasoJR/CP1_GenAi-Front.git)
    ```
2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3.  Execute o aplicativo:
    ```bash
    streamlit run app.py
    ```

## 🧠 Metodologia: Erro de Reconstrução

Diferente de classificadores tradicionais, o VAE tenta reconstruir a imagem de entrada. 
* **Pulmões Normais:** São reconstruídos com baixo erro (MSE baixo).
* **Anomalias (Pneumonia):** Geram divergências na reconstrução, resultando em um erro elevado (MSE alto), que é classificado com base em limiares ajustáveis pelo usuário.

## 👨‍⚕️ Disclaimer (Uso Médico)

Este sistema é uma ferramenta de auxílio e **não substitui o diagnóstico médico profissional**. Os resultados devem ser sempre validados por um radiologista, conforme implementado na aba de validação do sistema.
