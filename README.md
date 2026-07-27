# ifood-case-ds

Este repositório contém a solução para o case de Data Science do iFood.

## Estrutura dos arquivos

- `README.md`: Documentação do projeto e instruções de uso.
- `data/`: Pasta com os dados utilizados no case.
- `notebooks/`: Notebooks com análises exploratórias e modelagem.
- `src/`: Scripts Python para processamento de dados e treinamento de modelos.
- `requirements.txt`: Lista de dependências do projeto.

## Como executar o fluxo

1. Instale as dependências:
   bash
   pip install -r requirements.txt
   

2. Execute o notebook principal:
   bash
   jupyter notebook notebooks/main.ipynb
   

3. Para rodar os scripts diretamente:
   bash
   python src/preprocess.py
   python src/train_model.py