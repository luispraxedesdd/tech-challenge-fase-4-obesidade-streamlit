# Tech Challenge FIAP - Fase 4

Aplicação em Streamlit para auxiliar a equipe médica na previsão de nível de obesidade.

## Entrega

O projeto contém:

- `app.py`: aplicação Streamlit com sistema preditivo + dashboard analítico.
- `random.joblib`: modelo treinado usado na predição.
- `df_clean.csv`: base tratada usada no dashboard.
- `Obesity.csv`: base original/reserva.
- `pipeline_treinamento.py`: script demonstrando pipeline de Machine Learning, feature engineering, treinamento e avaliação.
- `requirements.txt`: dependências usadas no deploy.
- `entrega_links.txt`: arquivo para preencher com links da aplicação, painel e GitHub.

## Como rodar localmente

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Estrutura do app

O app possui 5 abas:

1. Sistema Preditivo
2. Dashboard Analítico
3. Insights Médicos
4. Dados
5. Entrega

## Observação

Este projeto é acadêmico. A previsão do modelo não substitui avaliação médica profissional.
