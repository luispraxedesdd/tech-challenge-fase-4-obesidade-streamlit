# -*- coding: utf-8 -*-
"""
Pipeline de treinamento - Tech Challenge FIAP Fase 4

Este script demonstra:
1) Leitura da base Obesity.csv
2) Feature engineering simples
3) Encoding das variáveis categóricas
4) Treinamento do modelo
5) Avaliação de acurácia
6) Salvamento do modelo em random.joblib

Execute:
python pipeline_treinamento.py
"""

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_FILE = "Obesity.csv"
MODEL_FILE = "random.joblib"
TARGET_COL = "Obesity"


def main():
    df = pd.read_csv(DATA_FILE)

    # Feature engineering: criação do IMC somente para análise/estudo.
    # Para manter compatibilidade com o app original, o modelo abaixo usa as colunas originais.
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    X = df.drop(columns=[TARGET_COL, "BMI"])
    y = df[TARGET_COL]

    encoders = {}

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Acurácia do modelo: {acc:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModelo salvo em: {MODEL_FILE}")


if __name__ == "__main__":
    main()
