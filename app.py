# -*- coding: utf-8 -*-
"""
Tech Challenge FIAP - Fase 4
Aplicação Streamlit: Sistema preditivo + Dashboard analítico de obesidade

Objetivo:
1) Manter o modelo preditivo existente.
2) Adicionar uma aba com dashboard e principais insights para a equipe médica.
3) Deixar a aplicação pronta para deploy no Streamlit Community Cloud.

Observação importante:
Este app é educacional e faz parte de um projeto acadêmico. A previsão do modelo
não substitui avaliação médica profissional.
"""

from pathlib import Path
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================

APP_URL = "https://tech-challenge---fase-4-lvabyfyvdbvsrbytztabt9.streamlit.app/"
GITHUB_URL = "COLE_AQUI_O_LINK_DO_REPOSITORIO_GITHUB"

DATA_FILES = ["df_clean.csv", "Obesity.csv", "df_clean (1).csv"]
MODEL_FILES = [
    "random.joblib",
    "modelos/random.joblib",
    "modelo.joblib",
    "modelos/modelo.joblib",
]

st.set_page_config(
    page_title="Tech Challenge Fase 4 | Obesidade",
    page_icon="⚖️",
    layout="wide",
)

# =============================================================================
# DICIONÁRIOS / MAPEAMENTOS
# =============================================================================

CLASS_LABELS = {
    0: ("Abaixo do Peso", "Insufficient_Weight", "🔵"),
    1: ("Peso Normal", "Normal_Weight", "🟢"),
    2: ("Obesidade Tipo I", "Obesity_Type_I", "🟠"),
    3: ("Obesidade Tipo II", "Obesity_Type_II", "🔴"),
    4: ("Obesidade Tipo III", "Obesity_Type_III", "🔴"),
    5: ("Sobrepeso Nível I", "Overweight_Level_I", "🟡"),
    6: ("Sobrepeso Nível II", "Overweight_Level_II", "🟡"),
}

CLASS_TRANSLATION = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III",
}

CLASS_ORDER_PT = [
    "Abaixo do Peso",
    "Peso Normal",
    "Sobrepeso Nível I",
    "Sobrepeso Nível II",
    "Obesidade Tipo I",
    "Obesidade Tipo II",
    "Obesidade Tipo III",
]

GENDER_MAP = {"Mulher": 0, "Homem": 1}
YESNO_MAP = {"Sim": 1, "Não": 0}
CAEC_MAP = {"Sempre": 0, "Frequentemente": 1, "Às vezes": 2, "Não": 3}
CALC_MAP = {"Sempre": 0, "Frequentemente": 1, "Às vezes": 2, "Não": 3}
MTRANS_MAP = {
    "Carro": 0,
    "Bicicleta": 1,
    "Moto": 2,
    "Transporte público": 3,
    "A pé": 4,
}
FCVC_MAP = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}
NCP_MAP = {"1": 1, "2": 2, "3": 3, "Mais de 3": 4}
CH2O_MAP = {"Menos de 1L": 1, "1–2L": 2, "Mais de 2L": 3}
FAF_MAP = {
    "Nenhuma": 0,
    "1–2x / semana": 1,
    "3–4x / semana": 2,
    "5x / semana ou mais": 3,
}
TUE_MAP = {"0–2h / dia": 0, "3–5h / dia": 1, "Mais de 5h / dia": 2}

GENDER_PT = {"Female": "Mulher", "Male": "Homem"}
YES_NO_PT = {"yes": "Sim", "no": "Não"}
CAEC_PT = {
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre",
}
CALC_PT = {
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre",
}
MTRANS_PT = {
    "Automobile": "Carro",
    "Motorbike": "Moto",
    "Bike": "Bicicleta",
    "Public_Transportation": "Transporte público",
    "Walking": "A pé",
}

DICT_DADOS = pd.DataFrame(
    [
        ["Gender", "Gênero"],
        ["Age", "Idade"],
        ["Height", "Altura em metros"],
        ["Weight", "Peso em kg"],
        ["family_history", "Histórico familiar de excesso de peso"],
        ["FAVC", "Consumo frequente de alimentos calóricos"],
        ["FCVC", "Frequência de consumo de vegetais"],
        ["NCP", "Número de refeições principais"],
        ["CAEC", "Consumo de alimentos entre refeições"],
        ["SMOKE", "Fuma"],
        ["CH2O", "Consumo diário de água"],
        ["SCC", "Monitoramento de calorias"],
        ["FAF", "Frequência de atividade física"],
        ["TUE", "Tempo de uso de dispositivos tecnológicos"],
        ["CALC", "Consumo de álcool"],
        ["MTRANS", "Meio de transporte habitual"],
        ["Obesity", "Nível de obesidade / classe alvo"],
    ],
    columns=["Campo", "Descrição"],
)


# =============================================================================
# FUNÇÕES DE CARGA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Carrega a primeira base encontrada na pasta do projeto."""
    for file_name in DATA_FILES:
        path = Path(file_name)
        if path.exists():
            return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    """Carrega o modelo treinado, mantendo compatibilidade com a estrutura atual."""
    for file_name in MODEL_FILES:
        path = Path(file_name)
        if path.exists():
            return joblib.load(path), str(path)

    return None, None


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas auxiliares para análise visual."""
    if df.empty:
        return df

    data = df.copy()

    target_col = None
    for possible_col in ["Obesity", "Obesity_level", "NObeyesdad"]:
        if possible_col in data.columns:
            target_col = possible_col
            break

    if target_col is None:
        st.error("Não encontrei a coluna alvo de obesidade na base.")
        return pd.DataFrame()

    data["Classe Técnica"] = data[target_col].astype(str)
    data["Classe"] = data["Classe Técnica"].map(CLASS_TRANSLATION).fillna(data["Classe Técnica"])

    if "Gender" in data.columns:
        data["Gênero"] = data["Gender"].map(GENDER_PT).fillna(data["Gender"])

    if "Height" in data.columns and "Weight" in data.columns:
        data["IMC"] = data["Weight"] / (data["Height"] ** 2)

    if "family_history" in data.columns:
        data["Histórico Familiar"] = data["family_history"].map(YES_NO_PT).fillna(data["family_history"])

    if "FAVC" in data.columns:
        data["Alimentos Calóricos"] = data["FAVC"].map(YES_NO_PT).fillna(data["FAVC"])

    if "CAEC" in data.columns:
        data["Lanches entre Refeições"] = data["CAEC"].map(CAEC_PT).fillna(data["CAEC"])

    if "CALC" in data.columns:
        data["Consumo de Álcool"] = data["CALC"].map(CALC_PT).fillna(data["CALC"])

    if "MTRANS" in data.columns:
        data["Transporte"] = data["MTRANS"].map(MTRANS_PT).fillna(data["MTRANS"])

    return data


# =============================================================================
# FUNÇÕES DE GRÁFICOS
# =============================================================================

def bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    sort: str | list | None = "-x",
):
    """Cria gráfico de barras simples."""
    encoding = {
        "x": alt.X(x, title=x.split(":")[0]),
        "y": alt.Y(y, sort=sort, title=y.split(":")[0]),
        "tooltip": [x, y],
    }

    if color:
        encoding["color"] = alt.Color(color, legend=alt.Legend(title=color.split(":")[0]))

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(**encoding)
        .properties(title=title, height=360)
    )

    st.altair_chart(chart, use_container_width=True)


def metric_card(label: str, value: str, help_text: str | None = None):
    st.metric(label=label, value=value, help=help_text)


def get_class_info(prediction):
    """Converte saída do modelo para rótulo amigável."""
    if isinstance(prediction, (np.integer, int)) or str(prediction).isdigit():
        idx = int(prediction)
        return CLASS_LABELS.get(idx, ("Classe desconhecida", str(prediction), "⚪"))

    pred_str = str(prediction)
    pt = CLASS_TRANSLATION.get(pred_str, pred_str)
    emoji = "⚪"
    for _, (label_pt, label_en, label_emoji) in CLASS_LABELS.items():
        if pred_str == label_en or pred_str == label_pt:
            emoji = label_emoji
            break

    return pt, pred_str, emoji


# =============================================================================
# COMPONENTES DA APLICAÇÃO
# =============================================================================

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("⚖️ Tech Challenge Fase 4")
    st.sidebar.caption("Sistema preditivo + dashboard analítico")

    if hasattr(st.sidebar, "link_button"):
        st.sidebar.link_button("🔗 Abrir app publicado", APP_URL)
    else:
        st.sidebar.markdown(f"[🔗 Abrir app publicado]({APP_URL})")

    st.sidebar.divider()
    st.sidebar.subheader("Filtros do dashboard")

    if df.empty:
        st.sidebar.warning("Base não carregada.")
        return df

    filtered = df.copy()

    if "Classe" in filtered.columns:
        classes = sorted(filtered["Classe"].dropna().unique().tolist())
        selected_classes = st.sidebar.multiselect(
            "Classe de obesidade",
            classes,
            default=classes,
        )
        filtered = filtered[filtered["Classe"].isin(selected_classes)]

    if "Gênero" in filtered.columns:
        genders = sorted(filtered["Gênero"].dropna().unique().tolist())
        selected_genders = st.sidebar.multiselect(
            "Gênero",
            genders,
            default=genders,
        )
        filtered = filtered[filtered["Gênero"].isin(selected_genders)]

    if "Age" in filtered.columns and not filtered["Age"].dropna().empty:
        min_age = int(np.floor(filtered["Age"].min()))
        max_age = int(np.ceil(filtered["Age"].max()))
        age_range = st.sidebar.slider(
            "Faixa de idade",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
        )
        filtered = filtered[
            (filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])
        ]

    return filtered


def render_predictive_tab(model):
    st.header("🔍 Sistema Preditivo")
    st.markdown(
        """
        Esta aba mantém o **modelo de análise/predição** que você já tinha no projeto.
        O usuário informa os dados do paciente e o modelo retorna a classificação prevista.
        """
    )
    st.info(
        "Uso acadêmico: o resultado auxilia a análise do projeto, mas não substitui avaliação médica profissional."
    )

    if model is None:
        st.warning(
            "Modelo `random.joblib` não encontrado. O dashboard funciona, mas a predição só aparece quando o arquivo do modelo estiver na raiz do projeto."
        )

    with st.form("form_paciente"):
        st.subheader("Dados Pessoais")
        genero = st.radio("Gênero", ["Homem", "Mulher"], horizontal=True)
        idade = st.number_input("Idade", min_value=14, max_value=100, value=25)

        col1, col2 = st.columns(2)
        with col1:
            peso = st.slider("Peso (kg)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        with col2:
            altura = st.slider("Altura (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)

        st.subheader("Histórico & Hábitos Alimentares")
        historico_familiar = st.radio(
            "Histórico familiar de obesidade", ["Sim", "Não"], horizontal=True
        )
        favc = st.radio("Consumo de alimentos calóricos (FAVC)", ["Sim", "Não"], horizontal=True)
        fcvc = st.select_slider(
            "Consumo de vegetais (FCVC)",
            options=["Raramente", "Às vezes", "Sempre"],
            value="Às vezes",
        )
        ncp = st.select_slider(
            "Número de refeições principais (NCP)",
            options=["1", "2", "3", "Mais de 3"],
            value="3",
        )
        caec = st.selectbox(
            "Consumo de lanches entre refeições (CAEC)",
            ["Não", "Às vezes", "Frequentemente", "Sempre"],
        )

        st.subheader("Monitoramento & Estilo de Vida")
        smoke = st.radio("Fuma? (SMOKE)", ["Sim", "Não"], horizontal=True)
        ch2o = st.select_slider(
            "Consumo diário de água (CH2O)",
            options=["Menos de 1L", "1–2L", "Mais de 2L"],
            value="1–2L",
        )
        scc = st.radio("Monitora as calorias diárias? (SCC)", ["Sim", "Não"], horizontal=True)
        faf = st.select_slider(
            "Frequência semanal de atividade física (FAF)",
            options=["Nenhuma", "1–2x / semana", "3–4x / semana", "5x / semana ou mais"],
            value="1–2x / semana",
        )
        tue = st.select_slider(
            "Tempo de uso de tela por dia (TUE)",
            options=["0–2h / dia", "3–5h / dia", "Mais de 5h / dia"],
            value="3–5h / dia",
        )
        calc = st.selectbox(
            "Consumo de álcool (CALC)",
            ["Não", "Às vezes", "Frequentemente", "Sempre"],
        )

        st.subheader("Transporte Habitual")
        mtrans = st.radio(
            "Meio de transporte principal (MTRANS)",
            ["Carro", "Moto", "Bicicleta", "Transporte público", "A pé"],
            horizontal=True,
        )

        submitted = st.form_submit_button("🔍 Classificar", use_container_width=True)

    if submitted:
        if model is None:
            st.error("Não foi possível classificar porque o modelo não foi encontrado.")
            return

        altura_m = altura / 100.0

        # A ordem das colunas segue o modelo original.
        input_values = [
            GENDER_MAP[genero],
            float(idade),
            altura_m,
            float(peso),
            YESNO_MAP[historico_familiar],
            YESNO_MAP[favc],
            float(FCVC_MAP[fcvc]),
            float(NCP_MAP[ncp]),
            float(CAEC_MAP[caec]),
            YESNO_MAP[smoke],
            float(CH2O_MAP[ch2o]),
            YESNO_MAP[scc],
            float(FAF_MAP[faf]),
            float(TUE_MAP[tue]),
            float(CALC_MAP[calc]),
            float(MTRANS_MAP[mtrans]),
        ]

        if hasattr(model, "feature_names_in_"):
            input_columns = list(model.feature_names_in_)
        else:
            input_columns = [
                "Gender",
                "Age",
                "Height",
                "Weight",
                "family_history",
                "FAVC",
                "FCVC",
                "NCP",
                "CAEC",
                "SMOKE",
                "CH2O",
                "SCC",
                "FAF",
                "TUE",
                "CALC",
                "MTRANS",
            ]

        input_data = pd.DataFrame([input_values], columns=input_columns)

        pred = model.predict(input_data)[0]
        label_pt, label_en, emoji = get_class_info(pred)

        st.divider()
        st.subheader("Resultado da Classificação")

        col_res, col_conf, col_imc = st.columns(3)
        with col_res:
            st.metric("Diagnóstico previsto", f"{emoji} {label_pt}")
            st.caption(f"Classe técnica: `{label_en}`")

        with col_imc:
            imc = peso / (altura_m ** 2)
            st.metric("IMC informado", f"{imc:.1f}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = float(np.max(proba)) * 100

            with col_conf:
                st.metric("Confiança do modelo", f"{confidence:.1f}%")

            if hasattr(model, "classes_"):
                classes_model = list(model.classes_)
            else:
                classes_model = list(range(len(proba)))

            proba_df = []
            for classe, prob in zip(classes_model, proba):
                classe_pt, classe_en, _ = get_class_info(classe)
                proba_df.append(
                    {
                        "Classe": classe_pt,
                        "Classe técnica": classe_en,
                        "Probabilidade": float(prob) * 100,
                    }
                )

            proba_df = pd.DataFrame(proba_df).sort_values("Probabilidade", ascending=False)

            chart = (
                alt.Chart(proba_df)
                .mark_bar()
                .encode(
                    x=alt.X("Probabilidade:Q", title="Probabilidade (%)"),
                    y=alt.Y("Classe:N", sort="-x", title="Classe"),
                    tooltip=["Classe", "Classe técnica", alt.Tooltip("Probabilidade:Q", format=".1f")],
                )
                .properties(title="Probabilidade por classe", height=350)
            )
            st.altair_chart(chart, use_container_width=True)


def render_dashboard_tab(df: pd.DataFrame):
    st.header("📊 Dashboard Analítico")
    st.markdown(
        """
        Esta aba atende ao requisito de **visão analítica em painel**, trazendo os principais
        indicadores e relações da base para apoiar a leitura da equipe médica.
        """
    )

    if df.empty:
        st.error("Base não carregada. Confirme se `df_clean.csv` ou `Obesity.csv` está na raiz do projeto.")
        return

    total = len(df)
    avg_age = df["Age"].mean() if "Age" in df.columns else 0
    avg_weight = df["Weight"].mean() if "Weight" in df.columns else 0
    avg_bmi = df["IMC"].mean() if "IMC" in df.columns else 0

    overweight_obesity_pct = 0
    if "Classe Técnica" in df.columns:
        overweight_obesity_pct = (
            df["Classe Técnica"].str.contains("Overweight|Obesity", case=False, na=False).mean() * 100
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Registros analisados", f"{total:,}".replace(",", "."))
    with col2:
        metric_card("Idade média", f"{avg_age:.1f} anos")
    with col3:
        metric_card("Peso médio", f"{avg_weight:.1f} kg")
    with col4:
        metric_card("% Sobrepeso/Obesidade", f"{overweight_obesity_pct:.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        if "Classe" in df.columns:
            dist = (
                df.groupby("Classe", as_index=False)
                .size()
                .rename(columns={"size": "Quantidade"})
            )
            dist["Classe"] = pd.Categorical(dist["Classe"], categories=CLASS_ORDER_PT, ordered=True)
            dist = dist.sort_values("Classe")
            bar_chart(
                dist,
                x="Quantidade:Q",
                y="Classe:N",
                title="Distribuição dos níveis de obesidade",
                sort=list(reversed(CLASS_ORDER_PT)),
            )

    with col_right:
        if "Classe" in df.columns and "IMC" in df.columns:
            bmi_class = (
                df.groupby("Classe", as_index=False)["IMC"]
                .mean()
                .rename(columns={"IMC": "IMC Médio"})
            )
            bmi_class["Classe"] = pd.Categorical(bmi_class["Classe"], categories=CLASS_ORDER_PT, ordered=True)
            bmi_class = bmi_class.sort_values("Classe")
            bar_chart(
                bmi_class,
                x="IMC Médio:Q",
                y="Classe:N",
                title="IMC médio por classe",
                sort=list(reversed(CLASS_ORDER_PT)),
            )

    col_left, col_right = st.columns(2)

    with col_left:
        if {"Height", "Weight", "Classe"}.issubset(df.columns):
            scatter = (
                alt.Chart(df)
                .mark_circle(size=70, opacity=0.55)
                .encode(
                    x=alt.X("Height:Q", title="Altura (m)"),
                    y=alt.Y("Weight:Q", title="Peso (kg)"),
                    color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
                    tooltip=[
                        alt.Tooltip("Age:Q", title="Idade"),
                        alt.Tooltip("Height:Q", title="Altura", format=".2f"),
                        alt.Tooltip("Weight:Q", title="Peso", format=".1f"),
                        "Classe:N",
                    ],
                )
                .properties(title="Relação entre altura, peso e classe", height=390)
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)

    with col_right:
        if {"Age", "Classe"}.issubset(df.columns):
            age_hist = (
                alt.Chart(df)
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X("Age:Q", bin=alt.Bin(maxbins=20), title="Idade"),
                    y=alt.Y("count():Q", title="Quantidade"),
                    color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
                    tooltip=["count():Q"],
                )
                .properties(title="Distribuição de idade por classe", height=390)
            )
            st.altair_chart(age_hist, use_container_width=True)

    st.subheader("Hábitos e fatores comportamentais")

    col_left, col_right = st.columns(2)

    with col_left:
        if {"Histórico Familiar", "Classe"}.issubset(df.columns):
            fam = (
                df.groupby(["Histórico Familiar", "Classe"], as_index=False)
                .size()
                .rename(columns={"size": "Quantidade"})
            )
            chart = (
                alt.Chart(fam)
                .mark_bar()
                .encode(
                    x=alt.X("Histórico Familiar:N", title="Histórico familiar"),
                    y=alt.Y("Quantidade:Q", title="Quantidade"),
                    color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
                    tooltip=["Histórico Familiar", "Classe", "Quantidade"],
                )
                .properties(title="Histórico familiar x classe", height=360)
            )
            st.altair_chart(chart, use_container_width=True)

    with col_right:
        if {"Alimentos Calóricos", "Classe"}.issubset(df.columns):
            favc = (
                df.groupby(["Alimentos Calóricos", "Classe"], as_index=False)
                .size()
                .rename(columns={"size": "Quantidade"})
            )
            chart = (
                alt.Chart(favc)
                .mark_bar()
                .encode(
                    x=alt.X("Alimentos Calóricos:N", title="Consumo calórico frequente"),
                    y=alt.Y("Quantidade:Q", title="Quantidade"),
                    color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
                    tooltip=["Alimentos Calóricos", "Classe", "Quantidade"],
                )
                .properties(title="Consumo de alimentos calóricos x classe", height=360)
            )
            st.altair_chart(chart, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        if {"FAF", "Classe"}.issubset(df.columns):
            faf = (
                df.groupby(["FAF", "Classe"], as_index=False)
                .size()
                .rename(columns={"size": "Quantidade"})
            )
            faf["FAF"] = faf["FAF"].astype(str)
            chart = (
                alt.Chart(faf)
                .mark_bar()
                .encode(
                    x=alt.X("FAF:N", title="Frequência de atividade física"),
                    y=alt.Y("Quantidade:Q", title="Quantidade"),
                    color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
                    tooltip=["FAF", "Classe", "Quantidade"],
                )
                .properties(title="Atividade física x classe", height=360)
            )
            st.altair_chart(chart, use_container_width=True)

    with col_right:
        if {"Transporte", "Classe"}.issubset(df.columns):
            mtrans = (
                df.groupby(["Transporte", "Classe"], as_index=False)
                .size()
                .rename(columns={"size": "Quantidade"})
            )
            chart = (
                alt.Chart(mtrans)
                .mark_bar()
                .encode(
                    x=alt.X("Transporte:N", title="Transporte habitual"),
                    y=alt.Y("Quantidade:Q", title="Quantidade"),
                    color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
                    tooltip=["Transporte", "Classe", "Quantidade"],
                )
                .properties(title="Meio de transporte x classe", height=360)
            )
            st.altair_chart(chart, use_container_width=True)


def render_insights_tab(df: pd.DataFrame):
    st.header("🧠 Principais Insights para a Equipe Médica")

    if df.empty:
        st.error("Base não carregada.")
        return

    insights = []

    if "Classe" in df.columns:
        main_class = df["Classe"].value_counts().idxmax()
        main_class_qtd = int(df["Classe"].value_counts().max())
        insights.append(
            f"Na base filtrada, a classe com maior volume é **{main_class}**, com **{main_class_qtd} registros**."
        )

    if "IMC" in df.columns and "Classe" in df.columns:
        bmi_by_class = df.groupby("Classe")["IMC"].mean().sort_values(ascending=False)
        top_bmi_class = bmi_by_class.index[0]
        top_bmi_value = bmi_by_class.iloc[0]
        insights.append(
            f"O maior IMC médio aparece em **{top_bmi_class}**, com média aproximada de **{top_bmi_value:.1f}**."
        )

    if {"Histórico Familiar", "Classe Técnica"}.issubset(df.columns):
        high_risk = df[df["Classe Técnica"].str.contains("Overweight|Obesity", case=False, na=False)]
        if len(high_risk) > 0:
            fam_yes = (high_risk["Histórico Familiar"] == "Sim").mean() * 100
            insights.append(
                f"Entre os registros de sobrepeso/obesidade, **{fam_yes:.1f}%** possuem histórico familiar marcado como 'Sim'."
            )

    if {"Alimentos Calóricos", "Classe Técnica"}.issubset(df.columns):
        high_risk = df[df["Classe Técnica"].str.contains("Overweight|Obesity", case=False, na=False)]
        if len(high_risk) > 0:
            favc_yes = (high_risk["Alimentos Calóricos"] == "Sim").mean() * 100
            insights.append(
                f"Entre os registros de sobrepeso/obesidade, **{favc_yes:.1f}%** indicam consumo frequente de alimentos calóricos."
            )

    st.subheader("Resumo executivo")
    for i, insight in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {insight}")

    st.subheader("Leitura de negócio")
    st.markdown(
        """
        Para uma apresentação de negócio, o painel mostra que a aplicação não é apenas um formulário.
        Ela permite observar padrões da base, apoiar triagens e transformar o modelo em uma ferramenta
        visual para discussão com a equipe médica.

        **Como vender a ideia na apresentação:**
        - O modelo faz a classificação preditiva.
        - O dashboard explica o comportamento da base.
        - Os filtros permitem analisar perfis específicos.
        - A visão final ajuda a equipe médica a entender padrões de risco na amostra analisada.
        """
    )

    st.warning(
        "Importante: os gráficos mostram associações na base analisada. Eles não provam causalidade e não substituem avaliação clínica."
    )


def render_data_tab(df: pd.DataFrame):
    st.header("📁 Dados e Dicionário")

    st.subheader("Dicionário de dados")
    st.dataframe(DICT_DADOS, use_container_width=True, hide_index=True)

    st.subheader("Amostra da base carregada")
    if df.empty:
        st.error("Base não carregada.")
        return

    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Ver estatísticas numéricas"):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.info("Não há colunas numéricas para descrever.")



# =============================================================================
# MAIN
# =============================================================================

def main():
    raw_df = load_dataset()
    df = prepare_dataset(raw_df)
    model, _model_path = load_model()

    filtered_df = render_sidebar(df)

    st.title("⚖️ Modelo de ML para Auxílio de Diagnóstico de Obesidade")
    st.caption("Tech Challenge FIAP - Fase 4 | Machine Learning em Produção com Streamlit")

    tab_pred, tab_dash, tab_insights, tab_data = st.tabs(
        [
            "🔍 Sistema Preditivo",
            "📊 Dashboard Analítico",
            "🧠 Insights Médicos",
            "📁 Dados",
        ]
    )

    with tab_pred:
        render_predictive_tab(model)

    with tab_dash:
        render_dashboard_tab(filtered_df)

    with tab_insights:
        render_insights_tab(filtered_df)

    with tab_data:
        render_data_tab(df)



if __name__ == "__main__":
    main()
