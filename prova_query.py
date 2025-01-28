import os
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import weaviate
import torch
from weaviate.gql.get import HybridFusion
#from weaviate.gql.query import HybridFusion

# Configurazione Weaviate
cloud_name = "SentenzeToscana"

key = os.getenv('WEAVIATE_API_KEY')
if not key:
    raise ValueError("La chiave API di Weaviate non è configurata. Verifica i secrets su Streamlit Cloud.")

url = "https://wdddqkdjqfwdxaklotxqqa.c0.europe-west3.gcp.weaviate.cloud"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
model_rag = load_model()


@st.cache_resource
def get_weaviate_client():
    return weaviate.Client(
    url=url,
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=key),
)
    
client = get_weaviate_client()  

# Funzione per generare gli embedding
def generate_embeddings(entry, model=model_rag):
    embedding = model.encode(entry, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def collect_paths(data):
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "path":
                paths.append(value[0])
            else:
                paths.extend(collect_paths(value))

    elif isinstance(data, list):
        for item in data:
            paths.extend(collect_paths(item))

    return paths



# Funzione per eseguire la query su Weaviate
def query_weaviate(query, num_max, alpha, filters, search_prop=["testo_parziale", 'summary'],
                   retrieved_proop=["testo_parziale", 'estrazione_mistral', 'summary', 'testo_completo', 'id_originale']):
    more_prop = collect_paths(filters)
    if len(more_prop)>=1:
        retrieved_proop = [*retrieved_proop, *more_prop]
        response = (
            client.query
            .get("TestoCompleto", retrieved_proop)
            .with_hybrid(
                query=query,
                vector=list(generate_embeddings(query)),
                properties=search_prop,
                alpha=alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
            )
            .with_where(filters)
            .do())
    else:
        response = (
            client.query
            .get("TestoCompleto", retrieved_proop)
            .with_hybrid(
                query=query,
                vector=list(generate_embeddings(query)),
                properties=search_prop,
                alpha=alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
            ).do())
        
    ids = []
    risposta_finale = []
    try:
        for i in response['data']['Get']["TestoCompleto"]:
            ids_temp = i['id_originale']
            if ids_temp not in ids:
                ids.append(ids_temp)
                diz = {}
                diz['id_originale'] = i['id_originale']
                diz['summary'] = i['summary']
                diz['metaDati'] = json.loads(i['estrazione_mistral'])
                risposta_finale.append(diz)
                risposta_finale=risposta_finale[0:num_max]
    except:
        risposta_finale=response
    return risposta_finale

# Struttura JSON con i parametri di filtro
filters_structure = {
    'Info_Generali_Sentenza': {
        "info_generali_sentenza__tipo_separazione": ['giudiziale/contenzioso', 'consensuale/congiunto', 'NON SPECIFICATO'],
        "info_generali_sentenza__modalita_esaurimento": ['omologazione', 'accoglimento', 'conciliazione', 'cambiamento di rito', 'archiviazione', 'estinzione', 'NON SPECIFICATO'],
        "info_generali_sentenza__violenza": ['Sì', 'No', 'NON SPECIFICATO'],
        "info_generali_sentenza__abusi": ['Sì', 'No', 'NON SPECIFICATO']
    },
    'dettagli_matrimonio': {
        "dettagli_matrimonio__rito": ['religioso', 'civile', 'NON SPECIFICATO'],
        "dettagli_matrimonio__regime_patrimoniale": ['comunione dei beni', 'separazione dei beni', 'NON SPECIFICATO']
    },
    'dettagli_figli': {
        "dettagli_figli__numero_di_figli_minorenni": [0, 10],
        "dettagli_figli__numero_di_figli_maggiorenni_economicamente_indipendenti": [0, 10],
        "dettagli_figli__numero_di_figli_maggiorenni_non_economicamente_indipendenti": [0, 10],
        "dettagli_figli__numero_di_figli_portatori_di_handicap": [0, 10],
        "dettagli_figli__tipo_di_affidamento": [
            'esclusivo al padre', 'esclusivo alla madre', 'esclusivo a terzi',
            'condiviso con prevalenza al padre', 'condiviso con prevalenza alla madre',
            'condiviso con frequentazione paritetica', 'NON SPECIFICATO'
        ],
        "dettagli_figli__contributi_economici__contributi_per_il_mantenimento_figli": ['Sì', 'No', 'NON SPECIFICATO'],
        "dettagli_figli__contributi_economici__importo_assegno_per_il_mantenimento_figli": [0, 5000],
        "dettagli_figli__contributi_economici__obbligato_al_mantenimento": ['padre', 'madre', 'NON SPECIFICATO'],
        "dettagli_figli__contributi_economici__beneficiario_assegno_per_mantenimento_figli": ['direttamente ai figli', 'all’altro genitore', 'diviso', 'NON SPECIFICATO'],
        "dettagli_figli__contributi_economici__modalita_pagamento_assegno_di_mantenimento_del_coniuge": ['mensile', 'una tantum', 'NON SPECIFICATO']
    }
}

# Sidebar per la query
st.sidebar.title("Query Settings")
query = st.sidebar.text_input("Inserisci la tua query:", value="violenza o abuso")
num_results = st.sidebar.number_input("Numero massimo di risultati:", min_value=1, max_value=100, value=5)
alpha = st.sidebar.slider(
    "Seleziona soglia di similarità (0-1):",
    min_value=0.0,
    max_value=1.0,
    value=0.8,  # Valore di default
    step=0.01
)



# Generazione dinamica dei filtri con menu a tendina espandibili
st.title("Filtri di Ricerca")

selected_filters = {}

# Itera attraverso i livelli principali del filtro
for main_key, sub_filters in filters_structure.items():
    with st.expander(main_key, expanded=False):
        st.write(f"Seleziona i filtri per **{main_key}**:")
        selected_filters[main_key] = {}
        for sub_key, options in sub_filters.items():
            if isinstance(options, list) and all(isinstance(i, str) for i in options):
                # Menu a tendina per opzioni testuali
                selected_filters[main_key][sub_key] = st.selectbox(
                    f"{sub_key}:",
                    options,
                    index=len(options)-1,
                    key=f"{main_key}_{sub_key}"
                )
            elif isinstance(options, list) and len(options) == 2 and all(isinstance(i, int) for i in options):
                # Slider per range numerici
                selected_filters[main_key][sub_key] = st.slider(
                    f"{sub_key}:",
                    min_value=options[0],
                    max_value=options[1],
                    value=(options[0], options[1]),
                    key=f"{main_key}_{sub_key}"
                )

# Pulsante per eseguire la query
if st.button("Esegui Ricerca"):
    st.write("Filtri selezionati:")
    

    # Convertire i filtri in formato Weaviate
    weaviate_filters = {
        "operator": "And",
        "operands": []
    }

    for main_key, sub_dict in selected_filters.items():
        for sub_key, value in sub_dict.items():
            if isinstance(value, tuple):  # Per range numerici
                if value[0] == 0 and value[1] == 0:
                    continue
                weaviate_filters["operands"].append({
                    "operator": "And",
                    "operands":[
                        {"path": [f"{sub_key}"],
                    "operator": "GreaterThanEqual",
                    "valueInt": value[0]
                    },
                    {
                        "path": [f"{sub_key}"],
                        "operator": "LessThanEqual",
                        "valueInt": value[1]
                    }]}  )
            elif value != "NON SPECIFICATO":  # Applica il filtro solo se diverso da "NON SPECIFICATO"
                weaviate_filters["operands"].append({
                    "path": [f"{sub_key}"],
                    "operator": "Equal",
                    "valueText": value
                })
    st.json(weaviate_filters)

    # Verifica se ci sono filtri validi
    if not weaviate_filters["operands"]:
        st.warning("Nessun filtro selezionato. Verrà eseguita una ricerca senza filtri.")
        # Esegui la query
        risultati = query_weaviate(query, num_results, alpha, {})
    else:
        risultati = query_weaviate(query, num_results, alpha, weaviate_filters)


    
    st.write("Risultati:")
    try:
        for r in risultati:
            st.write(f"**ID: {r['id_originale']}**")
            st.write(f"# Summary:")
            st.write(r['summary'])
            st.write("# Meta-Dati:")
            st.json(r['metaDati'])
            st.write("\n----------------------\n\n")
    except:
        st.write(risultati)
