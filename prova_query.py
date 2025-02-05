import os
import re
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import weaviate
import torch
from weaviate.gql.get import HybridFusion
#from weaviate.gql.query import HybridFusion

st.markdown(
    """
    <style>
        /* Cambia il colore dello sfondo generale */
        body, .stApp {
            background-color: white !important;
        }

        /* Cambia il colore della sidebar (grigio chiaro) */
        .stSidebar {
            background-color: #f0f0f0 !important;
        }

        /* Box dei filtri con background scuro */
        div.stExpander {
            background-color: #222 !important;  /* Nero scuro */
            border-radius: 10px;  /* Angoli arrotondati */
            padding: 10px;
        }

        /* Cambia colore del testo dentro i filtri */
        div.stExpander * {
            color: white !important;  /* Testo bianco */
        }


        /* Cambia colore dei pulsanti */
        .stButton>button {
            background-color: #007BFF !important;  /* Blu */
            color: white !important;
            border-radius: 8px;
            padding: 10px 20px;
        }

        /* Cambia colore quando il pulsante è premuto */
        .stButton>button:hover {
            background-color: #0056b3 !important;
        }


        .stDownloadButton > button {
            color: white !important;   /* Testo bianco */
            background-color: #1E90FF !important; /* Blu più chiaro */
            border-radius: 8px !important;  /* Angoli arrotondati */
            padding: 8px 16px !important;  /* Spaziatura interna */
            font-size: 16px !important;  /* Testo più leggibile */
            font-weight: bold !important; /* Testo più visibile */
        }


        
        div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
        background: #007BFF !important; }
        

        div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
        background-color: #007BFF !important;
        box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;}

        div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: #007BFF !important; }




        /* Linea di separazione nera */
        hr {
            border: 1px solid black !important;
        }


        


        /* Stile personalizzato per la separazione testuale */
        .custom-divider {
            color: black !important;
            font-weight: bold;
            font-size: 18px;
        }

        /* Cambia colore dei titoli */
        h1, h2, h3, h4, h5, h6 {
            color: #333 !important;
        }

        /* Cambia colore del testo normale */
        p, div {
            color: #444 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Configurazione Weaviate
cloud_name = "TestoCompleto"

key = os.getenv('WEAVIATE_API_KEY')
if not key:
    raise ValueError("La chiave API di Weaviate non è configurata. Verifica i secrets su Streamlit Cloud.")

url = "https://wdddqkdjqfwdxaklotxqqa.c0.europe-west3.gcp.weaviate.cloud"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
model_rag = load_model()


#@st.cache_resource
#def get_weaviate_client():
#    return weaviate.Client(
#    url=url,
#    auth_client_secret=weaviate.auth.AuthApiKey(api_key=key),
#)
    
#client = get_weaviate_client()  

client = weaviate.Client(
    url=url,
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=key),
)

# Funzione per generare gli embedding
def generate_embeddings(entry, model=model_rag):
    embedding = model.encode(entry,  convert_to_numpy=True, normalize_embeddings=True)
    return np.array([i for i in embedding])


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


def scores(text):
    # Trova tutti i valori dopo 'normalized score:'
    matches = re.findall(r'normalized score:\s*([\d.]+)', text)
    return [float(x) for x in matches]




# Funzione per eseguire la query su Weaviate
def query_weaviate(query, num_max, alpha, filters, search_prop=["testo_parziale", 'summary'],
                   retrieved_proop=["riferimenti_legge","testo_parziale", 'estrazione_mistral', 'summary', 'testo_completo', 'id_originale']):
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
            ).with_where(filters).with_additional(["score", "explainScore"]).do())
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
            ).with_additional(["score", "explainScore"]).do())
        
    ids = []
    risposta_finale = []
    try:
        for i in response['data']['Get']["TestoCompleto"]:
            ids_temp = i['id_originale']
            if ids_temp not in ids:
                ids.append(ids_temp)
                diz = {}
                try:
                    diz['query_score'] = float(i["_additional"]["score"])
                except:
                    diz['query_score'] = i["_additional"]["explainScore"]
                diz['id_originale'] = i['id_originale']
                diz['summary'] = i['summary']
                diz['testo_completo'] = i['testo_completo']
                diz['metaDati'] = json.loads(i['estrazione_mistral'])
                diz['riferimenti_legge']= i["riferimenti_legge"]
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
        "dettagli_figli__numero_totale_di_figli": [0, 10],
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
num_results = st.sidebar.number_input("Numero massimo di risultati:", min_value=1, max_value=100, value=5)
alpha = st.sidebar.slider(
    "Seleziona tipo di ricerca:  \n[0: solo keywords - 1: solo semantica]",
    min_value=0.0,
    max_value=1.0,
    value=0.8,  # Valore di default
    step=0.01
)


st.title("Query")
query = st.text_input("Inserisci la tua query:", value=" ")
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
                    f"{sub_key.split('__')[-1].replace('_',' ')}:",
                    options,
                    index=len(options)-1,
                    key=f"{main_key}_{sub_key}"
                )
            elif isinstance(options, list) and len(options) == 2 and all(isinstance(i, int) for i in options):
                # Slider per range numerici
                selected_filters[main_key][sub_key] = st.slider(
                    f"{sub_key.split('__')[-1].replace('_',' ')}:",
                    min_value=options[0],
                    max_value=options[1],
                    value=(options[0], options[1]),
                    key=f"{main_key}_{sub_key}"
                )

# Pulsante per eseguire la query
if st.button("Esegui Ricerca"):
    #st.write("Filtri selezionati:")
    

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
                elif sub_key=="dettagli_figli__numero_totale_di_figli":
                    weaviate_filters["operands"].append({
                        "operator": "And",
                        "operands":[
                            {"path": [f"{sub_key}"],
                        "operator": "GreaterThanEqual",
                        "valueNumber": value[0]
                        },
                        {
                            "path": [f"{sub_key}"],
                            "operator": "LessThanEqual",
                            "valueNumber": value[1]
                        }]}  )
                else:
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
    #st.json(weaviate_filters)

    # Verifica se ci sono filtri validi
    if not weaviate_filters["operands"]:
        st.warning("⚠️Nessun filtro selezionato. Verrà eseguita una ricerca senza filtri.")
        # Esegui la query
        risultati = query_weaviate(query, num_results, alpha, {})
    else:
        risultati = query_weaviate(query, num_results, alpha, weaviate_filters)


    
    st.write("# Risultati:")
    st.write("\n----------------------\n\n")
    try:
        if len(risultati)>1:
            for r in risultati:
                #st.write(f"**ID: {r['id_originale']}**")
                 # Creazione del file di testo per il download
                testo_sentenza = r['testo_completo']
                file_name = f"Sentenza_{r['id_originale']}.txt"
                if r['query_score'] < 0.3:
                    st.write("""‼️‼️Tutti i risultati seguenti hanno uno score di similarità troppo basso con la query inserita.""")
                    break
                st.write(f"Query hybrid score: {r['query_score']}")
                # Bottone per il download
                st.download_button(
                    label=f"📥 Scarica Sentenza",# {r['id_originale']}",
                    data=testo_sentenza,
                    file_name=file_name,
                    mime="text/plain"
                )
                st.write(f"## Summary:")
                st.write(r['summary'])
                with st.expander("📜 Riferimenti ad articoli e leggi (clicca per visualizzare)"):
                    try:
                        rr = json.loads(r['riferimenti_legge'])
                        stringx = []
                        for nome, http in rr.items():
                            if nome == "":
                                continue
                            stringx.append(f"[{nome}]({http})")
                        if len(stringx)==0:
                            st.write("‼️Non sono stati trovati riferimenti di legge.")
                        else:
                            for strs in stringx:
                                st.write(strs)  # Link cliccabili
                    except:
                        st.write("‼️Non sono stati trovati riferimenti di legge.")
                        
                with st.expander("Meta-Dati (clicca per visualizzare)"):
                    st.json(r['metaDati'])
                
                st.write("\n----------------------\n\n")
        else:
            st.write("🚨Nessun risultato compatibile con i criteri di ricerca🚨")
    except:
        #st.write(list(generate_embeddings(query)))
        st.write(risultati)
