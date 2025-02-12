import streamlit as st
from dotenv import load_dotenv
import moderation as md
import os
import time
import tempfile
import boto3

st.set_page_config(
    
    page_title="Content moderator Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "mailto:mathis.caprin@gmail.com",
        'About': "#Ce projet a été réalisé en classe de machine learning à SUP DE VINCI",
    }
)

def uploadPhoto(file):
    aws_session = boto3.Session(
        aws_access_key_id=st.session_state.accesskey,      # Récupère l'ID de clé d'accès depuis les variables d'environnement.
        aws_secret_access_key=st.session_state.secretkey,    # Récupère la clé d'accès secrète depuis les variables d'environnement.
        region_name="us-east-1"                           # Spécifie la région AWS à utiliser.
    )
    #aws_session = md.get_aws_session()

    s3 = aws_session.client('s3')
    rekognition = aws_session.client('rekognition')
    transcribe = aws_session.client('transcribe')
    comprehend = aws_session.client('comprehend')
    if st.session_state.currentFileType == "vidéo":
        s3.upload_file(file, st.session_state.bucket, os.path.basename(file))

    st.success("Connected !")
    st.session_state.resultat = md.process_media(file, rekognition, transcribe, comprehend, st.session_state.bucket)
    st.session_state.keywords = st.session_state.resultat["hashtags"]
    if st.session_state.currentFileType == "vidéo" and st.session_state.resultat["moderated"] == True:
        st.session_state.transcription = st.session_state.resultat["subtitles"]
    return st.session_state.resultat["moderated"] == True
    
def mainpage():
    st.title("Content moderator Pro")
    st.write("Le but de l'application est de modérer automatiquement vos photos et vidéos")
    st.write("")
    card_style = """
            <style>
                .card-container {
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 10px; 
                }
                .card {
                    display: inline-block;
                    padding: 5px;
                    margin: 10px;
                    border-radius: 15px;
                    background-color: #7ba1dd;
                    text-align: center;
                    width: auto;
                }
                .card:hover {
                    background-color: #e1f5fe;
                }
            </style>
    """
    if st.session_state.accesskey and st.session_state.secretkey and st.session_state.bucket:
        st.success("ACCESSKEY chargée ✅ : "+ st.session_state.accesskey)
        file = st.file_uploader("Choisissez un ficher (image ou vidéo)")
        if file is not None:
                with st.spinner("Wait for it..."):
                    st.success("Done!")
                    st.session_state.attente = "Upload file"
                    file_extension = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        # Copie le contenu du fichier uploadé dans le fichier temporaire
                        tmp_file.write(file.getvalue())
                        # Récupère le chemin du fichier temporaire
                        temp_file_path = tmp_file.name
                        type_file = md.check_filetype(temp_file_path)
                        st.session_state.currentFileType = md.check_filetype(temp_file_path)
                        moderated = uploadPhoto(temp_file_path)
                        if moderated:
                            if type_file == "vidéo":
                                video_file = open(temp_file_path, "rb")
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                            else:
                                st.image(temp_file_path)
                        else:
                            st.error("❌ Le contenu n'est pas approprié")
                        st.markdown(card_style, unsafe_allow_html=True)
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        for keyword in st.session_state.keywords:
                            st.markdown(f'<div class="card">{keyword}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
    st.text_area("Transcription", value=st.session_state.transcription, key="transcription")
def sidebar():
    st.sidebar.title("Configuration")
    st.sidebar.write("Chargez vos credentials depuis un fichier .env")
    env_file = st.sidebar.file_uploader("Choisir un fichier .env", type=["env"])
    if env_file is not None:
        try:
            with open(".env", "wb") as f:
                f.write(env_file.getbuffer())
            load_dotenv(".env")
            st.session_state.accesskey =  os.getenv("ACCESS_KEY")
            st.session_state.secretkey = os.getenv("SECRET_KEY")
            st.sidebar.success("Credentials chargés depuis le fichier .env")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur lors du chargement du fichier : {e}")

    accesskey = st.sidebar.text_input(
        "ACCESSKEY", value=st.session_state.accesskey, type="password", key="accesskey"
    )
    secretkey = st.sidebar.text_input(
        "SECRETKEY", value=st.session_state.secretkey , type="password", key="secretkey"
    )   
    bucket = st.sidebar.text_input(
        "Nom du bucket s3", value=st.session_state.bucket, key="bucket"
    )

def session_states():
    if "accesskey" not in st.session_state:
        st.session_state.accesskey = ""
    if "secretkey" not in st.session_state:
        st.session_state.secretkey = ""
    if "bucket" not in st.session_state:
        st.session_state.bucket = ""
    if "resultat" not in st.session_state:
        st.session_state.resultat = {}
    if "currentFileType" not in st.session_state:
        st.session_state.currentFile = ""
    if "keywords" not in st.session_state:
        st.session_state.keywords = []
    if "transcription" not in st.session_state:
        st.session_state.transcription = ""

session_states()
mainpage()
sidebar()
