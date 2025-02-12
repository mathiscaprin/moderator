import os
import boto3
from dotenv import load_dotenv
import time
import cv2
import nltk
import urllib.request
import json

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def check_filetype(filename):
    """
    Détermine le type de fichier en fonction de son extension.

    Cette fonction prend un nom de fichier en entrée, extrait son extension et détermine
    le type de fichier (par exemple, image, vidéo). Si l'extension du fichier est reconnue comme un format
    d'image courant (jpg, png, tiff, svg) ou un format de vidéo courant (mp4, avi, mkv), elle attribue
    le type correspondant. Sinon, le type de fichier est défini sur None.

    Paramètres :
    - filename (str) : Le chemin vers le fichier incluant le nom de fichier.

    Retourne :
    - str ou None : Le type de fichier déterminé ('image', 'vidéo') ou None si le type de fichier
      n'est pas reconnu.

    Exemple :
    >>> check_filetype("/chemin/vers/image.jpg")
    'image'
    >>> check_filetype("/chemin/vers/video.mp4")
    'vidéo'
    >>> check_filetype("/chemin/vers/fichierinconnu.xyz")
    None
    """

    # Extrait le nom de base du fichier à partir du chemin de fichier fourni.
    file_basename = os.path.basename(filename)

    # Sépare le nom de base sur le point et prend la dernière partie comme extension.
    extension = file_basename.split(".")[-1]

    # Détermine le type de fichier en fonction de l'extension.
    if extension in ["jpg", "png", "tiff", "svg"]:
        filetype = "image"
    elif extension in ["mp4", "avi", "mkv"]:
        filetype = "vidéo"
    else:
        filetype = None

    # Enregistre le type de fichier détecté.
    print(f"[INFO] : Le fichier {file_basename} est de type : {filetype}")
    
    return filetype


def extract_frame_video(video_path, frame_id):
    """
    Extrait une image spécifique d'une vidéo.

    Cette fonction utilise OpenCV pour ouvrir une vidéo à partir du chemin spécifié et extrait une image
    particulière en fonction de son ID. L'ID de l'image correspond à l'ordre de l'image dans la vidéo, en commençant
    par 0 pour la première image. Si l'extraction réussit, l'image est retournée sous forme d'un tableau Numpy.

    Paramètres :
    - video_path (str) : Le chemin vers le fichier vidéo d'où extraire l'image.
    - frame_id (int) : L'identifiant (ID) de l'image à extraire.

    Retourne :
    - ndarray ou None : L'image extraite (un tableau Numpy) si l'extraction est réussie,
      sinon `None`.

    Exemple :
    >>> image = extract_frame_video("/chemin/vers/video.mp4", 150)
    >>> type(image)
    <class 'numpy.ndarray'>
    """

    # Ouvre la vidéo à partir du chemin fourni.
    video = cv2.VideoCapture(video_path)

    # Positionne le lecteur vidéo sur l'image spécifiée par frame_id.
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    # Lit l'image actuelle.
    ret, image = video.read()

    # Si la lecture réussit (ret est True), retourne l'image.
    # Sinon, retourne None.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if ret else None


def get_aws_session():
    """
    Crée et retourne une session AWS.

    Cette fonction charge les variables d'environnement depuis un fichier .env situé dans le répertoire
    courant ou les parents de celui-ci, récupère les clés d'accès AWS (`ACCESS_KEY` et `SECRET_KEY`),
    et initialise une session AWS avec ces identifiants ainsi qu'avec une région spécifiée (dans cet exemple,
    'us-east-1'). Elle est particulièrement utile pour configurer une session AWS de manière sécurisée sans
    hardcoder les clés d'accès dans le code.

    Retourne :
    - Session : Un objet session de boto3 configuré avec les clés d'accès et la région AWS.

    Exemple d'utilisation :
    >>> session_aws = get_aws_session()
    >>> type(session_aws)
    <class 'boto3.session.Session'>
    """

    # Charge les variables d'environnement depuis .env.
    load_dotenv()

    # Crée une session AWS avec les clés d'accès et la région définies dans les variables d'environnement.
    aws_session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),        # Récupère l'ID de clé d'accès depuis les variables d'environnement.
        aws_secret_access_key=os.getenv("SECRET_KEY"),    # Récupère la clé d'accès secrète depuis les variables d'environnement.
        region_name="us-east-1"                           # Spécifie la région AWS à utiliser.
    )
    # Retourne l'objet session créé.
    return aws_session

def moderate_image(image_path, aws_service):
    """
    Détecte du contenu nécessitant une modération dans une image en utilisant un service AWS spécifié.

    Cette fonction ouvre une image depuis un chemin donné, puis utilise le service AWS (comme Amazon Rekognition)
    pour détecter les contenus potentiellement inappropriés ou sensibles (comme la nudité, la violence, etc.).
    Elle collecte et retourne une liste des étiquettes de modération identifiées pour cette image.

    Paramètres :
    - image_path (str) : Le chemin vers l'image à analyser.
    - aws_service (object) : Un objet de service AWS configuré, capable de réaliser des opérations de détection
      de contenu nécessitant une modération (par exemple, un client Amazon Rekognition).

    Retourne :
    - list[str] : Une liste des noms des étiquettes de modération détectées pour l'image.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> moderate_image("/chemin/vers/image.jpg", aws_rekognition_client)
    ['Nudity', 'Explicit Violence']
    """

    # Ouvre l'image puis utilise detect_moderation_labels de Amazon Rekognition qui permet de détecter les contenus inappropriés
    with open(image_path, 'rb') as image:
        response = aws_service.detect_moderation_labels(
        Image={
            'Bytes': image.read()
        })
    return [dic["Name"] for dic in response["ModerationLabels"]][:10]

# Crée un client AWS Transcribe.
def getStatus(aws_service, job_name, bucket_name, filename):
    aws_service.start_transcription_job(
    TranscriptionJobName=job_name,
    LanguageCode='fr-FR',
    MediaFormat='mp4',
    Media={
        'MediaFileUri': f's3://{bucket_name}/{filename}'
    },
    OutputBucketName=bucket_name
    )
    while True:
        status = aws_service.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)
    return status

# Convertit le statut du travail de transcription en texte.
def statusToText(status):
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        response =  urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        data = json.loads(response.read())
        texte = data['results']['transcripts'][0]['transcript']
        return texte
    else:
        return None

def get_text_from_speech(filename, aws_service, job_name, bucket_name):
    """
    Convertit de la parole en texte en utilisant AWS Transcribe.

    Cette fonction téléverse un fichier audio spécifié dans un seau S3, démarre un travail de transcription avec AWS Transcribe,
    attend que le travail soit terminé, et récupère le texte transcrit.

    Paramètres :
    - filename (str) : Chemin local vers le fichier audio à transcrire.
    - aws_service (object) : Client AWS Transcribe configuré.
    - job_name (str) : Nom unique pour le travail de transcription.
    - bucket_name (str) : Nom du seau S3 où le fichier audio est stocké.

    Retourne :
    - str : Le texte transcrit du fichier audio.

    Prérequis :
    - Le fichier audio doit déjà être téléversé dans le seau S3 spécifié.
    """

    status = getStatus(aws_service, job_name, bucket_name, filename)
    Retour = statusToText(status)
    return Retour


def clean_text(raw_text):
    """
    Nettoie un texte en retirant les mots vides et en normalisant les mots en minuscules.

    Cette fonction prend un texte brut en entrée, tokenise le texte pour séparer les mots,
    convertit les mots en minuscules, et retire les mots vides (stop words) en français. Les mots vides
    supplémentaires peuvent être ajoutés à la liste. Le texte résultant contient uniquement les mots significatifs
    en minuscules.

    Paramètres :
    - raw_text (str) : Le texte brut à nettoyer.

    Retourne :
    - str : Le texte nettoyé, sans mots vides et en minuscules.

    Exemple d'utilisation :
    >>> texte_brut = "Ceci est un exemple de texte à nettoyer."
    >>> clean_text(texte_brut)
    'exemple texte nettoyer'
    """

    # Tokenise le texte en mots.
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw_text)

    # Convertit les mots en minuscules et retire les mots vides.
    words = [word.lower() for word in words]
    stop_words = set(stopwords.words('french'))
    words = [word for word in words if word not in stop_words]

    # Retourne le texte nettoyé en joignant les mots ensemble.
    return ' '.join(words)

def extract_keyphrases(text, aws_service):
    """
    Extrait les expressions clés d'un texte et retourne les 10 expressions les plus pertinentes comme hashtags.

    Cette fonction utilise un service AWS, tel que Amazon Comprehend, pour détecter les expressions clés dans
    un texte donné. Elle trie ces expressions par leur score de pertinence fourni par AWS et retourne les 10
    expressions clés les plus pertinentes sous forme de hashtags.

    Paramètres :
    - text (str) : Le texte duquel extraire les expressions clés.
    - aws_service (object) : Un objet de service AWS configuré pour détecter les expressions clés.

    Retourne :
    - list[str] : Une liste des 10 hashtags les plus pertinents basés sur les expressions clés du texte.

    Exemple d'utilisation :
    >>> aws_comprehend_client = boto3.client('comprehend', region_name='us-east-1')
    >>> extract_keyphrases("Ceci est un exemple de texte.", aws_comprehend_client)
    ['#exemple', '#texte']
    """

    # La service AWS va détecter les expressions clés.
    response = aws_service.detect_key_phrases(Text=text, LanguageCode='fr')
    
    key_phrases = response['KeyPhrases']
    key_phrases_sorted = sorted(key_phrases, key=lambda x: x['Score'], reverse=True)
    top_key_phrases = key_phrases_sorted[:10]
    top_key_phrases = top_key_phrases[0]['Text']
    text = top_key_phrases.split(" ")
    hashtags = []
    for element in text:
        hashtag = '#' + element.lower()
        hashtags.append(hashtag)

    return hashtags

def detect_objects(image_path, aws_service):
    """
    Détecte les objets dans une image en utilisant Amazon Rekognition.

    Cette fonction ouvre une image depuis un chemin spécifié, utilise un service AWS (Amazon Rekognition) pour
    détecter les objets présents dans l'image avec une confiance minimale de 50%, et retourne les noms des 10
    objets les plus pertinents détectés.

    Paramètres :
    - image_path (str) : Le chemin vers l'image à analyser.
    - aws_service (object) : Un client AWS Rekognition configuré.

    Retourne :
    - list[str] : Une liste contenant les noms des 10 premiers objets détectés dans l'image.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> detect_objects("/chemin/vers/image.jpg", aws_rekognition_client)
    ['Voiture', 'Arbre', 'Personne']
    """

    # Detect_labels de amazon rekognition permet de détecter les objets dans une image
    with open(image_path, 'rb') as image:
        reponse = aws_service.detect_labels(
            Image={
                'Bytes': image.read()
            },
            MaxLabels=100,
            MinConfidence=50
        )

    # Renvoie les 10 premiers
    return [dic["Name"] for dic in reponse["Labels"]][:10]

def detect_celebrities(image_path, aws_service):
    """
    Identifie les célébrités dans une image en utilisant le service Amazon Rekognition.

    Cette fonction ouvre une image depuis un chemin donné et utilise le service AWS Rekognition pour reconnaître les
    célébrités présentes dans l'image. Elle retourne une liste contenant les noms des célébrités identifiées, limitée
    aux 10 premiers résultats pour simplifier l'output.

    Paramètres :
    - image_path (str) : Le chemin vers l'image dans laquelle détecter les célébrités.
    - aws_service (object) : Un client AWS Rekognition configuré.

    Retourne :
    - list[str] : Une liste des noms des célébrités identifiées dans l'image, jusqu'à un maximum de 10.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> detect_celebrities("/chemin/vers/limage.jpg", aws_rekognition_client)
    ['Leonardo DiCaprio', 'Kate Winslet']
    """
    # Recognize_celebrities de amazon rekognition permet de détecter les célébrités dans une image
    with open(image_path, 'rb') as image:
        response = aws_service.recognize_celebrities(
            Image={
                'Bytes': image.read()
            }
        )
    # Renvoie les noms des célébrités détectées
    return [celeb['Name'] for celeb in response['CelebrityFaces']]

def detect_emotions(image_path, aws_service):
    """
    Détecte les émotions sur les visages présents dans une image en utilisant Amazon Rekognition.
    
    Cette fonction analyse une image pour détecter les visages et leurs émotions associées.
    Pour chaque visage, elle retourne les émotions détectées avec leur niveau de confiance.
    
    Paramètres :
    - image_path (str) : Chemin vers l'image à analyser
    - aws_service (boto3.client) : Client AWS Rekognition configuré
    
    Retourne :
    - list[dict] : Liste des visages détectés avec leurs émotions
                  Format: [
                      {
                          'BoundingBox': dict,
                          'Emotions': [
                              {
                                  'Type': str,  # HAPPY, SAD, ANGRY, CONFUSED, etc.
                                  'Confidence': float
                              },
                              ...
                          ],
                          'AgeRange': {'Low': int, 'High': int},
                          'Gender': {'Value': str, 'Confidence': float}
                      },
                      ...
                  ]
    
    Exemple :
    >>> rekognition = boto3.client('rekognition')
    >>> emotions = detect_emotions("./photo.jpg", rekognition)
    >>> for face in emotions:
    ...     print(f"Émotions détectées : {face['Emotions']}")
    """

    # Detect_faces de amazon rekognition permet de détecter les visages et leurs émotions
    with open(image_path, 'rb') as image:
        reponse = aws_service.detect_faces(
            Image={
                'Bytes': image.read()
            },
            Attributes=['ALL']
        )
    # La liste contenant les dictionnaires associées aux visages détectés
    liste_visages = []
    # Pour chaque visage détecté, on récupère les informations sur les émotions, l'âge et le genre pour les mettres dans un dictionnaire
    #   BoudingBox : coordonnées du visage (Left, Top, Width, Height)
    #   Emotions : liste des émotions détectées (Type, Confidence)
    #   AgeRange : tranche d'âge (min, max)
    #   Gender : genre (Valeur, Confiance)
    dico_visage = {}
    for face in reponse['FaceDetails']:
        dico_visage['BoundingBox'] = face['BoundingBox']
        dico_visage['Emotions'] = face['Emotions']
        dico_visage['AgeRange'] = face['AgeRange']
        dico_visage['Gender'] = face['Gender']
        liste_visages.append(dico_visage)
        
    return liste_visages

def summarize_emotions(faces_info):
    """
    Résume les émotions détectées sur tous les visages d'une image.
    
    Cette fonction agrège les émotions de tous les visages et calcule les émotions
    dominantes dans l'image.
    
    Paramètres :
    - faces_info (list[dict]) : Liste des informations des visages détectés
    
    Retourne :
    - dict : Résumé des émotions dominantes et statistiques
    
    Exemple :
    >>> emotions = detect_emotions("./group_photo.jpg", rekognition)
    >>> summary = summarize_emotions(emotions)
    >>> print(f"Émotion dominante : {summary['dominant_emotion']}")
    """
    # dictionnaire de retour contenant toutes les données
    dico = {}
    # dictionnaire contenant les émotions et leur moyenne de confiance
    dico_emotion = {}
    # dictionnaire contenant la confiance moyenne pour chaque émotion
    dico_emotion["confidence"] = {}
    # dictionnaire contenant le nombre de visages pour chaque émotion
    dico_emotion["count"] = {}
    # dictionnaire contenant les statistiques sur l'âge
    dico_age = {}
    # dictionnaire contenant le nombre de visages par genre
    dico_genre = {}
    # liste contenant les âges min et max
    age_min = []
    age_max = []
    #AgeRange = {'Low': X, 'High': Y}



    # Remplit les diconnaires dico_emotion, dico_genre et les listes age_min et age_max
    for face in faces_info:
        for emotion in face["Emotions"]:
            if emotion["Type"] not in dico_emotion["confidence"]:
                dico_emotion["confidence"][emotion["Type"]] = emotion["Confidence"]
                dico_emotion["count"][emotion["Type"]] = 1
            else:
                dico_emotion["confidence"][emotion["Type"]] = dico_emotion["confidence"][emotion["Type"]] + emotion["Confidence"]
                dico_emotion["count"][emotion["Type"]] = dico_emotion["count"][emotion["Type"]] + 1

        if face["Gender"]["Value"] not in dico_genre:
            dico_genre[face["Gender"]["Value"]] = 1
        else :
            dico_genre[face["Gender"]["Value"]] = dico_genre[face["Gender"]["Value"]] + 1
            
        age_min.append(face['AgeRange']['Low'])
        age_max.append(face['AgeRange']['High'])
    else:
        dico_age['min'] = None
        dico_age['max'] = None
        dico_age['mean'] = None
            
    # Remplit dico_age avec les statistiques sur l'âge
    
    if age_min and age_max:
        dico_age['min'] = min(age_min)
        dico_age['max'] = max(age_max)
        dico_age['mean'] = sum(age_min + age_max) / (2 * len(faces_info))


    # Convertit la somme des confiances en moyenne
    for i in dico_emotion["confidence"]:
        dico_emotion["confidence"][i] = dico_emotion["confidence"][i] / dico_emotion["count"][i]

    # Remplit le dictionnaire principal
    inverse = [(value, key) for key, value in dico_emotion["confidence"].items()]
    dico["emotion_dominante"] = max(inverse)[1] if inverse else ""
    dico["nombre_visages"] = len(faces_info)
    dico["statistiques"] = dico_emotion
    dico["statistiques_age"] = dico_age
    dico["statistiques_genre"] = dico_genre

    return dico


def process_media(media_file, rekognition, transcribe, comprehend, bucket_name):
    """
    Traite un fichier multimédia (image ou vidéo) pour modérer le contenu, détecter des objets/célébrités,
    transcrire le discours et extraire des expressions clés.

    Selon le type de fichier, cette fonction applique une chaîne de traitement appropriée en utilisant différents
    services AWS. Pour les images, elle modère le contenu, détecte des objets, émotions faciales et des célébrités. Pour les vidéos,
    elle extrait une image, modère le contenu, téléverse la vidéo sur S3, transcrit le discours en texte, nettoie le texte,
    et extrait des expressions clés.

    Paramètres :
    - media_file (str) : Chemin vers le fichier multimédia à traiter.
    - rekognition (object) : Client AWS Rekognition configuré.
    - transcribe (object) : Client AWS Transcribe configuré.
    - comprehend (object) : Client AWS Comprehend configuré.
    - bucket_name (str) : Nom du seau S3 pour stocker les fichiers vidéo.

    Retourne :
    - dict : Dictionnaire contenant des hashtags pour les images ou des sous-titres et hashtags pour les vidéos.
    """

    # Si le fichier est une image:
    #  - Modère le contenu
    #  - Détecte les objets, émotions et célébrités
    #  - Extrait des expressions clés
    # - Retourne les hashtags
    if check_filetype(media_file) == "image":
        keywords = moderate_image(media_file, rekognition)
        if keywords == []:
            objects = detect_objects(media_file, rekognition)
            emotions = summarize_emotions(detect_emotions(media_file, rekognition))
            celebrities = detect_celebrities(media_file, rekognition)
            print(" ".join(objects + celebrities) + " " + emotions["emotion_dominante"])
            keyphrases = extract_keyphrases(" ".join(objects + celebrities) + emotions["emotion_dominante"], comprehend)
            return {"hashtags" : keyphrases, "moderated" : True}
        else:
            return {"hashtags" : keywords, "moderated" : False}
    elif check_filetype(media_file) == "vidéo":
    # si le fichier est une vidéo:
    #  - Extrait la première image
    #  - Modère le contenu
    #  - Transcrit le texte
    #  - Nettoie le texte
    #  - Extrait des expressions clés
    #  - Retourne les sous-titres et les hashtags
        image = extract_frame_video(media_file, 1)
        image_file = f"{os.path.basename(media_file).split('.')[0]}.png"
        cv2.imwrite(image_file, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keywords = moderate_image(image_file, rekognition)
        if keywords == []:
            job_name = f"transcription-{int(time.time())}"
            text = get_text_from_speech(os.path.basename(media_file), transcribe, job_name, bucket_name)
            cleaned_text = clean_text(text)
            keyphrases = extract_keyphrases(cleaned_text, comprehend)
            return {"subtitles": text, "hashtags": keyphrases, "moderated": True}
        else:
            return {"hashtags" : keywords, "moderated" : False}
    else:
        return {"hashtags" : [], "moderated" : False}


def main():
    print("ok")
    TEST_VIDEO_FILE = "../assets/tuto_jeux-video.mp4"
    TEST_IMAGE_FILE = "./assets/selfie_with_johnny-depp.png"
    BUCKET_NAME = 'mcaprintp03'
    aws_session = get_aws_session()
    rekognition = aws_session.client('rekognition')
    transcribe = aws_session.client('transcribe')
    comprehend = aws_session.client('comprehend')
    s3 = aws_session.client('s3')

    print(extract_keyphrases("Animal Canine Dog Husky Mammal Pet Person Sitting Adult Male HAPPY", comprehend))
    s3.create_bucket(Bucket=BUCKET_NAME)

    s3.upload_file(TEST_VIDEO_FILE, BUCKET_NAME, os.path.basename(TEST_VIDEO_FILE))
    test = process_media(TEST_VIDEO_FILE, rekognition, transcribe, comprehend, BUCKET_NAME)
    print(test)
    test = process_media(TEST_IMAGE_FILE, rekognition, transcribe, comprehend, BUCKET_NAME)
    print(test)
