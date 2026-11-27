# Chat with PDFs

Application web interactive developpee avec Streamlit qui permet de poser des
questions en langage naturel a partir du contenu de plusieurs fichiers PDF.

Les documents sont indexes localement dans une base vectorielle FAISS, puis les
reponses sont generees avec le modele TinyLlama execute par Hugging Face
Transformers.

## Fonctionnalites

- Import de plusieurs fichiers PDF
- Extraction du texte de chaque page
- Decoupage du contenu en segments indexables
- Recherche des passages les plus pertinents
- Questions en langage naturel sur les documents
- Conservation de l'historique de la conversation
- Interface web responsive

## Technologies

- [Streamlit](https://streamlit.io/) : interface web
- [pypdf](https://pypi.org/project/pypdf/) : extraction du texte PDF
- [LangChain](https://www.langchain.org/) : chaine RAG conversationnelle
- [FAISS](https://github.com/facebookresearch/faiss) : index vectoriel local
- [Sentence Transformers](https://sbert.net/) : generation des embeddings
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) :
  execution du modele `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- [PyTorch](https://pytorch.org/) : moteur d'inference

## Fonctionnement

1. Les fichiers PDF sont telecharges depuis la barre laterale.
2. `pypdf` extrait le texte de toutes les pages.
3. LangChain divise le texte en segments de 1 000 caracteres avec un chevauchement
   de 200 caracteres.
4. Chaque segment est transforme en vecteur avec
   `sentence-transformers/all-MiniLM-L6-v2`.
5. Les vecteurs sont stockes dans FAISS.
6. Lorsqu'une question est saisie, les passages les plus proches sont recuperes.
7. TinyLlama utilise ces passages pour construire une reponse.

## Structure du projet

```text
pdfs/
|-- app.py             # Application Streamlit et pipeline RAG
|-- htmlTemplate.py    # Styles CSS et modeles des messages
|-- requirements.txt   # Dependances Python
|-- images/
|   `-- img.png
`-- README.md
```

## Prerequis

- Python 3.10 ou plus recent
- pip
- Au moins 4 Go de RAM
- Une connexion Internet lors du premier telechargement des modeles Hugging Face
- De l'espace disque pour les modeles et le cache

Les modeles sont telecharges depuis Hugging Face lorsque l'application les utilise
pour la premiere fois.

## Installation

Cloner le projet :

```bash
git clone https://github.com/Aina-Arivelo/pdfs.git
cd pdfs
```

Creer et activer un environnement virtuel :

```bash
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

Installer les dependances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PyTorch n'est pas disponible dans votre environnement, installer la version CPU :

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Utilisation

Lancer l'application depuis la racine du projet :

```bash
streamlit run app.py
```

Streamlit ouvre ensuite l'application dans le navigateur, generalement a l'adresse
`http://localhost:8501`.

Pour utiliser l'application :

1. Charger un ou plusieurs documents PDF dans la barre laterale.
2. Cliquer sur **Process**.
3. Attendre le traitement et le chargement du modele.
4. Saisir une question dans le champ principal.
5. Lire la reponse generee avec les passages du document.

## Configuration du cache

Le fichier `app.py` definit actuellement un emplacement cache Hugging Face
specifique a l'environnement de developpement :

```python
HF_HOME = "/goinfre/sarakoto/.cache/huggingface"
TRANSFORMERS_CACHE = "/goinfre/sarakoto/.cache/huggingface/hub"
```

Sur Windows ou dans un autre environnement, modifier ces chemins pour utiliser
un dossier disponible, par exemple :

```python
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
```

## Variables d'environnement

L'application appelle `load_dotenv()`, mais aucune cle API n'est actuellement
requise pour le pipeline local utilise.

Si une cle OpenAI est ajoutee pour une implementation future, creez un fichier
`.env` a la racine :

```env
OPENAI_API_KEY=votre_cle_api
```

Ne partagez jamais ce fichier ni ne publiez vos cles dans le depot Git.

## Limitations

- La qualite d'extraction depend de la structure des PDF.
- Les PDF uniquement constitues d'images ne sont pas exploites sans OCR.
- L'inference locale peut etre lente sans GPU.
- Les premieres utilisations peuvent prendre du temps a cause du telechargement
  des modeles.
- TinyLlama peut produire des reponses imprecises ou des hallucinations.
- Le modele fonctionne mieux avec des questions directes liees au contenu.

## Ameliorations possibles

- Ajouter l'OCR pour les PDF scannes
- Persister l'index FAISS sur disque
- Afficher les passages sources utilises pour chaque reponse
- Ajouter une limite de taille ou de nombre de PDF
- Permettre le choix du modele de langage
- Ajouter l'acceleration GPU CUDA
- Ajouter des tests automatises
- Deplacer les chemins du cache dans un fichier de configuration

