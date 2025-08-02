import os #Crée le répertoire temporaire et gère les fichiers
import re#Crée les regex pour identifier les attributs dans le texte extrait
import base64 #Encode les images pour l'API
import io # Input/Output streams
import asyncio#Programmation asynchrone (non-bloquante)
from concurrent.futures import ThreadPoolExecutor #Traite plusieurs PDFs en parallèle
# Imports des modules Python standard pour la gestion des fichiers, expressions régulières, encodage base64, entrées/sorties, programmation asynchrone et exécution parallèle
from typing import List, BinaryIO, Optional, Dict, Any, Tuple # Types de données pour le typage statique
from loguru import logger
from PIL import Image# traitement d'images
import fitz  # PyMuPDF , Conversion PDF vers image 
from mistralai.client import MistralClient
from langchain.docstore.document import Document #: Classes LangChain pour les documents et la segmentation de texte
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import difflib #pour comparer des séquences (comme des chaînes de caractères, des listes, etc.) et calculer leurs différences.
import config

# Global thread pool for PDF processing
pdf_thread_pool = ThreadPoolExecutor(max_workers=2)  #traiter les PDFs en parallèle,
# --- Load Attribute Dictionary ---
ATTRIBUTE_DICTIONARY_PATH = os.getenv("ATTRIBUTE_DICTIONARY_PATH", "attribute_dictionary.json")
try:
    with open(ATTRIBUTE_DICTIONARY_PATH, "r", encoding="utf-8") as f:
        ATTRIBUTE_DICTIONARY = json.load(f)
    logger.info(f"Successfully loaded attribute dictionary with {len(ATTRIBUTE_DICTIONARY)} attributes")
    logger.debug(f"Attribute dictionary keys: {list(ATTRIBUTE_DICTIONARY.keys())}")
except Exception as e:
    logger.warning(f"Could not load attribute dictionary: {e}")
    ATTRIBUTE_DICTIONARY = {}

# --- Build Regexes for Each Attribute ---
def build_attribute_regexes(attribute_dict):
    regexes = {}
    for attr, values in attribute_dict.items():  #crée des expressions régulières pour chaque attribut du dictionnaire
        clean_values = [re.escape(v) for v in values if v] #ajoute un backslash \ devant tous les caractères spéciaux pour qu'ils soient traités comme du texte littéral dans l'expression régulière.
        if not clean_values:
            continue
        pattern = r'(' + '|'.join(clean_values) + r')'
        regexes[attr] = re.compile(pattern, re.IGNORECASE) #IGNORECASE : Flag pour ignorer la casse (majuscules/minuscules)
        # Debug for Contact Systems
        if attr == "Contact Systems":
            logger.info(f"Contact Systems regex pattern: {pattern}")
            logger.info(f"Contact Systems clean values (first 5): {clean_values[:5]}")
    
    return regexes

ATTRIBUTE_REGEXES = build_attribute_regexes(ATTRIBUTE_DICTIONARY)

# --- Tagging Utility --- #Fonction qui tag un texte avec les attributs trouvés
def tag_chunk_with_dictionary(chunk_text, attribute_regexes):
    tags = {}
    logger.info(f"Tagging chunk with {len(attribute_regexes)} attribute regexes")
    
    # Special debugging for Contact Systems
    if "Contact Systems" in attribute_regexes:
        contact_regex = attribute_regexes["Contact Systems"]
        contact_matches = contact_regex.findall(chunk_text)
        logger.info(f"Contact Systems regex matches: {contact_matches}")
        logger.info(f"Looking for 'MCP 2.8' in text: {'MCP 2.8' in chunk_text}")
    
    for attr, regex in attribute_regexes.items():    # regex = regex compilé pour cet attribut
        matches = regex.findall(chunk_text) # Trouve toutes les correspondances dans le texte
        # Convert matches to a list and handle empty lists properly for Chroma metadata
        match_list = sorted({m.strip() for m in matches}) #m.strip() : Supprime les espaces en début/fin de chaque correspondance
#{...} : Crée un set pour éliminer les doublons
#sorted(...) : Trie les résultats par ordre alphabétique
        if match_list:
            tags[attr] = ", ".join(match_list) #Stockage des résultats
            logger.info(f"Found matches for '{attr}': {match_list}")
        else:
            # If no matches, store as None (Chroma accepts None as metadata value)
            tags[attr] = None
            logger.debug(f"No matches found for '{attr}'")
    
    # Log summary of what was found
    found_attrs = [attr for attr, value in tags.items() if value is not None]#Filtrage des attributs trouvés
    logger.info(f"Generated tags for {len(found_attrs)} attributes: {found_attrs}")
    logger.debug(f"All generated tags: {tags}")
    return tags


def encode_pil_image(pil_image: Image.Image, format: str = "PNG") -> Tuple[str, str]: #Tuple contenant (chaîne base64, format d'image)
    """Encode PIL Image to Base64 string."""
    buffered = io.BytesIO() #Crée un buffer en mémoire pour stocker temporairement l'image
    # Ensure image is in RGB mode
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    save_format = format.upper()
    if save_format not in ["PNG", "JPEG"]:
        logger.warning(f"Unsupported format '{format}', defaulting to PNG.")
        save_format = "PNG"

    pil_image.save(buffered, format=save_format)
    img_byte = buffered.getvalue()# Récupération des données binaires de l'image
    return base64.b64encode(img_byte).decode('utf-8'), save_format.lower()#Convertit les bytes en base64
#On peut ensuite encoder ces bytes(binaire) en base64[ascii], les envoyer à une AP
async def process_single_pdf(file_path: str, file_basename: str, client: MistralClient, model_name: str) -> List[Document]:
    """Process a single PDF file and return its documents."""
    all_docs = []
    total_pages_processed = 0
    pdf_document = None
    
    try:
        logger.info(f"Starting processing of PDF: {file_basename}")
        logger.debug(f"File path: {file_path}")
        logger.debug(f"Using model: {model_name}")
        
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)#Comptage des pages totales
        logger.info(f"Successfully opened PDF with {total_pages} pages")
        
        # Define the prompt for Mistral Vision
        markdown_prompt = """
You are an expert document analysis assistant. Extract ALL text content from the image and format it as clean, well-structured GitHub Flavored Markdown.

Follow these formatting instructions:
1. Use appropriate Markdown heading levels based on visual hierarchy
2. Format tables using GitHub Flavored Markdown table syntax
3. Format key-value pairs using bold for keys: `**Key:** Value`
4. Represent checkboxes as `[x]` or `[ ]`
5. Preserve bulleted/numbered lists using standard Markdown syntax
6. Maintain paragraph structure and line breaks
7. Extract text labels from diagrams/images
8. Ensure all visible text is captured accurately

Output only the generated Markdown content.
"""
#Instructions précises pour Mistral Vision
# Formatage en Markdown structuré
# Extraction complète de tout le texte visible
        for page_num in range(total_pages):
            logger.info(f"\n{'='*50}") # Crée une ligne de 50 caractères "
            logger.info(f"Processing page {page_num + 1}/{total_pages} of {file_basename}")# Numéro de page (commence à 1, pas 0),Nombre total de pages
            logger.debug(f"Page dimensions: {pdf_document[page_num].rect}")
            logger.info(f"{'='*50}\n")
            
            try:
                # Get the page
                page = pdf_document[page_num]
                
                # Convert page to image with higher resolution
                logger.debug("Converting page to high-resolution image...")#Conversion en pixels haute résolution (300 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)#Création d'image PIL depuis les pixels
                logger.debug(f"Image created with dimensions: {img.size}")
                
                # Encode image to base64
                logger.debug("Encoding image to base64...")
                base64_image, image_format = encode_pil_image(img)
                logger.debug(f"Image encoded in {image_format} format")
                
                # Prepare message for Mistral Vision
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": markdown_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        ]
                    }
                ]
                
                # Call Mistral Vision API
                logger.info("Sending page to Mistral Vision API...")
                try:
                    chat_response = client.chat(
                        model=model_name,
                        messages=messages#: Structure préparée avec prompt + image
                    )
                    logger.debug("Successfully received response from Mistral Vision API")
                except Exception as api_error:
                    logger.error(f"Mistral Vision API error: {str(api_error)}")
                    raise
                
                # Get extracted text
                page_content = chat_response.choices[0].message.content
                
                if page_content:
                    # --- Tag the chunk with dictionary matches ---
                    chunk_tags = tag_chunk_with_dictionary(page_content, ATTRIBUTE_REGEXES)#Marquage avec les attributs
                    # Log the extracted content
                    logger.info("\nExtracted Content:")
                    logger.debug("-" * 40)
                    logger.debug(page_content)
                    logger.debug("-" * 40)
                    
                    # Instead of splitting into chunks, treat the whole page as one document
                    chunk_doc = Document(#pour l'intégration avec les systèmes de recherche vectorielle)
                        page_content=page_content,
                        metadata={
                            'source': file_basename,
                            'page': page_num + 1,
                            **chunk_tags  # Add all attribute tags to metadata
                        }
                    )
                    all_docs.append(chunk_doc)
                    logger.debug(f"Created document for page {page_num + 1}")# Compteur de pages traitées avec succès
                    
                    logger.success(f"Successfully processed page {page_num + 1} from {file_basename}")#confirmation  Compteur de pages traitées avec succès
                    total_pages_processed += 1
                else:
                    logger.warning(f"No content extracted from page {page_num + 1} of {file_basename}")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1} with Mistral Vision: {str(e)}", exc_info=True)#: Capture les erreurs lors du traitement d'une page spécifique
                
    except Exception as e:
        logger.error(f"Error processing {file_basename}: {str(e)}", exc_info=True)#Erreurs au niveau du fichier entier,Problèmes d'ouverture du PDF
    finally:
        # Close the PDF document if it was opened
        if pdf_document is not None:
            try:
                pdf_document.close()
                logger.debug(f"Closed PDF document: {file_basename}")
            except Exception as e:
                logger.warning(f"Error closing PDF document {file_basename}: {str(e)}")
    
    if not all_docs:
        logger.error(f"No text could be extracted from {file_basename}")
    else:
        logger.info(f"\nProcessing Summary for {file_basename}:")
        logger.info(f"Total pages processed: {total_pages_processed}")
        logger.info(f"Total chunks created: {len(all_docs)}")
        logger.debug(f"Average chunk size: {sum(len(doc.page_content) for doc in all_docs) / len(all_docs):.2f} characters")
    
    return all_docs#liste de tous les documents traités

async def process_uploaded_pdfs(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> List[Document]:#fonction asynchrone qui traite plusieurs fichiers PDF uploadés en parallèle
   #Répertoire temporaire pour sauvegarder les fichiers (défaut: "temp_pdf")
    """Process uploaded PDFs using Mistral Vision for better text extraction."""
    all_docs: List[Document] = []# Liste vide qui va contenir tous les documents extraits
    saved_file_paths: List[str] = []#Liste vide qui va contenir les chemins des fichiers temporaires
    
    logger.info(f"Starting batch processing of {len(uploaded_files)} PDF files")#Affiche le nombre de fichiers à traiter
    logger.debug(f"Temporary directory: {temp_dir}")#Affiche le répertoire temporaire utilisé
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)#Crée le répertoire s'il n'existe pas
    logger.debug("Ensured temporary directory exists")
    
    # Initialize Mistral client
    try:
        client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        model_name = config.VISION_MODEL_NAME
        logger.info(f"Initialized Mistral Vision client with model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Mistral client: {str(e)}", exc_info=True)
        return []
    
    try:
        # Save all files first
        for uploaded_file in uploaded_files:#Chaque fichier uploadé est sauvegardé temporairement sur le disque
            file_basename = uploaded_file.name
            file_path = os.path.join(temp_dir, file_basename)
            saved_file_paths.append(file_path)
            logger.debug(f"Saving uploaded file: {file_basename}")
            # Save uploaded file temporarily
            with open(file_path, "wb") as f:#Sauvegarde en mode binaire ("wb")
                f.write(uploaded_file.getvalue())
            logger.debug(f"Successfully saved file: {file_basename}")
        
        # Process PDFs in parallel using ThreadPoolExecutor
        logger.info(f"Starting parallel processing of {len(saved_file_paths)} files")
        with ThreadPoolExecutor(max_workers=min(len(saved_file_paths), 4)) as executor:
            # Create tasks for each PDF
            loop = asyncio.get_event_loop()# Récupère la boucle d'événements asynchrone
            tasks: List[asyncio.Task] = []#Liste vide pour stocker les tâches
            for file_path in saved_file_paths:
                file_basename = os.path.basename(file_path)#Extrait juste le nom du fichier du chemin complet
                logger.debug(f"Creating task for file: {file_basename}")
                # Create a task that runs in the thread pool
                task = loop.run_in_executor(
                    executor,
                    lambda p, b: asyncio.run(process_single_pdf(p, b, client, model_name)), # Fonction synchrone qui lance une async
                    file_path,
                    file_basename
                )
                tasks.append(task)# Ajoute la tâche à la liste
            
            # Wait for all PDFs to be processed
            logger.info("Waiting for all PDF processing tasks to complete...")
            results = await asyncio.gather(*tasks)#Attend que toutes les tâches se terminent
            logger.info("All PDF processing tasks completed")
            
            # Combine all results in a deterministic order (by file name)
            results = [docs for docs in results if docs]
            results.sort(key=lambda docs: docs[0].metadata['source'] if docs and hasattr(docs[0], 'metadata') and 'source' in docs[0].metadata else '')#Trie les résultats par nom de fichier source
            all_docs = []
            for docs in results:
                all_docs.extend(docs)#Ajoute tous les documents de ce résultat
                logger.debug(f"Added {len(docs)} documents from a processed file")#Affiche le nombre de documents ajoutés
            
    except Exception as e:
        logger.error(f"Error during batch PDF processing: {str(e)}", exc_info=True)
    finally:
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for path in saved_file_paths:
            try:
                os.remove(path)#Supprime le fichier temporaire
                logger.debug(f"Removed temporary file: {path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {path}: {str(e)}")
    
    if not all_docs:
        logger.error("No text could be extracted from any provided PDF files.")# Si aucun document n'a été extrait
    else:
        logger.info("\nFinal Processing Summary:")
        logger.info(f"Total documents processed: {len(saved_file_paths)}")
        logger.info(f"Total chunks created: {len(all_docs)}")
        logger.debug(f"Average chunks per document: {len(all_docs) / len(saved_file_paths):.2f}")
    
    return all_docs

def process_pdfs_in_background(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> asyncio.Task[List[Document]]:
    """Start PDF processing in the background and return a task that can be awaited later."""
    return asyncio.create_task(process_uploaded_pdfs(uploaded_files, temp_dir))


