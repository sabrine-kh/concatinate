import json
from typing import List, Dict, Optional
from loguru import logger #bibliothèque pour afficher des messages  
from langchain.vectorstores.base import VectorStoreRetriever  #retrouver des parties similaires à une requête
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate  #Permet de créer des modèles de prompt (questions ou instructions) 
from langchain_core.runnables import RunnablePassthrough, RunnableParallel #Permettent de composer des chaînes d’opérations
from langchain_core.output_parsers import StrOutputParser #ransforme la sortie du LLM en chaîne de caractères exploitable.
import config 
import asyncio #Permet d’exécuter des fonctions asynchrones (par exemple, pour le web scraping sans bloquer le programme).
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
# Outil pour naviguer et extraire des données de pages web de façon asynchrone.
# Paramètres pour configurer le comportement du crawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
#Stratégie pour extraire des éléments HTML via des sélecteurs CSS et les structurer en JSON.
from bs4 import BeautifulSoup #Bibliothèque pour parser et manipuler du HTML 
import hashlib# Pour créer des empreintes (hash) de textes (pour chaque partie de texte we created an identity)
# Pour identifier de manière unique chaque "chunk" (morceau de document) et éviter les doublons.
import datetime #Pour manipuler les dates et heures 
import os #Pour interagir avec le système de fichiers (chemins, ouverture de fichiers,)
# Permet de trouver automatiquement les fichiers de configuration, peu importe où le programme est exécuté
from typing import Dict, Any, Optional

RETRIEVED_CHUNKS_LOG = os.path.join(os.path.dirname(__file__), 'retrieved_chunks_log.jsonl')

def load_attribute_dictionary():
    """Load the attribute dictionary from JSON file."""
    try:
        dict_path = os.path.join(os.path.dirname(__file__), 'attribute_dictionary.json')
        with open(dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load attribute dictionary: {e}")
        return {}

ATTRIBUTE_DICT = load_attribute_dictionary() #Variable globale qui contient le dictionnaire d'attributs chargé

def _hash_chunk(chunk):  # "_" fonction "privée
    # Hash chunk content and metadata for reproducibility
    m = hashlib.sha256()
    m.update(chunk.page_content.encode('utf-8'))#Crée un objet hash SHA-256
    m.update(json.dumps(chunk.metadata, sort_keys=True).encode('utf-8'))#Ajoute le contenu du chunk au hash (converti en bytes UTF-8)
    return m.hexdigest()#Retourne l'identifiant unique en format hexadécimal

def _log_retrieved_chunks(attribute_key, query, chunks):  #Fonction privée qui enregistre les chunks récupérés
    # Store a record of retrieved chunks for this attribute and query
    record = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'attribute_key': attribute_key,
        'query': query,
        'chunk_hashes': [_hash_chunk(chunk) for chunk in chunks],
        'chunk_metadata': [chunk.metadata for chunk in chunks],
        'num_chunks': len(chunks)
    }
    with open(RETRIEVED_CHUNKS_LOG, 'a', encoding='utf-8') as f:  # Écriture dans le fichier de log
        f.write(json.dumps(record) + '\n')

# --- Initialize LLM ---
@logger.catch(reraise=True) # Keep catch for unexpected errors during init
def initialize_llm():
    """Initializes and returns the Groq LLM client. No internal logging."""
    if not config.GROQ_API_KEY:
        # logger.error("GROQ_API_KEY not found.") # Remove internal logging
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    try:
        llm = ChatGroq(
            temperature=0.0,    #très déterministe (même réponse pour même question)
            top_p=1.0,                 # leave at 1 for greedy decoding( pas de filtrage, toutes les options sont possibles.)
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL_NAME,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            frequency_penalty=0.0,   #Pas de pénalité pour la répétition de mots
            presence_penalty=0.0,   #Pas de pénalité pour l'apparition de nouveaux sujets
            seed=42                    # optional but guarantees replayability ,( obtiens toujours la même réponse si tu relances le code.)
        )
        # logger.info(f"Groq LLM initialized with model: {config.LLM_MODEL_NAME}") # Remove internal logging
        return llm
    except Exception as e:
        # logger.error(f"Failed to initialize Groq LLM: {e}") # Remove internal logging
        # Re-raise a more specific error if needed, or let @logger.catch handle it
        raise ConnectionError(f"Could not initialize Groq LLM: {e}")

# --- Option 1: Using LangChain's Groq Integration (Recommended) ---

def format_docs(docs: List[Document]) -> str:   # Cette fonction prend une liste de documents ( chunks pertinentes) et les formate en une seule chaîne de texte pour input de llm . /Le LLM sait d'où vient chaque information
    """Formats retrieved documents into a string for the prompt."""
    # Keep detailed formatting as it might help LLM locate info in PDFs
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        context_parts.append(  ,#formatage de chaîne)
            f"Document {i+1} from '{source}' (Page {page}):\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

def create_enhanced_search_queries(attribute_key: str, base_query: str) -> list: #Crée des requêtes de recherche améliorées en utilisant les valeurs du dictionnaire et les synonymes.
    """Create enhanced search queries using dictionary values and synonyms."""
    queries = [base_query]  # Always include the original query
    
    # Get dictionary values for this attribute
    dict_values = ATTRIBUTE_DICT.get(attribute_key, [])
    # Create attribute-specific search terms
    attribute_terms = {
        "Material Filling": ["filling", "additive", "filler", "glass fiber", "GF", "GB", "MF", "talcum"],
        "Material Name": ["material", "polymer", "PA", "PBT", "PP", "PET", "PC", "silicone", "rubber"],
        "Pull-To-Seat": ["pull to seat", "pull-back", "tug-lock", "terminal insertion", "seating"],
        "Gender": ["gender", "male", "female", "plug", "receptacle", "socket", "header"],
        "Height [MM]": ["height", "Y-axis", "total height", "thickness"],
        "Length [MM]": ["length", "Z-axis", "depth", "insertion depth"],
        "Width [MM]": ["width", "X-axis", "diameter"],
        "Number Of Cavities": ["cavity", "position", "way", "pin count", "terminal count"],
        "Number Of Rows": ["row", "grid", "arrangement", "layout"],
        "Mechanical Coding": ["coding", "keying", "polarization", "mechanical key"],
        "Colour": ["color", "colour", "black", "white", "red", "blue", "yellow"],
        "Colour Coding": ["color coding", "colour coding", "coded components"],
        "Working Temperature": ["temperature", "thermal", "operating temp", "min temp", "max temp"],
        "Housing Seal": ["seal", "sealing", "radial seal", "interface seal", "ring seal"],
        "Wire Seal": ["wire seal", "individual seal", "mat seal", "gel seal"],
        "Sealing": ["sealing", "waterproof", "dustproof", "IP rating"],
        "Sealing Class": ["IP", "ingress protection", "IPx0", "IPx4", "IPx6", "IPx7"],
        "Contact Systems": ["contact system", "terminal system", "MQS", "MCP", "TAB", "MLK"],
        "Terminal Position Assurance": ["TPA", "terminal position assurance", "anti-backout"],
        "Connector Position Assurance": ["CPA", "connector position assurance", "secondary lock"],
        "Name Of Closed Cavities": ["closed cavity", "blocked position", "plugged cavity"],
        "Pre-assembled": ["pre-assembled", "assembly", "disassembly", "delivered as"],
        "Type Of Connector": ["connector type", "standard", "antenna", "relay holder", "bulb holder"],
        "Set/Kit": ["set", "kit", "accessories", "components"],
        "HV Qualified": ["HV", "high voltage", "voltage", "qualified", "certified"]
    }## Le système essaiera chaque requête pour trouver le maximum d'informations pertinentes
    
    # Add attribute-specific terms
    if attribute_key in attribute_terms:# ajouter les termes spécifiques à l'attribut.
        queries.extend(attribute_terms[attribute_key]) # ajoute tous les éléments de la liste à queries
    
    # Add dictionary values as search terms (for better matching)
    for value in dict_values[:10]:  # Limit to first 10 values to avoid too many queries
        if isinstance(value, str) and len(value) > 1: #vérifie que c'est une chaîne de caractères
            queries.append(value)
    
    # Create combined queries using base query + dictionary values
    if dict_values:
        for value in dict_values[:5]:  # Use top 5 dictionary values
            if isinstance(value, str) and len(value) > 1:
                combined_query = f"{base_query} {value}"
                queries.append(combined_query)
    
    # Remove duplicates and limit total queries
    unique_queries = list(dict.fromkeys(queries))[:20]  # Incr eased to 20 queries
    return unique_queries

def retrieve_and_log_chunks(retriever, query: str, attribute_key: str):
    """Retrieves chunks from the retriever using enhanced search queries and logs them for debugging."""
    logger.info(f"🔍 RETRIEVING CHUNKS for attribute '{attribute_key}' with base query: '{query}'")
    
    # Create enhanced search queries
    enhanced_queries = create_enhanced_search_queries(attribute_key, query)
    logger.info(f"📋 Using {len(enhanced_queries)} enhanced search queries: {enhanced_queries[:5]}...")
    
    all_chunks = []
    seen_chunks = set()  # Track unique chunks by content hash
    
    try:
        # Try each enhanced query
        for i, search_query in enumerate(enhanced_queries):
            logger.debug(f"🔍 Search {i+1}/{len(enhanced_queries)}: '{search_query}'")
            
            try:
                chunks = retriever.invoke(search_query)
                
                if chunks:
                    # Add unique chunks only
                    for chunk in chunks:
                        chunk_hash = _hash_chunk(chunk)
                        if chunk_hash not in seen_chunks:
                            seen_chunks.add(chunk_hash)
                            all_chunks.append(chunk)
                            logger.debug(f"  ✅ Added unique chunk from query '{search_query}'")
                
            except Exception as e:
                logger.warning(f"❌ Query '{search_query}' failed: {e}")
                continue
        
        # Limit total chunks to avoid overwhelming the LLM
        max_chunks = 10
        if len(all_chunks) > max_chunks:
            logger.info(f"📊 Limiting chunks from {len(all_chunks)} to {max_chunks}")
            all_chunks = all_chunks[:max_chunks]
        
        if not all_chunks:
            logger.warning(f"❌ No chunks retrieved for attribute '{attribute_key}' after trying {len(enhanced_queries)} queries")
            _log_retrieved_chunks(attribute_key, query, [])
            return []
        
        logger.info(f"✅ Retrieved {len(all_chunks)} unique chunks for attribute '{attribute_key}' from {len(enhanced_queries)} queries:")
        
        for i, chunk in enumerate(all_chunks):
            source = chunk.metadata.get('source', 'Unknown')
            page = chunk.metadata.get('page', 'N/A')
            start_index = chunk.metadata.get('start_index', 'N/A')
            logger.info(f"  📄 Chunk {i+1}: Source='{source}', Page={page}, StartIndex={start_index}")
            logger.info(f"     Content: {chunk.page_content[:200]}{'...' if len(chunk.page_content) > 200 else ''}")
        
        _log_retrieved_chunks(attribute_key, query, all_chunks)
        
        return all_chunks
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced retrieval for attribute '{attribute_key}': {e}")
        _log_retrieved_chunks(attribute_key, query, [])
        return []

# Configure websites to scrape, in order of preference.
# We now target the main table/container holding the product features.
WEBSITE_CONFIGS = [
    {
        "name": "TE Connectivity",
        "base_url_template": "https://www.te.com/en/product-{part_number}.html",
        # JS to click the features expander button if it's not already expanded
        "pre_extraction_js": (
            "(async () => {"
            "    const expandButtonSelector = '#pdp-features-expander-btn';"
            "    const featuresPanelSelector = '#pdp-features-tabpanel';"
            "    const expandButton = document.querySelector(expandButtonSelector);"
            "    const featuresPanel = document.querySelector(featuresPanelSelector);"
            "    if (expandButton && expandButton.getAttribute('aria-selected') === 'false') {"
            "        console.log('Features expand button indicates collapsed state, clicking...');"
            "        expandButton.click();"
            "        await new Promise(r => setTimeout(r, 1500));"
            "        console.log('Expand button clicked and waited.');"
            "    } else if (expandButton) {"
            "        console.log('Features expand button already indicates expanded state.');"
            "    } else {"
            "        console.log('Features expand button selector not found:', expandButtonSelector);"
            "        if (featuresPanel && !featuresPanel.offsetParent) {"
            "           console.warn('Button not found, but panel seems hidden. JS might need adjustment.');"
            "        } else if (!featuresPanel) {"
            "           console.warn('Neither expand button nor features panel found.');"
            "        }"
            "    }"
            "})();"
        ),
        # Selector for the main container holding the features/specifications table
        "table_selector": "#pdp-features-tabpanel" # Example selector - VERIFY!
    }
]

# --- HTML Cleaning Function ---
def clean_scraped_html(html_content: str, site_name: str) -> Optional[str]:
    """
    Parses scraped HTML using BeautifulSoup and extracts key-value pairs
    from known structures (e.g., TE Connectivity feature lists).

    Args:
        html_content: The raw HTML string scraped from the website.
        site_name: The name of the site (e.g., "TE Connectivity") to apply specific parsing logic.

    Returns:
        A cleaned string representation (e.g., "Key: Value\\nKey: Value") or None if parsing fails.
    """
    if not html_content:
        return None

    logger.debug(f"Cleaning HTML content from {site_name}...")
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_texts = []

    try:
        # --- Add site-specific parsing logic here --- 
        if site_name == "TE Connectivity":
            # Find all feature list items within the main panel
            feature_items = soup.find_all('li', class_='product-feature')
            if not feature_items:
                 # Maybe the main selector was wrong? Try finding the panel first
                 panel = soup.find(id='pdp-features-tabpanel')
                 if panel:
                      feature_items = panel.find_all('li', class_='product-feature')
                 
            if feature_items:
                for item in feature_items:
                    title_span = item.find('span', class_='feature-title')
                    value_em = item.find('em', class_='feature-value')
                    if title_span and value_em:
                        title = title_span.get_text(strip=True).replace(':', '').strip()
                        value = value_em.get_text(strip=True)
                        if title and value:
                            extracted_texts.append(f"{title}: {value}")
                logger.info(f"Extracted {len(extracted_texts)} features from TE Connectivity HTML.")
            else:
                 logger.warning(f"Could not find 'li.product-feature' items in the TE Connectivity HTML provided.")

        # Add logic for other sites if needed
        else:
            logger.warning(f"No specific HTML cleaning logic defined for site: {site_name}. Returning raw text content as fallback.")
            # Fallback: return just the text content of the whole block
            return soup.get_text(separator=' ', strip=True)

        if not extracted_texts:
            logger.warning(f"HTML cleaning for {site_name} resulted in no text extracted.")
            return None # Return None if nothing was extracted

        return "\\n".join(extracted_texts)

    except Exception as e:
        logger.error(f"Error cleaning HTML for {site_name}: {e}", exc_info=True)
        return None # Return None on parsing error

# --- Web Scraping Function (Revised to call cleaner) ---
async def scrape_website_table_html(part_number: str) -> Optional[str]:
    """
    Attempts to scrape the outer HTML of a features table, then cleans it.
    """
    if not part_number:
        logger.debug("Web scraping skipped: No part number provided.")
        return None

    logger.info(f"Attempting web scrape for features table / Part#: '{part_number}'...")

    for site_config in WEBSITE_CONFIGS:
        selector = site_config.get("table_selector")
        site_name = site_config.get("name", "Unknown Site") # Get site name for cleaner
        if not selector:
             logger.warning(f"No table_selector defined for {site_name}. Skipping.")
             continue

        target_url = site_config["base_url_template"].format(part_number=part_number)
        js_code = site_config.get("pre_extraction_js")
        logger.debug(f"Attempting scrape on {site_name} ({target_url}) for table selector '{selector}'")

        # Configure crawler run - Use JsonCssExtractionStrategy to get outerHTML
        extraction_schema = {
            "name": "TableHTML",
            "baseSelector": "html", # Apply to whole document
            "fields": [
                # Try type: "html" to get the inner/outer HTML of the element
                {"name": "html_content", "selector": selector, "type": "html"}
            ]
        }
        run_config = CrawlerRunConfig(
                 cache_mode=CacheMode.BYPASS,
                 js_code=[js_code] if js_code else None,
                 page_timeout=20000,
                 verbose=False, # Set to True for detailed crawl4ai logs
                 extraction_strategy=JsonCssExtractionStrategy(extraction_schema) # Add strategy
            )
        browser_config = BrowserConfig(verbose=False) # Headless default

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Pass the single run_config object
                results = await crawler.arun_many(urls=[target_url], config=run_config)
                result = results[0]

                # Check for success and extracted content from the strategy
                if result.success and result.extracted_content:
                    raw_html = None
                    try:
                        extracted_data_list = json.loads(result.extracted_content)
                        if extracted_data_list and isinstance(extracted_data_list, list) and len(extracted_data_list) > 0:
                            first_item = extracted_data_list[0]
                            if isinstance(first_item, dict) and "html_content" in first_item:
                                raw_html = str(first_item["html_content"]).strip()
                        else:
                            logger.debug(f"Extraction strategy did not find or extract HTML for selector '{selector}' on {site_name}.")

                    except json.JSONDecodeError:
                         logger.warning(f"Failed to parse JSON from crawl4ai extraction result for table HTML on {site_name}: {result.extracted_content[:100]}...")
                    except Exception as parse_error:
                         logger.error(f"Error processing extracted JSON for {site_name}: {parse_error}", exc_info=True)

                    # --- Pass raw HTML to cleaner --- 
                    if raw_html:
                        cleaned_text = clean_scraped_html(raw_html, site_name)
                        if cleaned_text:
                            logger.success(f"Successfully scraped and cleaned features table from {site_name}.")
                            return cleaned_text # Return the cleaned text
                        else:
                             logger.warning(f"HTML was scraped from {site_name}, but cleaning failed or yielded no text.")
                    # else: (already logged failure to extract HTML)

                elif result.error_message:
                     logger.warning(f"Scraping page failed for {site_name} ({target_url}): {result.error_message}")
                else:
                    logger.debug(f"Scraping attempt for {site_name} yielded no extracted content or error message.")

        except asyncio.TimeoutError:
             logger.warning(f"Scraping timed out for {site_name} ({target_url})")
        except Exception as e:
            logger.error(f"Unexpected error during web scraping for {site_name} ({target_url}): {e}", exc_info=True)

    logger.info(f"Web scraping finished for features table. No usable cleaned text found across configured sites.")
    return None


# --- PDF Extraction Chain (Using Retriever and Detailed Instructions) ---
def create_pdf_extraction_chain(retriever, llm):
    """
    Creates a RAG chain that uses ONLY PDF context (via retriever)
    and detailed instructions to answer an extraction task.
    """
    if retriever is None or llm is None:
        logger.error("Retriever or LLM is not initialized for PDF extraction chain.")
        return None

    # Template using only PDF context and detailed instructions passed at runtime
    template = """
You are an expert data extractor. Your goal is to extract a specific piece of information based on the Extraction Instructions provided below, using ONLY the Document Context from PDFs.

Part Number Information (if provided by user):
{part_number}

--- Document Context (from PDFs) ---
{context}
--- End Document Context ---

Extraction Instructions:
{extraction_instructions}

Available Dictionary Values for "{attribute_key}":
{available_values}

---
IMPORTANT: For the attribute key "{attribute_key}", do the following:
1. Look for information in the Document Context that matches the Extraction Instructions
2. Find the BEST MATCH from the Available Dictionary Values above
3. If no match is found in the dictionary, use "NOT FOUND" or appropriate default value
4. Respond with ONLY a single, valid JSON object containing exactly one key-value pair:
   - The key MUST be the string: "{attribute_key}"
   - The value MUST be one of the available dictionary values or "NOT FOUND"
5. Do NOT include any explanations, intermediate answers, reasoning, or any text outside of the single JSON object in your response.

Example Output Format:
{{"{attribute_key}": "best_match_from_dictionary"}}

Output:
"""
    prompt = PromptTemplate.from_template(template)

    # Chain uses retriever to get PDF context based on extraction instructions
    pdf_chain = (
        RunnableParallel(
            context=lambda x: format_docs(retrieve_and_log_chunks(retriever, x['extraction_instructions'], x['attribute_key'])),
            extraction_instructions=lambda x: x['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key'],
            part_number=lambda x: x.get('part_number', "Not Provided"),
            available_values=lambda x: str(ATTRIBUTE_DICT.get(x['attribute_key'], []))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("PDF Extraction RAG chain created successfully.")
    return pdf_chain

# --- Web Data Extraction Chain (Using Cleaned Web Text and Simple Prompt) ---
def create_web_extraction_chain(llm):
    """
    Creates a simpler chain that uses ONLY cleaned website data
    and a direct instruction to extract an attribute strictly.
    """
    if llm is None:
        logger.error("LLM is not initialized for Web extraction chain.")
        return None

    # Simplified template allowing reasoning based on web data and instructions
    template = """
You are an expert data extractor. Your goal is to answer a specific piece of information by applying the logic described in the 'Extraction Instructions' to the 'Cleaned Scraped Website Data' provided below. Use ONLY the provided website data as your context.

--- Cleaned Scraped Website Data ---
{cleaned_web_data}
--- End Cleaned Scraped Website Data ---

Extraction Instructions:
{extraction_instructions}

---
IMPORTANT: For the attribute key "{attribute_key}", do the following:
1. Independently answer the extraction task THREE times, as if reasoning from scratch each time, using only the provided Cleaned Scraped Website Data and Extraction Instructions.
2. Internally compare your three answers and select the one that is most consistent or most frequent among them. If all three answers are different, choose the one you believe is most justified by the context and instructions.
3. Respond with ONLY a single, valid JSON object containing exactly one key-value pair:
   - The key MUST be the string: "{attribute_key}"
   - The value MUST be the final answer you selected (as a JSON string).
   - If the information cannot be determined from the Cleaned Scraped Website Data based on the instructions, the value MUST be "NOT FOUND".
4. Do NOT include any explanations, intermediate answers, reasoning, or any text outside the JSON object.

Example Output Format:
{{"{attribute_key}": "extracted_value_based_on_instructions"}}

Output:
"""
    prompt = PromptTemplate.from_template(template)

    # Chain structure simplified to handle inputs directly
    web_chain = (
        RunnableParallel(
            cleaned_web_data=lambda x: x['cleaned_web_data'],
            extraction_instructions=lambda x: x['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key']
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("Web Data Extraction chain created successfully (accepts instructions).")
    return web_chain


# --- NuMind Integration for Structured Extraction ---

# NuMind configuration
NUMIND_API_KEY = os.getenv("NUMIND_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJkVWRIUGZnUlk3NzBiMHNvZlRFUWlWU2MyMW9kRENRbmcxZE5ZZjR2b1dBIn0.eyJleHAiOjE3ODM2MjA5NTksImlhdCI6MTc1MjA5MDU0MiwiYXV0aF90aW1lIjoxNzUyMDg0OTU5LCJqdGkiOiJiNzIzYzc1MS00MWUyLTRmNTMtODYzMC1kNjU3NzE1YzMxMGEiLCJpc3MiOiJodHRwczovL3VzZXJzLm51bWluZC5haS9yZWFsbXMvZXh0cmFjdC1wbGF0Zm9ybSIsImF1ZCI6WyJhY2NvdW50IiwiYXBpIl0sInN1YiI6IjNlOTEyNTlhLWVkZGEtNDc0YS04ZWZhLWZlOWMzYzg2NjcxOSIsInR5cCI6IkJlYXJlciIsImF6cCI6InVzZXIiLCJzaWQiOiIwOTA3NDE5ZC1lM2Y1LTRlOTctOWMxZi00ZmVlMGE4M2Q5MjUiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIi8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1leHRyYWN0LXBsYXRmb3JtIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJvcmdhbml6YXRpb25zIjp7fSwibmFtZSI6IkhhbWRpIEJhYW5hbm91IiwiY2xpZW50IjoiYXBpIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYmFhbmFub3Vjb250YWN0QGdtYWlsLmNvbSIsImdpdmVuX25hbWUiOiJIYW1kaSIsImZhbWlseV9uYW1lIjoiQmFhbmFub3UiLCJlbWFpbCI6ImJhYW5hbm91Y29udGFjdEBnbWFpbC5jb20ifQ.DSAc5gkuzR8Kip40QFA32pVRYfmn7dzCNHcEZUIryI5n1z2U5m5gQ70qRH4brwgwuzEiUnn3TgJ0gALAbjNRU1V4K-KICPBny_eNmm2UhQBEUHqUqyjPbIjYZD6K4-gcBbdMoZzSNpFaSmYfZBK1xt4QDmXrKkLhumm8cJ5P_sphtRpYHhQ6CmAorfRQ4Bzg2jaYc20Pu4-Vqn2uxtGEG_KOW2wkwUPcDfGY0cx1H5oTFk7P4o1u6w8tzvMcjgf510cTgyk0rtYnPY8UguORuoY35D0cCTygWUhXZSHkEOSsSEs8MlR6wXn5EQ_4Ht1ZM5vjFRfWOdJO4zP0pd6Yxw")
NUMIND_PROJECT_ID = os.getenv("NUMIND_PROJECT_ID", "dab6080e-5409-43b0-8f02-7a844ba933d5")

def create_numind_extraction_chain():
    """
    Creates a NuMind extraction chain for structured data extraction.
    Returns the NuMind client if properly configured, None otherwise.
    """
    try:
        from numind import NuMind
        
        if not NUMIND_API_KEY or not NUMIND_PROJECT_ID:
            logger.warning("NuMind API key or project ID not configured. NuMind extraction will be disabled.")
            return None
            
        client = NuMind(api_key=NUMIND_API_KEY)
        logger.info("NuMind extraction chain created successfully.")
        return client
        
    except ImportError:
        logger.warning("NuMind SDK not installed. Install with: pip install numind")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize NuMind client: {e}")
        return None

async def extract_with_numind_from_bytes(client, file_bytes: bytes, attribute_key: str) -> Optional[Dict[str, Any]]:
    """
    Extract specific attribute using NuMind API with file bytes.
    Uses the extraction schema configured in the NuMind project.
    
    Args:
        client: NuMind client instance
        file_bytes: File content as bytes
        attribute_key: The attribute key to extract
        
    Returns:
        Dictionary with extraction result or None if failed
    """
    if not client or not file_bytes or not attribute_key:
        logger.warning("NuMind extraction skipped: missing client, file_bytes, or attribute_key")
        return None
        
    try:
        logger.info(f"Starting NuMind extraction for attribute '{attribute_key}' from file bytes (size: {len(file_bytes)})")
        
        # Get the extraction schema from the project first
        try:
            project_info = client.get_api_projects_projectid(NUMIND_PROJECT_ID)
            logger.info(f"Retrieved project info: {project_info}")
        except Exception as e:
            logger.warning(f"Could not retrieve project info: {e}")
            project_info = None
        
        # Call the NuMind API with the configured extraction schema
        # The API will use the extraction template configured in your NuMind project
        output_schema = client.post_api_projects_projectid_extract(NUMIND_PROJECT_ID, file_bytes)
        
        if output_schema and hasattr(output_schema, 'model_dump'):
            result = output_schema.model_dump()
            logger.success(f"NuMind extraction completed for '{attribute_key}'")
            logger.debug(f"NuMind result structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result
        else:
            logger.warning(f"NuMind extraction returned invalid result for '{attribute_key}'")
            return None
            
    except Exception as e:
        logger.error(f"NuMind extraction failed for '{attribute_key}': {e}")
        return None

async def extract_with_numind_using_schema(client, file_bytes: bytes, extraction_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract using NuMind API. The extraction template should be configured in the NuMind project.
    
    Args:
        client: NuMind client instance
        file_bytes: File content as bytes
        extraction_schema: The extraction schema (for reference, but not passed to API)
        
    Returns:
        Dictionary with extraction result or None if failed
    """
    if not client or not file_bytes:
        logger.warning("NuMind extraction skipped: missing client or file_bytes")
        return None
        
    try:
        logger.info(f"Starting NuMind extraction from file bytes (size: {len(file_bytes)})")
        
        # Call the NuMind API - it uses the extraction template configured in the project
        output_schema = client.post_api_projects_projectid_extract(
            NUMIND_PROJECT_ID, 
            file_bytes
        )
        
        if output_schema and hasattr(output_schema, 'model_dump'):
            result = output_schema.model_dump()
            logger.success("NuMind extraction completed")
            logger.debug(f"NuMind result structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result
        else:
            logger.warning("NuMind extraction returned invalid result")
            return None
            
    except Exception as e:
        logger.error(f"NuMind extraction failed: {e}")
        return None

def get_default_extraction_schema() -> Dict[str, Any]:
    """
    Returns the default extraction schema that matches your NuMind playground configuration.
    You should replace this with the actual schema from your NuMind project.
    """
    # This is a template - you need to replace this with your actual NuMind extraction schema
    # You can get this from your NuMind playground by copying the schema configuration
    return {
        "type": "object",
        "properties": {
            "Material Filling": {
                "type": "string",
                "description": "The material filling or additive used in the connector"
            },
            "Material Name": {
                "type": "string", 
                "description": "The main material name of the connector"
            },
            "Pull-To-Seat": {
                "type": "string",
                "description": "Pull-to-seat force or mechanism information"
            },
            "Gender": {
                "type": "string",
                "description": "The gender of the connector (male/female)"
            },
            "Height [MM]": {
                "type": "string",
                "description": "Height of the connector in millimeters"
            },
            "Length [MM]": {
                "type": "string",
                "description": "Length of the connector in millimeters"
            },
            "Width [MM]": {
                "type": "string",
                "description": "Width of the connector in millimeters"
            },
            "Number Of Cavities": {
                "type": "string",
                "description": "Number of cavities or positions in the connector"
            },
            "Number Of Rows": {
                "type": "string",
                "description": "Number of rows in the connector"
            },
            "Mechanical Coding": {
                "type": "string",
                "description": "Mechanical coding or keying information"
            },
            "Colour": {
                "type": "string",
                "description": "Color of the connector"
            },
            "Colour Coding": {
                "type": "string",
                "description": "Color coding information"
            },
            "Max. Working Temperature [°C]": {
                "type": "string",
                "description": "Maximum working temperature in Celsius"
            },
            "Min. Working Temperature [°C]": {
                "type": "string",
                "description": "Minimum working temperature in Celsius"
            },
            "Housing Seal": {
                "type": "string",
                "description": "Housing seal information"
            },
            "Wire Seal": {
                "type": "string",
                "description": "Wire seal information"
            },
            "Sealing": {
                "type": "string",
                "description": "Sealing information"
            },
            "Sealing Class": {
                "type": "string",
                "description": "Sealing class or IP rating"
            },
            "Contact Systems": {
                "type": "string",
                "description": "Contact system type"
            },
            "Terminal Position Assurance": {
                "type": "string",
                "description": "Terminal position assurance information"
            },
            "Connector Position Assurance": {
                "type": "string",
                "description": "Connector position assurance information"
            },
            "Name Of Closed Cavities": {
                "type": "string",
                "description": "Information about closed cavities"
            },
            "Pre-assembled": {
                "type": "string",
                "description": "Pre-assembly information"
            },
            "Type Of Connector": {
                "type": "string",
                "description": "Type of connector"
            },
            "Set/Kit": {
                "type": "string",
                "description": "Set or kit information"
            },
            "HV Qualified": {
                "type": "string",
                "description": "High voltage qualification information"
            }
        },
        "required": []
    }

def extract_specific_attribute_from_numind_result(numind_result: Dict[str, Any], attribute_key: str) -> Optional[str]:
    """
    Extract a specific attribute value from NuMind extraction result.
    Based on the NuMind API response structure.
    
    Args:
        numind_result: The result dictionary from NuMind extraction
        attribute_key: The specific attribute key to extract
        
    Returns:
        The extracted value as string, or None if not found
    """
    if not numind_result or not isinstance(numind_result, dict):
        logger.warning(f"Invalid NuMind result for attribute '{attribute_key}': {type(numind_result)}")
        return None
        
    try:
        # NuMind response structure: result -> schemas -> attribute_name -> value
        if 'result' in numind_result and 'schemas' in numind_result['result']:
            schemas = numind_result['result']['schemas']
            
            if attribute_key in schemas:
                schema_data = schemas[attribute_key]
                
                # Handle different schema types based on NuMind structure
                if isinstance(schema_data, dict):
                    # String value
                    if 'value' in schema_data:
                        return str(schema_data['value']).strip()
                    # Array of values
                    elif 'values' in schema_data and isinstance(schema_data['values'], list):
                        # Return the first value or join multiple values
                        values = [str(v.get('value', v)) for v in schema_data['values'] if v]
                        return ', '.join(values) if values else None
                    # Boolean value
                    elif 'value' in schema_data and isinstance(schema_data['value'], bool):
                        return str(schema_data['value'])
                    # Number value
                    elif 'value' in schema_data and isinstance(schema_data['value'], (int, float)):
                        return str(schema_data['value'])
                
                # Direct string value
                elif isinstance(schema_data, str):
                    return schema_data.strip()
                # Direct number value
                elif isinstance(schema_data, (int, float)):
                    return str(schema_data)
        
        # Fallback: try to get the value directly from the result
        if attribute_key in numind_result:
            value = numind_result[attribute_key]
            if value is not None:
                return str(value).strip()
        
        # If not found directly, try to find it in nested structures
        for key, value in numind_result.items():
            if isinstance(value, dict) and attribute_key in value:
                nested_value = value[attribute_key]
                if nested_value is not None:
                    return str(nested_value).strip()
        
        logger.debug(f"Attribute '{attribute_key}' not found in NuMind result: {list(numind_result.keys())}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting attribute '{attribute_key}' from NuMind result: {e}")
        return None

# --- Helper function to invoke chain and process response (KEEP THIS) ---
async def _invoke_chain_and_process(chain, input_data, attribute_key):
    """Helper to invoke chain, handle errors, and clean response."""
    # Log the chunk/context and prompt sent to the LLM
    context_type = None
    context_value = None
    extraction_instructions = None
    if 'context' in input_data:
        context_type = 'PDF'
        context_value = input_data['context'] if isinstance(input_data['context'], str) else str(input_data['context'])
    elif 'cleaned_web_data' in input_data:
        context_type = 'Web'
        context_value = input_data['cleaned_web_data'] if isinstance(input_data['cleaned_web_data'], str) else str(input_data['cleaned_web_data'])#Vérifie si la réponse N'EST PAS une string 
    if 'extraction_instructions' in input_data:
        extraction_instructions = input_data['extraction_instructions'] if isinstance(input_data['extraction_instructions'], str) else str(input_data['extraction_instructions'])
    logger.debug(f"CHUNK SENT TO LLM ({context_type}):\nContext: {context_value[:1000]}\n---\nExtraction Instructions: {extraction_instructions}\n---\nAttribute Key: {attribute_key}")
    response = await chain.ainvoke(input_data)
    if response is None or not isinstance(response, str) or not response.strip():
        logger.error(f"Chain invocation returned None or empty for '{attribute_key}'")
        return json.dumps({"error": f"Chain invocation returned None or empty for {attribute_key}"})
    log_msg = f"Chain invoked successfully for '{attribute_key}'."
    # Add response length to log for debugging potential truncation/verboseness
    if response:
         log_msg += f" Response length: {len(response)}"
    logger.info(log_msg)

    return response # Validation happens in the caller (app.py now)