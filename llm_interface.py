# llm_interface.py
import json
from typing import List, Dict, Optional
from loguru import logger
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore.document import Document

# Recommended: Use LangChain's Groq integration
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import config # Import configuration
import asyncio # Need asyncio for crawl4ai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from bs4 import BeautifulSoup # Import BeautifulSoup


import os



# Load attribute dictionary
def load_attribute_dictionary():
    """Load the attribute dictionary from JSON file."""
    try:
        dict_path = os.path.join(os.path.dirname(__file__), 'attribute_dictionary.json')
        with open(dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load attribute dictionary: {e}")
        return {}

# Load the attribute dictionary
ATTRIBUTE_DICT = load_attribute_dictionary()

# --- LLM Initialization ---
@logger.catch(reraise=True) # Keep catch for unexpected errors during init
def initialize_llm():
    """
    Initialize the LLM client with proper error handling and fallback.
    Returns the initialized LangChain Groq LLM or None if initialization fails.
    """
    try:

        
        if not config.GROQ_API_KEY:
            logger.error("GROQ_API_KEY not found in environment variables.")
            return None
            
        llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS
        )
        logger.info("LangChain Groq LLM initialized successfully.")
        return llm
        
    except ImportError:
        logger.error("LangChain Groq library not installed. Install with: pip install langchain-groq")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LangChain Groq LLM: {e}")
        return None

# --- Document Formatting ---
def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

# --- Website Scraping Functions ---

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

    # Chain uses SimpleRetriever to get PDF context based on extraction instructions
    pdf_chain = (
        RunnableParallel(
            context=lambda x: format_docs(retriever.retrieve(
                query=x['extraction_instructions'],
                attribute_key=x['attribute_key'],
                part_number=x.get('part_number')
            )),
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
import os
from typing import Dict, Any, Optional
from numind_schema_config import get_custom_template

# NuMind configuration
NUMIND_API_KEY = os.getenv("NUMIND_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJkVWRIUGZnUlk3NzBiMHNvZlRFUWlWU2MyMW9kRENRbmcxZE5ZZjR2b1dBIn0.eyJleHAiOjE3ODM2MjA5NTksImlhdCI6MTc1MjA5MDU0MiwiYXV0aF90aW1lIjoxNzUyMDg0OTU5LCJqdGkiOiJiNzIzYzc1MS00MWUyLTRmNTMtODYzMC1kNjU3NzE1YzMxMGEiLCJpc3MiOiJodHRwczovL3VzZXJzLm51bWluZC5haS9yZWFsbXMvZXh0cmFjdC1wbGF0Zm9ybSIsImF1ZCI6WyJhY2NvdW50IiwiYXBpIl0sInN1YiI6IjNlOTEyNTlhLWVkZGEtNDc0YS04ZWZhLWZlOWMzYzg2NjcxOSIsInR5cCI6IkJlYXJlciIsImF6cCI6InVzZXIiLCJzaWQiOiIwOTA3NDE5ZC1lM2Y1LTRlOTctOWMxZi00ZmVlMGE4M2Q5MjUiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIi8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1leHRyYWN0LXBsYXRmb3JtIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJvcmdhbml6YXRpb25zIjp7fSwibmFtZSI6IkhhbWRpIEJhYW5hbm91IiwiY2xpZW50IjoiYXBpIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYmFhbmFub3Vjb250YWN0QGdtYWlsLmNvbSIsImdpdmVuX25hbWUiOiJIYW1kaSIsImZhbWlseV9uYW1lIjoiQmFhbmFub3UiLCJlbWFpbCI6ImJhYW5hbm91Y29udGFjdEBnbWFpbC5jb20ifQ.DSAc5gkuzR8Kip40QFA32pVRYfmn7dzCNHcEZUIryI5n1z2U5m5gQ70qRH4brwgwuzEiUnn3TgJ0gALAbjNRU1V4K-KICPBny_eNmm2UhQBEUHqUqyjPbIjYZD6K4-gcBbdMoZzSNpFaSmYfZBK1xt4QDmXrKkLhumm8cJ5P_sphtRpYHhQ6CmAorfRQ4Bzg2jaYc20Pu4-Vqn2uxtGEG_KOW2wkwUPcDfGY0cx1H5oTFk7P4o1u6w8tzvMcjgf510cTgyk0rtYnPY8UguORuoY35D0cCTygWUhXZSHkEOSsSEs8MlR6wXn5EQ_4Ht1ZM5vjFRfWOdJO4zP0pd6Yxw")

def create_numind_extraction_chain():
    """
    Creates a NuMind extraction chain for structured data extraction.
    Returns the NuMind client if properly configured, None otherwise.
    """
    try:
        from numind import NuMind
        
        if not NUMIND_API_KEY:
            logger.warning("NuMind API key not configured. NuMind extraction will be disabled.")
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

async def extract_with_numind_using_template(client, file_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Extract using NuMind API with template-based extraction.
    
    Args:
        client: NuMind client instance
        file_bytes: File content as bytes
        
    Returns:
        Dictionary with extraction result or None if failed
    """
    if not client or not file_bytes:
        logger.warning("NuMind extraction skipped: missing client or file_bytes")
        return None
        
    try:
        logger.info(f"Starting NuMind template-based extraction from file bytes (size: {len(file_bytes)})")
        
        # Get the custom template from configuration
        template = get_custom_template()
        
        # Call the NuMind API using template-based extraction
        output_schema = client.extract(
            template=template,
            input_file=file_bytes
        )
        
        if output_schema:
            logger.success("NuMind template-based extraction completed")
            logger.debug(f"NuMind response type: {type(output_schema)}")
            
            # Handle ExtractionResponse object - convert to dictionary
            if hasattr(output_schema, 'model_dump'):
                result = output_schema.model_dump()
                logger.debug(f"NuMind result structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                return result
            elif isinstance(output_schema, dict):
                logger.debug(f"NuMind result structure: {list(output_schema.keys())}")
                return output_schema
            else:
                logger.warning(f"Unexpected NuMind result type: {type(output_schema)}")
                # Try to get more info about the object
                try:
                    logger.debug(f"NuMind response attributes: {dir(output_schema)}")
                    if hasattr(output_schema, '__dict__'):
                        logger.debug(f"NuMind response __dict__: {output_schema.__dict__}")
                except Exception as e:
                    logger.debug(f"Could not inspect NuMind response: {e}")
                return None
        else:
            logger.warning("NuMind template-based extraction returned invalid result")
            return None
            
    except Exception as e:
        logger.error(f"NuMind template-based extraction failed: {e}")
        return None

def extract_specific_attribute_from_numind_result(numind_result: Dict[str, Any], attribute_key: str) -> Optional[str]:
    """
    Extract a specific attribute value from NuMind template-based extraction result.
    The template-based extraction returns a flat dictionary with attribute names as keys.
    
    Args:
        numind_result: The result dictionary from NuMind template-based extraction
        attribute_key: The specific attribute key to extract
        
    Returns:
        The extracted value as string, or None if not found
    """
    if not numind_result:
        logger.warning(f"Invalid NuMind result for attribute '{attribute_key}': None or empty")
        return None
        
    if not isinstance(numind_result, dict):
        logger.warning(f"Invalid NuMind result for attribute '{attribute_key}': {type(numind_result)}")
        return None
        
    try:
        # Template-based extraction returns a flat dictionary with attribute names as keys
        if attribute_key in numind_result:
            value = numind_result[attribute_key]
            if value is not None:
                return str(value).strip()
        
        # If not found directly, try to find it in nested structures (fallback)
        for key, value in numind_result.items():
            if isinstance(value, dict) and attribute_key in value:
                nested_value = value[attribute_key]
                if nested_value is not None:
                    return str(nested_value).strip()
        
        logger.debug(f"Attribute '{attribute_key}' not found in NuMind template result: {list(numind_result.keys()) if isinstance(numind_result, dict) else type(numind_result)}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting attribute '{attribute_key}' from NuMind template result: {e}")
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
        context_value = input_data['cleaned_web_data'] if isinstance(input_data['cleaned_web_data'], str) else str(input_data['cleaned_web_data'])
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