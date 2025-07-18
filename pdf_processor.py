# pdf_processor.py
import os
import re
import base64
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, BinaryIO, Optional, Dict, Any, Tuple
from loguru import logger
from PIL import Image
import fitz  # PyMuPDF
from mistralai.client import MistralClient
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import difflib

import config

# Global thread pool for PDF processing
pdf_thread_pool = ThreadPoolExecutor(max_workers=2)  # Adjust based on your needs

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
    for attr, values in attribute_dict.items():
        clean_values = [re.escape(v) for v in values if v]
        if not clean_values:
            continue
        pattern = r'(' + '|'.join(clean_values) + r')'
        regexes[attr] = re.compile(pattern, re.IGNORECASE)
        
        # Debug for Contact Systems
        if attr == "Contact Systems":
            logger.info(f"Contact Systems regex pattern: {pattern}")
            logger.info(f"Contact Systems clean values (first 5): {clean_values[:5]}")
    
    return regexes

ATTRIBUTE_REGEXES = build_attribute_regexes(ATTRIBUTE_DICTIONARY)

# --- Tagging Utility ---
def tag_chunk_with_dictionary(chunk_text, attribute_regexes):
    tags = {}
    logger.info(f"Tagging chunk with {len(attribute_regexes)} attribute regexes")
    
    # Special debugging for Contact Systems
    if "Contact Systems" in attribute_regexes:
        contact_regex = attribute_regexes["Contact Systems"]
        contact_matches = contact_regex.findall(chunk_text)
        logger.info(f"Contact Systems regex matches: {contact_matches}")
        logger.info(f"Looking for 'MCP 2.8' in text: {'MCP 2.8' in chunk_text}")
    
    for attr, regex in attribute_regexes.items():
        matches = regex.findall(chunk_text)
        # Convert matches to a list and handle empty lists properly for Chroma metadata
        match_list = sorted({m.strip() for m in matches})
        if match_list:
            # If we have matches, store them as a comma-separated string
            tags[attr] = ", ".join(match_list)
            logger.info(f"Found matches for '{attr}': {match_list}")
        else:
            # If no matches, store as None (Chroma accepts None as metadata value)
            tags[attr] = None
            logger.debug(f"No matches found for '{attr}'")
    
    # Log summary of what was found
    found_attrs = [attr for attr, value in tags.items() if value is not None]
    logger.info(f"Generated tags for {len(found_attrs)} attributes: {found_attrs}")
    logger.debug(f"All generated tags: {tags}")
    return tags

# --- Canonicalization Utility ---
def canonicalise(raw, attr_key):
    canonicals = ATTRIBUTE_DICTIONARY.get(attr_key, [])
    if not canonicals:
        return raw
    match = difflib.get_close_matches(str(raw), canonicals, n=1, cutoff=0.6)
    return match[0] if match else raw

def encode_pil_image(pil_image: Image.Image, format: str = "PNG") -> Tuple[str, str]:
    """Encode PIL Image to Base64 string."""
    buffered = io.BytesIO()
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
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8'), save_format.lower()

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
        total_pages = len(pdf_document)
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
        
        for page_num in range(total_pages):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing page {page_num + 1}/{total_pages} of {file_basename}")
            logger.debug(f"Page dimensions: {pdf_document[page_num].rect}")
            logger.info(f"{'='*50}\n")
            
            try:
                # Get the page
                page = pdf_document[page_num]
                
                # Convert page to image with higher resolution
                logger.debug("Converting page to high-resolution image...")
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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
                        messages=messages
                    )
                    logger.debug("Successfully received response from Mistral Vision API")
                except Exception as api_error:
                    logger.error(f"Mistral Vision API error: {str(api_error)}")
                    raise
                
                # Get extracted text
                page_content = chat_response.choices[0].message.content
                
                if page_content:
                    # --- Tag the chunk with dictionary matches ---
                    chunk_tags = tag_chunk_with_dictionary(page_content, ATTRIBUTE_REGEXES)
                    # Log the extracted content
                    logger.info("\nExtracted Content:")
                    logger.debug("-" * 40)
                    logger.debug(page_content)
                    logger.debug("-" * 40)
                    
                    # Instead of splitting into chunks, treat the whole page as one document
                    chunk_doc = Document(
                        page_content=page_content,
                        metadata={
                            'source': file_basename,
                            'page': page_num + 1,
                            **chunk_tags  # Add all attribute tags to metadata
                        }
                    )
                    all_docs.append(chunk_doc)
                    logger.debug(f"Created document for page {page_num + 1}")
                    
                    logger.success(f"Successfully processed page {page_num + 1} from {file_basename}")
                    total_pages_processed += 1
                else:
                    logger.warning(f"No content extracted from page {page_num + 1} of {file_basename}")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1} with Mistral Vision: {str(e)}", exc_info=True)
                
    except Exception as e:
        logger.error(f"Error processing {file_basename}: {str(e)}", exc_info=True)
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
    
    return all_docs

async def process_uploaded_pdfs(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> List[Document]:
    """Process uploaded PDFs using Mistral Vision for better text extraction."""
    all_docs: List[Document] = []
    saved_file_paths: List[str] = []
    
    logger.info(f"Starting batch processing of {len(uploaded_files)} PDF files")
    logger.debug(f"Temporary directory: {temp_dir}")
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
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
        for uploaded_file in uploaded_files:
            file_basename = uploaded_file.name
            file_path = os.path.join(temp_dir, file_basename)
            saved_file_paths.append(file_path)
            
            logger.debug(f"Saving uploaded file: {file_basename}")
            # Save uploaded file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            logger.debug(f"Successfully saved file: {file_basename}")
        
        # Process PDFs in parallel using ThreadPoolExecutor
        logger.info(f"Starting parallel processing of {len(saved_file_paths)} files")
        with ThreadPoolExecutor(max_workers=min(len(saved_file_paths), 4)) as executor:
            # Create tasks for each PDF
            loop = asyncio.get_event_loop()
            tasks: List[asyncio.Task] = []
            for file_path in saved_file_paths:
                file_basename = os.path.basename(file_path)
                logger.debug(f"Creating task for file: {file_basename}")
                # Create a task that runs in the thread pool
                task = loop.run_in_executor(
                    executor,
                    lambda p, b: asyncio.run(process_single_pdf(p, b, client, model_name)),
                    file_path,
                    file_basename
                )
                tasks.append(task)
            
            # Wait for all PDFs to be processed
            logger.info("Waiting for all PDF processing tasks to complete...")
            results = await asyncio.gather(*tasks)
            logger.info("All PDF processing tasks completed")
            
            # Combine all results in a deterministic order (by file name)
            results = [docs for docs in results if docs]
            results.sort(key=lambda docs: docs[0].metadata['source'] if docs and hasattr(docs[0], 'metadata') and 'source' in docs[0].metadata else '')
            all_docs = []
            for docs in results:
                all_docs.extend(docs)
                logger.debug(f"Added {len(docs)} documents from a processed file")
            
    except Exception as e:
        logger.error(f"Error during batch PDF processing: {str(e)}", exc_info=True)
    finally:
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for path in saved_file_paths:
            try:
                os.remove(path)
                logger.debug(f"Removed temporary file: {path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {path}: {str(e)}")
    
    if not all_docs:
        logger.error("No text could be extracted from any provided PDF files.")
    else:
        logger.info("\nFinal Processing Summary:")
        logger.info(f"Total documents processed: {len(saved_file_paths)}")
        logger.info(f"Total chunks created: {len(all_docs)}")
        logger.debug(f"Average chunks per document: {len(all_docs) / len(saved_file_paths):.2f}")
    
    return all_docs

def process_pdfs_in_background(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> asyncio.Task[List[Document]]:
    """Start PDF processing in the background and return a task that can be awaited later."""
    return asyncio.create_task(process_uploaded_pdfs(uploaded_files, temp_dir))

def fetch_chunks(retriever, part_number, attr_key, k=8):
    """
    Tag-aware retrieval: returns only chunks that match the part_number and have a non-empty tag for attr_key.
    Uses get_relevant_documents for compatibility with VectorStoreRetriever.
    """
    from loguru import logger
    
    # Get initial dense results
    dense_results = retriever.get_relevant_documents(attr_key)[:k]
    # Sort by source and page for deterministic order
    dense_results = sorted(dense_results, key=lambda c: (c.metadata.get('source', ''), c.metadata.get('page', 0)))
    logger.debug(f"Retrieved {len(dense_results)} initial chunks for attribute '{attr_key}'")
    
    # Special debugging for Contact Systems
    if attr_key == "Contact Systems":
        logger.info(f"=== SPECIAL DEBUG FOR CONTACT SYSTEMS ===")
        logger.info(f"Retrieved {len(dense_results)} chunks for Contact Systems")
        for i, chunk in enumerate(dense_results):
            logger.info(f"Chunk {i+1} metadata: {chunk.metadata}")
            logger.info(f"Chunk {i+1} has Contact Systems tag: {chunk.metadata.get('Contact Systems', 'NOT FOUND')}")
            logger.info(f"Chunk {i+1} content preview: {chunk.page_content[:200]}...")
    
    # Log metadata for debugging
    for i, chunk in enumerate(dense_results):
        logger.debug(f"Chunk {i+1} metadata: {chunk.metadata}")
    
    filtered = []
    for chunk in dense_results:
        chunk_part_number = chunk.metadata.get("part_number", "")
        chunk_attr_value = chunk.metadata.get(attr_key)
        
        # Check part number match (if part_number is provided AND stored in metadata)
        part_number_match = True
        if part_number and chunk_part_number:  # Only check if both are provided
            part_number_match = str(chunk_part_number).strip() == str(part_number).strip()
            logger.debug(f"Part number check: chunk='{chunk_part_number}' vs query='{part_number}' -> {part_number_match}")
        elif part_number and not chunk_part_number:
            # If user provided part number but chunk doesn't have it, skip the check
            logger.debug(f"Part number provided '{part_number}' but chunk has no part_number field, skipping part number check")
            part_number_match = True  # Allow through since we can't verify
        
        # Check attribute tag exists and is not empty
        attr_tag_exists = chunk_attr_value is not None and chunk_attr_value != ""
        logger.debug(f"Attribute tag check: {attr_key}='{chunk_attr_value}' -> {attr_tag_exists}")
        
        if part_number_match and attr_tag_exists:
            filtered.append(chunk)
            logger.debug(f"Chunk accepted: part_number={chunk_part_number}, {attr_key}={chunk_attr_value}")
        else:
            logger.debug(f"Chunk rejected: part_number_match={part_number_match}, attr_tag_exists={attr_tag_exists}")
    
    # If no chunks found with attribute tags, fall back to semantic similarity only
    if not filtered and dense_results:
        logger.warning(f"No chunks found with '{attr_key}' tag. Falling back to semantic similarity retrieval.")
        filtered = dense_results[:k]  # Take the top k semantically similar chunks
        logger.info(f"Fallback: Using {len(filtered)} semantically similar chunks for '{attr_key}'")

    # Sort filtered results for deterministic order
    filtered = sorted(filtered, key=lambda c: (c.metadata.get('source', ''), c.metadata.get('page', 0)))
    
    logger.info(f"Final result: {len(filtered)} chunks for attribute '{attr_key}' and part number '{part_number}'")
    return filtered