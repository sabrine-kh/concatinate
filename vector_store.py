# vector_store.py
from typing import List, Optional
from loguru import logger
import os
import time
import requests
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from chromadb import Client as ChromaClient
from langchain.embeddings.base import Embeddings

import config # Import configuration
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# --- Custom Hugging Face API Embeddings ---
class HuggingFaceAPIEmbeddings(Embeddings):
    """Custom embeddings class that uses Hugging Face API instead of local model."""
    
    def __init__(self, api_url: str = "https://hbaananou-embedder-model.hf.space/embed"):
        self.api_url = api_url
        logger.info(f"Initialized HuggingFace API embeddings with URL: {api_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Hugging Face API with batching and text length limiting."""
        if not texts:
            return []
        
        # Batch size - adjust based on your API's capacity
        batch_size = config.EMBEDDING_BATCH_SIZE
        max_text_length = config.EMBEDDING_MAX_TEXT_LENGTH
        all_embeddings = []
        
        # Pre-process texts to limit length
        processed_texts = []
        for text in texts:
            if len(text) > max_text_length:
                logger.warning(f"Truncating text from {len(text)} to {max_text_length} characters")
                processed_text = text[:max_text_length]
            else:
                processed_text = text
            processed_texts.append(processed_text)
        
        # Process texts in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(processed_texts) + batch_size - 1)//batch_size
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} with {len(batch_texts)} texts")
            
            # Calculate total characters in this batch
            total_chars = sum(len(text) for text in batch_texts)
            logger.debug(f"Batch {batch_num} total characters: {total_chars}")
            
            # Retry logic for failed batches
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Prepare the request payload
                    payload = {"texts": batch_texts}
                    
                    # Make the API request with increased timeout for batches
                    response = requests.post(
                        self.api_url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=config.EMBEDDING_TIMEOUT  # Configurable timeout for batch processing
                    )
                    
                    # Check if the request was successful
                    response.raise_for_status()
                    
                    # Parse the response
                    result = response.json()
                    
                    # Extract embeddings from the response
                    # Handle different API response formats
                    if "embeddings" in result:
                        batch_embeddings = result["embeddings"]
                    elif "vectors" in result:
                        batch_embeddings = result["vectors"]
                    elif isinstance(result, list):
                        # If the API returns embeddings directly as a list
                        batch_embeddings = result
                    else:
                        # Try to find embeddings in the response structure
                        batch_embeddings = result.get("data", result.get("result", result))
                        if not isinstance(batch_embeddings, list):
                            raise ValueError(f"Unexpected API response format: {result}")
                    
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(f"Successfully embedded batch {batch_num} with {len(batch_texts)} documents")
                    
                    # Add a small delay between batches to avoid overwhelming the API
                    if i + batch_size < len(processed_texts):
                        time.sleep(1.0)  # Increased delay for large files
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.Timeout as e:
                    logger.warning(f"Timeout for batch {batch_num} (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        logger.error(f"All retries failed for batch {batch_num}")
                        raise
                    time.sleep(2.0)  # Wait before retry
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"API request failed for batch {batch_num}: {e}")
                    raise
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to parse API response for batch {batch_num}: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error during embedding for batch {batch_num}: {e}")
                    raise
        
        logger.debug(f"Successfully embedded all {len(texts)} documents via API in batches")
        return all_embeddings
    
    def embed_documents_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback method: embed documents one by one if batch processing fails."""
        logger.warning("Using fallback method: processing texts individually")
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Limit text length
                if len(text) > config.EMBEDDING_MAX_TEXT_LENGTH:
                    text = text[:config.EMBEDDING_MAX_TEXT_LENGTH]
                    logger.warning(f"Truncating text {i+1} from {len(text)} to {config.EMBEDDING_MAX_TEXT_LENGTH} characters")
                
                # Embed single text
                embedding = self.embed_query(text)
                embeddings.append(embedding)
                
                logger.debug(f"Successfully embedded text {i+1}/{len(texts)}")
                
                # Small delay between individual requests
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to embed text {i+1}: {e}")
                # Return a zero vector as fallback
                zero_vector = [0.0] * config.EMBEDDING_DIMENSIONS
                embeddings.append(zero_vector)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using the Hugging Face API."""
        if not text:
            return []
        
        try:
            # Prepare the request payload
            payload = {"texts": [text]}
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=config.EMBEDDING_TIMEOUT
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract embedding from the response
            if "embeddings" in result:
                return result["embeddings"][0]
            elif "vectors" in result:
                return result["vectors"][0]
            elif isinstance(result, list):
                return result[0]
            else:
                return result.get("data", result.get("result", result))[0]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for single query: {e}")
            raise
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response for single query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during single query embedding: {e}")
            raise

# --- Embedding Function ---
@logger.catch(reraise=True) # Automatically log exceptions
def get_embedding_function():
    """Initializes and returns the embedding function (API-based or local)."""
    # Check if we should use the API-based embeddings
    use_api_embeddings = os.getenv("USE_API_EMBEDDINGS", "true").lower() == "true"
    
    if use_api_embeddings:
        api_url = os.getenv("EMBEDDING_API_URL", "https://hbaananou-embedder-model.hf.space/embed")
        logger.info(f"Using HuggingFace API embeddings: {api_url}")
        return HuggingFaceAPIEmbeddings(api_url=api_url)
    else:
        # Fallback to local embeddings
        logger.info("Using local HuggingFace embeddings")
        model_kwargs = {'device': config.EMBEDDING_DEVICE}
        encode_kwargs = {'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}

        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embeddings

# --- ChromaDB Setup and Retrieval ---
_chroma_client = None # Module-level client cache

def get_chroma_client():
    """Gets or creates the ChromaDB client based on config."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing Chroma client (Persistent: {config.CHROMA_SETTINGS.is_persistent})")
        if config.CHROMA_SETTINGS.is_persistent:
            logger.info(f"Chroma persistence directory: {config.CHROMA_PERSIST_DIRECTORY}")
            # Ensure directory exists if persistent
            if config.CHROMA_PERSIST_DIRECTORY and not os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
                 os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        _chroma_client = ChromaClient(config.CHROMA_SETTINGS)
        logger.success("Chroma client initialized.")
    return _chroma_client

# --- Vector Store Setup ---
@logger.catch(reraise=True)
def setup_vector_store(
    documents: List[Document],
    embedding_function,
) -> Optional[VectorStoreRetriever]:
    """
    Sets up the Chroma vector store. Creates a new one if it doesn't exist,
    or potentially adds to an existing one (current logic replaces).
    Args:
        documents: List of Langchain Document objects.
        embedding_function: The embedding function to use.
    Returns:
        A VectorStoreRetriever object or None if setup fails.
    """
    if not documents:
        logger.warning("No documents provided to setup_vector_store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for setup_vector_store.")
        return None

    # Sort documents by source and page for deterministic indexing
    documents = sorted(documents, key=lambda doc: (doc.metadata.get('source', ''), doc.metadata.get('page', 0)))

    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    logger.info(f"Setting up vector store. Persistence directory: '{persist_directory}', Collection: '{collection_name}'")

    # Check if the directory exists, maybe Chroma needs it? (Optional check)
    # if persist_directory and not os.path.exists(persist_directory):
    #     logger.info(f"Creating persistence directory: {persist_directory}")
    #     os.makedirs(persist_directory)

    try:
        # If persisting, Chroma.from_documents handles creation and persistence directly
        # when the persist_directory argument is provided.
        logger.info(f"Creating/Updating vector store '{collection_name}' with {len(documents)} documents...")

        # *** Add persist_directory argument here ***
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory=persist_directory # <-- This is the crucial addition
            )
        except Exception as e:
            logger.warning(f"Batch processing failed, trying fallback method: {e}")
            # If batch processing fails, try individual processing
            if hasattr(embedding_function, 'embed_documents_fallback'):
                # Create a temporary embedding function that uses fallback
                class FallbackEmbeddingFunction:
                    def __init__(self, original_function):
                        self.original_function = original_function
                    
                    def embed_documents(self, texts):
                        return self.original_function.embed_documents_fallback(texts)
                    
                    def embed_query(self, text):
                        return self.original_function.embed_query(text)
                
                fallback_embedding = FallbackEmbeddingFunction(embedding_function)
                
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=fallback_embedding,
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
            else:
                raise e

        # Ensure persistence after creation/update
        if persist_directory:
            logger.info(f"Persisting vector store to directory: {persist_directory}")
            vector_store.persist() # Explicitly call persist just in case

        logger.success(f"Vector store '{collection_name}' created/updated and persisted successfully.")
        # Return the retriever with similarity threshold
        return ThresholdRetriever(
            vectorstore=vector_store,
            search_kwargs={"k": config.RETRIEVER_K},
            threshold=config.VECTOR_SIMILARITY_THRESHOLD
        )

    except Exception as e:
        logger.error("Failed to create or populate Chroma vector store '{}': {}".format(collection_name, e), exc_info=True)
        return None

# --- Load Existing Vector Store ---
@logger.catch(reraise=True)
def load_existing_vector_store(embedding_function) -> Optional[VectorStoreRetriever]:
    """
    Loads an existing Chroma vector store from the persistent directory.
    Args:
        embedding_function: The embedding function to use.
    Returns:
        A VectorStoreRetriever object if the store exists and loads, otherwise None.
    """
    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    if not persist_directory:
        logger.warning("Persistence directory not configured. Cannot load existing store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for load_existing_vector_store.")
        return None

    if not os.path.exists(persist_directory):
         logger.warning(f"Persistence directory '{persist_directory}' does not exist. Cannot load.")
         return None

    logger.info(f"Attempting to load existing vector store from: '{persist_directory}', Collection: '{collection_name}'")

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        # Simple check to see if it loaded something (e.g., count items)
        # Note: .count() might not exist directly, use a different check if needed
        # A simple successful initialization might be enough indication
        # try:
        #     count = vector_store._collection.count() # Example internal access, might change
        #     logger.info(f"Successfully loaded collection '{collection_name}' with {count} items.")
        # except Exception:
        #      logger.warning(f"Loaded collection '{collection_name}', but could not verify item count.")

        logger.success(f"Successfully loaded vector store '{collection_name}'.")
        return ThresholdRetriever(
            vectorstore=vector_store,
            search_kwargs={"k": config.RETRIEVER_K},
            threshold=config.VECTOR_SIMILARITY_THRESHOLD
        )

    except Exception as e:
        # This exception block might catch cases where the collection *within* the directory doesn't exist
        # or other Chroma loading errors.
        logger.warning("Failed to load existing vector store '{}' from '{}': {}".format(collection_name, persist_directory, e), exc_info=False) # Log less verbosely maybe
        # Log specific known issues like collection not found separately if possible
        if "does not exist" in str(e).lower(): # Basic check
             logger.warning("Persistent collection '{}' not found in directory '{}'. Cannot load.".format(collection_name, persist_directory))

        return None

# --- Custom Retriever with Similarity Threshold ---
class ThresholdRetriever:
    """Custom retriever that applies similarity threshold filtering."""
    
    def __init__(self, vectorstore, search_kwargs, threshold):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
        self.threshold = threshold
    
    def invoke(self, query: str) -> List[Document]:
        """Get documents with similarity threshold filtering."""
        # Get documents with scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, 
            k=self.search_kwargs.get("k", 8)
        )
        
        # Filter by threshold
        filtered_docs = []
        for doc, score in docs_and_scores:
            if score >= self.threshold:
                filtered_docs.append(doc)
                logger.debug("Chunk passed threshold (score: {:.3f}): {}".format(score, doc.page_content[:100]))
            else:
                logger.debug("Chunk below threshold (score: {:.3f}): {}".format(score, doc.page_content[:100]))
        
        logger.info("Retrieved {} chunks, {} passed threshold {}".format(len(docs_and_scores), len(filtered_docs), self.threshold))
        return filtered_docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain compatibility method - same as invoke."""
        return self.invoke(query)
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async version of document retrieval."""
        return self.invoke(query)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async LangChain compatibility method."""
        return self.invoke(query)