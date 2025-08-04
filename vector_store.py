# vector_store.py
from typing import List, Optional
from loguru import logger
import time
import requests
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import config # Import configuration

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
                    
                    # Success, break out of retry loop
                    break
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Batch {batch_num} timed out (attempt {retry + 1}/{max_retries})")
                    if retry == max_retries - 1:
                        logger.error(f"Batch {batch_num} failed after {max_retries} timeout attempts")
                        raise
                    time.sleep(1)  # Wait before retry
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Batch {batch_num} failed (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        logger.error(f"Batch {batch_num} failed after {max_retries} attempts")
                        raise
                    time.sleep(1)  # Wait before retry
                    
                except Exception as e:
                    logger.error(f"Unexpected error in batch {batch_num}: {e}")
                    raise
        
        logger.info(f"Successfully embedded {len(processed_texts)} documents in {len(all_embeddings)} batches")
        return all_embeddings

    def embed_documents_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback embedding method for individual document processing."""
        if not texts:
            return []
        
        max_text_length = config.EMBEDDING_MAX_TEXT_LENGTH
        all_embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Limit text length
                if len(text) > max_text_length:
                    logger.warning(f"Truncating text {i+1} from {len(text)} to {max_text_length} characters")
                    processed_text = text[:max_text_length]
                else:
                    processed_text = text
                
                # Process single document
                payload = {"texts": [processed_text]}
                
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=config.EMBEDDING_TIMEOUT
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract single embedding
                if "embeddings" in result:
                    embedding = result["embeddings"][0]
                elif "vectors" in result:
                    embedding = result["vectors"][0]
                elif isinstance(result, list):
                    embedding = result[0]
                else:
                    embedding = result.get("data", result.get("result", result))[0]
                
                all_embeddings.append(embedding)
                logger.debug(f"Successfully embedded document {i+1}/{len(texts)}")
                
            except Exception as e:
                logger.error(f"Failed to embed document {i+1}: {e}")
                # Return zero vector as fallback
                all_embeddings.append([0.0] * 1024)  # Assuming 768-dimensional embeddings
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if not text:
            return []
        
        max_text_length = config.EMBEDDING_MAX_TEXT_LENGTH
        
        # Limit text length
        if len(text) > max_text_length:
            logger.warning(f"Truncating query from {len(text)} to {max_text_length} characters")
            processed_text = text[:max_text_length]
        else:
            processed_text = text
        
        try:
            # Prepare the request payload
            payload = {"texts": [processed_text]}
            
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
                embedding = result["embeddings"][0]
            elif "vectors" in result:
                embedding = result["vectors"][0]
            elif isinstance(result, list):
                embedding = result[0]
            else:
                embedding = result.get("data", result.get("result", result))[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024  # Assuming 768-dimensional embeddings

# --- Embedding Function Setup ---
@logger.catch(reraise=True) # Automatically log exceptions
def get_embedding_function():
    """
    Creates and returns the embedding function based on configuration.
    Returns:
        HuggingFaceAPIEmbeddings instance if successful, None otherwise.
    """
    try:
        # Use the custom HuggingFace API embeddings
        embedding_function = HuggingFaceAPIEmbeddings(
            api_url=config.EMBEDDING_API_URL
        )
        
        # Test the embedding function with a simple query
        test_embedding = embedding_function.embed_query("test")
        if test_embedding and len(test_embedding) > 0:
            logger.success(f"Embedding function initialized successfully with {len(test_embedding)} dimensions")
            return embedding_function
        else:
            logger.error("Embedding function test failed - returned empty embedding")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
        return None

# --- Unified Simple Retriever (NEW) ---
class SimpleRetriever:
    """
    Centralized retrieval system that replaces all confusing retrieval methods.
    
    This ONE method handles all retrieval cases:
    - Simple semantic search
    - Attribute-specific search with enhanced queries
    - Part number filtering
    - Threshold filtering
    - Early stopping for performance
    """
    
    def __init__(self, vectorstore, config):
        self.vectorstore = vectorstore
        self.config = config
    

    

    
    def retrieve(self, query: str, attribute_key: str = None, 
                part_number: str = None) -> List[Document]:
        """
        Simplified retrieval: similarity search + tagging only.
        
        Args:
            query: The search query
            attribute_key: Optional attribute for tag filtering
            part_number: Optional part number for filtering
        
        Returns:
            List of relevant documents (max 5)
        """
        logger.info(f"🔍 SIMPLIFIED RETRIEVAL: query='{query}', attribute='{attribute_key}', part_number='{part_number}'")
        
        # 1. Simple similarity search with threshold filtering
        all_chunks = self._get_chunks_with_threshold(query)
        logger.info(f"📋 Retrieved {len(all_chunks)} chunks with similarity search")
        
        # 2. Apply part number filtering if needed
        if part_number:
            all_chunks = self._filter_by_part_number(all_chunks, part_number)
            logger.info(f"🔍 Part number filtering: {len(all_chunks)} chunks after filtering")
        
        # 3. Apply attribute tag filtering if needed (tag-aware retrieval)
        if attribute_key:
            original_count = len(all_chunks)
            all_chunks = self._filter_by_attribute_tag(all_chunks, attribute_key)
            logger.info(f"🏷️ Tag filtering: {len(all_chunks)} chunks after filtering (was {original_count})")
            
            # If no chunks found with attribute tags, fall back to semantic similarity only
            if not all_chunks and original_count > 0:
                logger.warning(f"No chunks found with '{attribute_key}' tag. Falling back to semantic similarity retrieval.")
                all_chunks = self._get_chunks_with_threshold(query)[:5]  # Take top 5 semantically similar chunks
                logger.info(f"Fallback: Using {len(all_chunks)} semantically similar chunks for '{attribute_key}'")
        
        # 4. Limit total chunks to avoid overwhelming the LLM
        max_chunks = 5
        if len(all_chunks) > max_chunks:
            logger.info(f"📊 Limiting chunks from {len(all_chunks)} to {max_chunks}")
            all_chunks = all_chunks[:max_chunks]
        
        logger.info(f"✅ Retrieved {len(all_chunks)} chunks for query '{query}'")
        return all_chunks
    

    
    def _get_chunks_with_threshold(self, query: str) -> List[Document]:
        """Get chunks with similarity threshold filtering."""
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.config.RETRIEVER_K
        )
        
        filtered_docs = []
        for doc, score in docs_and_scores:
            if score >= self.config.VECTOR_SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                logger.debug(f"Chunk passed threshold (score: {score:.3f}): {doc.page_content[:100]}")
            else:
                logger.debug(f"Chunk below threshold (score: {score:.3f}): {doc.page_content[:100]}")
        
        logger.info(f"Retrieved {len(docs_and_scores)} chunks, {len(filtered_docs)} passed threshold {self.config.VECTOR_SIMILARITY_THRESHOLD}")
        return filtered_docs
    
    def _filter_by_part_number(self, chunks: List[Document], part_number: str) -> List[Document]:
        """Filter chunks by part number if available."""
        filtered = []
        for chunk in chunks:
            chunk_part_number = chunk.metadata.get("part_number", "")
            
            # Check part number match (if part_number is provided AND stored in metadata)
            part_number_match = True
            if part_number and chunk_part_number:
                part_number_match = str(chunk_part_number).strip() == str(part_number).strip()
                logger.debug(f"Part number check: chunk='{chunk_part_number}' vs query='{part_number}' -> {part_number_match}")
            elif part_number and not chunk_part_number:
                # If user provided part number but chunk doesn't have it, skip the check
                logger.debug(f"Part number provided '{part_number}' but chunk has no part_number field, skipping part number check")
                part_number_match = True  # Allow through since we can't verify
            
            if part_number_match:
                filtered.append(chunk)
                logger.debug(f"Chunk accepted: part_number={chunk_part_number}")
            else:
                logger.debug(f"Chunk rejected: part_number_match={part_number_match}")
        
        return filtered
    
    def _filter_by_attribute_tag(self, chunks: List[Document], attribute_key: str) -> List[Document]:
        """Filter chunks by attribute tag (tag-aware retrieval)."""
        filtered = []
        for chunk in chunks:
            chunk_attr_value = chunk.metadata.get(attribute_key)
            
            # Check attribute tag exists and is not empty
            attr_tag_exists = chunk_attr_value is not None and chunk_attr_value != ""
            logger.debug(f"Attribute tag check: {attribute_key}='{chunk_attr_value}' -> {attr_tag_exists}")
            
            if attr_tag_exists:
                filtered.append(chunk)
                logger.debug(f"Chunk accepted: {attribute_key}={chunk_attr_value}")
            else:
                logger.debug(f"Chunk rejected: attr_tag_exists={attr_tag_exists}")
        
        return filtered

# --- Vector Store Setup Functions ---
@logger.catch(reraise=True)
def setup_vector_store(
    documents: List[Document],
    embedding_function,
) -> Optional[SimpleRetriever]:
    """
    Sets up a Chroma vector store with the provided documents and embedding function.
    Args:
        documents: List of documents to add to the vector store.
        embedding_function: The embedding function to use.
    Returns:
        A SimpleRetriever object if successful, otherwise None.
    """
    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    if not persist_directory:
        logger.warning("Persistence directory not configured. Cannot setup vector store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for setup_vector_store.")
        return None

    logger.info(f"Setting up vector store '{collection_name}' with {len(documents)} documents...")

    try:
        # Create the vector store with batch processing
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory=persist_directory
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
        # Return the new SimpleRetriever
        return SimpleRetriever(
            vectorstore=vector_store,
            config=config
        )

    except Exception as e:
        logger.error(f"Failed to setup vector store: {e}", exc_info=True)
        return None