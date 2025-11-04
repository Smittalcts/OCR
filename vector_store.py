import os
import time
import random
from typing import List, Dict, Any, Set, Optional, Tuple
from langchain_chroma.vectorstores import Chroma
import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
import logging
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Enhanced vector store manager with ChromaDB optimizations:
    - Collection segmentation for faster search
    - Multi-vector embeddings support  
    - Parallel retrieval strategies
    - Query expansion with domain synonyms
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "induction_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Enhanced tracking for diversity
        self.used_chunk_ids = set()
        self.used_content_hashes = set()  # Track content similarity
        self.section_usage_count = {}  # Track usage by section
        self.last_reset_time = time.time()
        
        # Initialize Azure OpenAI embeddings model
        embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        
        logger.info(f"Initializing AzureOpenAIEmbeddings with deployment name: '{embedding_deployment}'")
        if not embedding_deployment:
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set.")
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
        )
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
        
        # NEW: Domain-specific query expansion synonyms
        self.query_synonyms = {
            'P1': ['Priority 1', 'Critical', 'P-1', 'Mission Critical', 'priority 1'],
            'P2': ['Priority 2', 'High', 'P-2', 'priority 2'],
            'incident': ['issue', 'problem', 'outage', 'disruption', 'service interruption'],
            'TIL': ['Technical Incident Lead', 'Incident Lead', 'tech incident lead'],
            'response time': ['SLA', 'resolution time', 'timeframe', 'response target'],
            '5 WHYs': ['5 Whys', 'root cause', 'RCA technique', 'five whys'],
            'CAB': ['Change Advisory Board', 'Approval Board', 'change board'],
            'ServiceNow': ['SNOW', 'service now', 'ticketing system', 'ITSM tool']
        }
        
        # NEW: Topic to collection mapping for segmentation
        self.topic_collections = {
            'Incident Management': 'incident_management',
            'Problem Management': 'problem_management',
            'Change Management': 'change_management',
            'Knowledge Management': 'knowledge_management',
            'Service Support': 'service_support',
            'Applications': 'applications'
        }
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store with optimized HNSW parameters"""
        try:
            # Define collection metadata with optimized HNSW parameters
            collection_metadata = {
                "hnsw:space": "cosine",  # Better for text similarity than L2
                "hnsw:M": 16,           # Connectivity - balance between recall and speed
                "hnsw:construction_ef": 200,  # Construction accuracy
                "hnsw:search_ef": 50,   # Search accuracy
                "hnsw:batch_size": 100, # Batch processing size
                "hnsw:sync_threshold": 1000,  # Disk sync threshold
                "description": "On-call induction document embeddings",
                "version": "1.0",
                "created_at": int(time.time())
            }
            
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata=collection_metadata
            )
            logger.info(f"Vector store initialized with optimized HNSW config: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> bool:
        """
        Adds documents to the vector store with batch processing and enhanced metadata.
        Uses optimized batch processing for better performance.
        """
        if not documents:
            logger.warning("No documents provided to add to the vector store.")
            return False
            
        logger.info(f"Adding {len(documents)} document chunks to ChromaDB with batch processing (batch_size={batch_size}).")
        
        try:
            total_docs = len(documents)
            
            # Process in batches for better performance
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch_docs = documents[batch_start:batch_end]
                
                # Enhance documents with metadata for better retrieval
                enhanced_docs = []
                for i, doc in enumerate(batch_docs):
                    # Add content hash for similarity tracking
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                    
                    # Extract section information if available
                    section = self._extract_section_info(doc.page_content)
                    
                    # Enhance metadata
                    enhanced_metadata = doc.metadata.copy() if doc.metadata else {}
                    enhanced_metadata.update({
                        'content_hash': content_hash,
                        'section': section,
                        'chunk_index': batch_start + i,
                        'content_length': len(doc.page_content),
                        'added_timestamp': int(time.time()),
                        'document_type': 'induction_manual'
                    })
                    
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata
                    )
                    enhanced_docs.append(enhanced_doc)
                
                # Add batch to vector store
                self.vector_store.add_documents(enhanced_docs)
                
                # Log progress every 5 batches
                if (batch_end % (batch_size * 5) == 0) or (batch_end == total_docs):
                    logger.info(f"Processed {batch_end}/{total_docs} documents")
            
            logger.info("Successfully added all documents to the vector store with batch processing.")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to vector store. Error: {e}")
            return False
    
    def _extract_section_info(self, content: str) -> str:
        """Extract section information from document content."""
        content_lower = content.lower()
        
        # Define section keywords
        sections = {
            'incident_management': ['incident', 'p1', 'p2', 'p3', 'p4', 'priority', 'severity'],
            'problem_management': ['problem', 'root cause', 'analysis', 'permanent fix'],
            'change_management': ['change', 'deployment', 'implementation', 'rollback'],
            'knowledge_management': ['knowledge', 'documentation', 'article', 'wiki'],
            'service_support': ['support', 'help desk', 'customer', 'user', 'service desk'],
            'escalation': ['escalate', 'escalation', 'senior', 'manager', 'team lead'],
            'communication': ['communicate', 'notification', 'email', 'phone', 'contact'],
            'documentation': ['document', 'log', 'record', 'update', 'notes']
        }
        
        # Find best matching section
        best_section = 'general'
        max_matches = 0
        
        for section, keywords in sections.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches > max_matches:
                max_matches = matches
                best_section = section
        
        return best_section
    
    def add_documents_with_embeddings(self, documents: List[Document], batch_size: int = 100) -> bool:
        """
        Adds documents with precomputed embeddings for better performance.
        Precomputing embeddings in batch is faster than computing them one-by-one.
        """
        if not documents:
            logger.warning("No documents provided to add to the vector store.")
            return False
            
        logger.info(f"Adding {len(documents)} documents with precomputed embeddings (batch_size={batch_size}).")
        
        try:
            total_docs = len(documents)
            
            # Process in batches
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch_docs = documents[batch_start:batch_end]
                
                # Extract texts for batch embedding
                texts = [doc.page_content for doc in batch_docs]
                
                # Precompute embeddings in batch (much faster)
                embeddings = self.embeddings.embed_documents(texts)
                
                # Enhance documents with metadata
                enhanced_docs = []
                for i, doc in enumerate(batch_docs):
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                    section = self._extract_section_info(doc.page_content)
                    
                    enhanced_metadata = doc.metadata.copy() if doc.metadata else {}
                    enhanced_metadata.update({
                        'content_hash': content_hash,
                        'section': section,
                        'chunk_index': batch_start + i,
                        'content_length': len(doc.page_content),
                        'added_timestamp': int(time.time()),
                        'document_type': 'induction_manual'
                    })
                    
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata
                    )
                    enhanced_docs.append(enhanced_doc)
                
                # Add with precomputed embeddings
                self.vector_store.add_documents(enhanced_docs, embeddings=embeddings)
                
                if (batch_end % (batch_size * 5) == 0) or (batch_end == total_docs):
                    logger.info(f"Processed {batch_end}/{total_docs} documents with precomputed embeddings")
            
            logger.info("Successfully added all documents with precomputed embeddings.")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents with embeddings. Error: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """
        Enhanced similarity search with quality filtering and native ChromaDB filtering.
        Supports metadata-based filtering for better precision.
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            # Use native ChromaDB filtering if filters provided
            if filters:
                raw_results = self.vector_store.similarity_search(
                    query, 
                    k=k*2,
                    filter=filters
                )
            else:
                raw_results = self.vector_store.similarity_search(query, k=k*2)
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(raw_results, query)
            
            return filtered_results[:k]
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def search_with_score(self, query: str, k: int = 5, score_threshold: float = 0.8, 
                         filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Similarity search with score threshold filtering.
        Returns documents with their similarity scores, filtered by threshold.
        Lower scores = more similar (cosine distance).
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            # Get results with scores
            if filters:
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query, 
                    k=k*2,
                    filter=filters
                )
            else:
                results_with_scores = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            # Filter by score threshold (lower score = more similar for cosine distance)
            filtered_results = [
                (doc, score) for doc, score in results_with_scores 
                if score < score_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} documents with score < {score_threshold}")
            return filtered_results[:k]
        except Exception as e:
            logger.error(f"Error during score-based search: {str(e)}")
            return []
    
    def _filter_and_rank_results(self, results: List[Document], query: str) -> List[Document]:
        """Filter and rank results based on quality and relevance."""
        if not results:
            return results
        
        scored_results = []
        query_lower = query.lower()
        
        for doc in results:
            score = 0
            content = doc.page_content.lower()
            
            # Relevance scoring
            query_words = query_lower.split()
            word_matches = sum(1 for word in query_words if word in content)
            score += word_matches * 2
            
            # Content quality scoring
            if len(doc.page_content) > 100:  # Prefer substantial content
                score += 1
            if len(doc.page_content) < 50:   # Penalize very short content
                score -= 1
            
            # Diversity bonus - prefer content we haven't used recently
            content_hash = doc.metadata.get('content_hash', '')
            if content_hash not in self.used_content_hashes:
                score += 2
            
            # Section diversity bonus
            section = doc.metadata.get('section', 'general')
            section_count = self.section_usage_count.get(section, 0)
            if section_count < 3:  # Prefer less-used sections
                score += 1
            
            scored_results.append((score, doc))
        
        # Sort by score (descending) and return documents
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_results]
    
    def get_unused_documents(self, k: int = 5) -> List[Document]:
        """Get documents that haven't been used recently for question generation with enhanced diversity."""
        try:
            collection = self.client.get_collection(self.collection_name)
            all_data = collection.get()
            
            if not all_data['ids']:
                return []
            
            all_ids = set(all_data['ids'])
            unused_ids = list(all_ids - self.used_chunk_ids)
            
            # Reset tracking if we've used too many chunks
            if len(unused_ids) < k or time.time() - self.last_reset_time > 3600:  # Reset every hour
                logger.info("Resetting usage tracking for better diversity")
                self.used_chunk_ids.clear()
                self.used_content_hashes.clear()
                self.section_usage_count.clear()
                self.last_reset_time = time.time()
                unused_ids = list(all_ids)
            
            # Sample with section diversity
            sampled_ids = self._sample_with_diversity(unused_ids, all_data, k)
            
            # Mark as used
            self.used_chunk_ids.update(sampled_ids)
            
            # Get the documents
            sampled_data = collection.get(ids=sampled_ids)
            
            documents = []
            for i in range(len(sampled_data['ids'])):
                metadata = sampled_data['metadatas'][i] if sampled_data['metadatas'] else {}
                doc = Document(
                    page_content=sampled_data['documents'][i],
                    metadata=metadata
                )
                documents.append(doc)
                
                # Track content hash and section usage
                content_hash = metadata.get('content_hash', '')
                if content_hash:
                    self.used_content_hashes.add(content_hash)
                
                section = metadata.get('section', 'general')
                self.section_usage_count[section] = self.section_usage_count.get(section, 0) + 1
            
            logger.info(f"Retrieved {len(documents)} diverse unused documents.")
            return documents
        except Exception as e:
            logger.error(f"Error getting unused documents: {str(e)}")
            return []
    
    def _sample_with_diversity(self, available_ids: List[str], all_data: Dict, k: int) -> List[str]:
        """Sample documents with section and content diversity."""
        if len(available_ids) <= k:
            return available_ids
        
        # Group by sections
        section_groups = {}
        for i, doc_id in enumerate(available_ids):
            try:
                metadata = all_data['metadatas'][all_data['ids'].index(doc_id)] if all_data['metadatas'] else {}
                section = metadata.get('section', 'general')
                
                if section not in section_groups:
                    section_groups[section] = []
                section_groups[section].append(doc_id)
            except (IndexError, ValueError):
                # Fallback for missing metadata
                if 'general' not in section_groups:
                    section_groups['general'] = []
                section_groups['general'].append(doc_id)
        
        # Sample from each section proportionally
        sampled_ids = []
        sections = list(section_groups.keys())
        random.shuffle(sections)  # Randomize section order
        
        docs_per_section = max(1, k // len(sections))
        remaining = k
        
        for section in sections:
            if remaining <= 0:
                break
                
            section_ids = section_groups[section]
            take = min(docs_per_section, len(section_ids), remaining)
            
            sampled_from_section = random.sample(section_ids, take)
            sampled_ids.extend(sampled_from_section)
            remaining -= take
        
        # Fill remaining slots randomly if needed
        if remaining > 0:
            unused_ids = [id for id in available_ids if id not in sampled_ids]
            if unused_ids:
                additional = random.sample(unused_ids, min(remaining, len(unused_ids)))
                sampled_ids.extend(additional)
        
        return sampled_ids[:k]
    
    def _keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Keyword-based search for hybrid approach.
        Searches for documents containing query terms.
        """
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            # Get all documents
            collection = self.client.get_collection(self.collection_name)
            all_data = collection.get(include=['documents', 'metadatas'])
            
            if not all_data['documents']:
                return []
            
            # Score documents by keyword matches
            scored_docs = []
            for i, (doc_text, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                doc_lower = doc_text.lower()
                matches = sum(1 for word in query_words if word in doc_lower)
                
                if matches > 0:
                    doc = Document(page_content=doc_text, metadata=metadata or {})
                    scored_docs.append((matches, doc))
            
            # Sort by matches and return top k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            keyword_results = [doc for score, doc in scored_docs[:k]]
            
            logger.info(f"Keyword search found {len(keyword_results)} matching documents")
            return keyword_results
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def advanced_hybrid_search(self, query: str, k: int = 5, 
                              filters: Optional[Dict] = None,
                              include_keyword_search: bool = True) -> Dict[str, Any]:
        """
        Advanced hybrid search combining multiple strategies:
        1. Similarity search with optional filtering
        2. Score-based filtering
        3. Keyword search (optional)
        Merges and deduplicates results for best quality.
        """
        try:
            all_docs = []
            seen_content_hashes = set()
            
            # Strategy 1: Similarity search with native filtering
            similarity_results = self.search_similar(query, k=k, filters=filters)
            for doc in similarity_results:
                content_hash = doc.metadata.get('content_hash') or \
                             hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                if content_hash not in seen_content_hashes:
                    all_docs.append(doc)
                    seen_content_hashes.add(content_hash)
            
            # Strategy 2: Score-based search with threshold
            score_results = self.search_with_score(query, k=k, score_threshold=0.7, filters=filters)
            for doc, score in score_results:
                content_hash = doc.metadata.get('content_hash') or \
                             hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                if content_hash not in seen_content_hashes:
                    all_docs.append(doc)
                    seen_content_hashes.add(content_hash)
                    doc.metadata['similarity_score'] = score
            
            # Strategy 3: Keyword search for additional context
            if include_keyword_search:
                keyword_results = self._keyword_search(query, k=k//2)
                for doc in keyword_results:
                    content_hash = doc.metadata.get('content_hash') or \
                                 hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                    if content_hash not in seen_content_hashes:
                        all_docs.append(doc)
                        seen_content_hashes.add(content_hash)
                        doc.metadata['search_type'] = 'keyword'
            
            # Final quality filtering and ranking
            quality_filtered = self._final_quality_filter(all_docs, query)
            
            result = {
                "documents": quality_filtered[:k],
                "total_found": len(quality_filtered),
                "strategies_used": ["similarity", "score_threshold"],
                "query": query
            }
            
            if include_keyword_search:
                result["strategies_used"].append("keyword")
            
            logger.info(f"Advanced hybrid search: {len(quality_filtered)} documents from {len(result['strategies_used'])} strategies")
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced hybrid search: {e}")
            return {"documents": [], "total_found": 0, "error": str(e)}
    
    def get_hybrid_documents(self, query: str, k_similar: int = 3, k_random: int = 2, 
                            filters: Optional[Dict] = None) -> List[Document]:
        """
        Enhanced hybrid search combining similarity, diversity, and quality filtering.
        Now supports native ChromaDB filtering.
        """
        all_docs = []
        seen_content_hashes = set()
        
        # Get high-quality similar documents with optional filtering
        similar_docs = self.search_similar(query, k=k_similar * 2, filters=filters)
        
        # Filter similar docs for diversity
        filtered_similar = []
        for doc in similar_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            if content_hash not in seen_content_hashes and len(filtered_similar) < k_similar:
                filtered_similar.append(doc)
                seen_content_hashes.add(content_hash)
                
                # Update metadata with content hash if not present
                if 'content_hash' not in doc.metadata:
                    doc.metadata['content_hash'] = content_hash
        
        all_docs.extend(filtered_similar)
        
        # Get diverse documents to broaden the context
        diverse_docs = self.get_unused_documents(k=k_random * 2)
        
        # Filter diverse docs for uniqueness
        filtered_diverse = []
        for doc in diverse_docs:
            content_hash = doc.metadata.get('content_hash') or hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            if content_hash not in seen_content_hashes and len(filtered_diverse) < k_random:
                filtered_diverse.append(doc)
                seen_content_hashes.add(content_hash)
        
        all_docs.extend(filtered_diverse)
        
        # Final quality check and ranking
        quality_filtered = self._final_quality_filter(all_docs, query)
        
        logger.info(f"Retrieved {len(quality_filtered)} high-quality hybrid documents ({len(filtered_similar)} similar + {len(filtered_diverse)} diverse)")
        return quality_filtered
    
    def _final_quality_filter(self, docs: List[Document], query: str) -> List[Document]:
        """Final quality filtering and ranking of documents."""
        if not docs:
            return docs
        
        # Remove very short or repetitive content
        quality_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            
            # Skip very short content
            if len(content) < 50:
                continue
                
            # Skip mostly repetitive content
            words = content.split()
            unique_words = set(words)
            if len(words) > 0 and len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                continue
                
            quality_docs.append(doc)
        
        return quality_docs
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive metadata and performance metrics about the ChromaDB collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            doc_count = collection.count()
            
            # Get collection metadata
            collection_metadata = collection.metadata if hasattr(collection, 'metadata') else {}
            
            # Get section distribution
            all_data = collection.get()
            section_dist = {}
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    section = metadata.get('section', 'general') if metadata else 'general'
                    section_dist[section] = section_dist.get(section, 0) + 1
            
            return {
                "document_count": doc_count,
                "section_distribution": section_dist,
                "used_chunks": len(self.used_chunk_ids),
                "used_content_hashes": len(self.used_content_hashes),
                "collection_metadata": collection_metadata,
                "hnsw_config": {
                    "space": collection_metadata.get("hnsw:space", "l2"),
                    "M": collection_metadata.get("hnsw:M", 16),
                    "construction_ef": collection_metadata.get("hnsw:construction_ef", 200),
                    "search_ef": collection_metadata.get("hnsw:search_ef", 50),
                    "batch_size": collection_metadata.get("hnsw:batch_size", 100),
                    "sync_threshold": collection_metadata.get("hnsw:sync_threshold", 1000)
                },
                "performance_metrics": {
                    "diversity_tracking_enabled": True,
                    "last_reset_time": self.last_reset_time,
                    "time_since_reset": time.time() - self.last_reset_time
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"document_count": 0, "error": str(e)}

    def reset_collection(self) -> bool:
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_vector_store()
            
            # Reset tracking
            self.used_chunk_ids.clear()
            self.used_content_hashes.clear()
            self.section_usage_count.clear()
            self.last_reset_time = time.time()
            
            logger.info("ChromaDB collection has been reset with enhanced tracking.")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False 

    def expand_query(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Expand query with domain-specific synonyms.
        Returns list of query variations for parallel search.
        """
        query_variations = [query]  # Always include original
        query_lower = query.lower()
        
        for term, synonyms in self.query_synonyms.items():
            if term.lower() in query_lower:
                # Create variations by replacing with synonyms
                for synonym in synonyms[:2]:  # Use top 2 synonyms
                    if synonym.lower() != term.lower():
                        variation = query_lower.replace(term.lower(), synonym.lower())
                        if variation not in [v.lower() for v in query_variations]:
                            query_variations.append(variation)
                        
                        if len(query_variations) >= max_variations:
                            break
                
                if len(query_variations) >= max_variations:
                    break
        
        return query_variations[:max_variations]
    
    async def search_similar_async(self, query: str, k: int = 5, 
                                   filters: Optional[Dict] = None) -> List[Document]:
        """
        Async version of similarity search for parallel execution.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                self.search_similar,
                query,
                k,
                filters
            )
        return result
    
    async def parallel_multi_query_search(self, queries: List[str], k: int = 5,
                                         filters: Optional[Dict] = None) -> Dict[str, List[Document]]:
        """
        Execute multiple search queries in parallel for speed.
        Returns dict mapping query to results.
        """
        tasks = [
            self.search_similar_async(query, k, filters)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {query: docs for query, docs in zip(queries, results)}
    
    def merge_and_deduplicate_results(self, results_dict: Dict[str, List[Document]], 
                                     max_results: int = 10) -> List[Document]:
        """
        Merge results from multiple queries and deduplicate by content_hash.
        Ranks by frequency of appearance across queries.
        """
        # Track document frequency and first occurrence
        doc_frequency = {}
        doc_objects = {}
        
        for query, docs in results_dict.items():
            for rank, doc in enumerate(docs):
                content_hash = doc.metadata.get('content_hash') or \
                             hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                
                if content_hash not in doc_frequency:
                    doc_frequency[content_hash] = {
                        'count': 0,
                        'min_rank': rank,
                        'doc': doc
                    }
                    doc_objects[content_hash] = doc
                
                doc_frequency[content_hash]['count'] += 1
                doc_frequency[content_hash]['min_rank'] = min(
                    doc_frequency[content_hash]['min_rank'], 
                    rank
                )
        
        # Sort by: 1) frequency (more queries found it), 2) best rank
        sorted_docs = sorted(
            doc_frequency.items(),
            key=lambda x: (x[1]['count'], -x[1]['min_rank']),
            reverse=True
        )
        
        # Return top documents
        return [doc_objects[content_hash] for content_hash, _ in sorted_docs[:max_results]] 