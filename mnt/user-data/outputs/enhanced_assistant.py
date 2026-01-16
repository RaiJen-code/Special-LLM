"""
Enhanced Voice Assistant with Improved RAG System
Designed for Jetson Orin Nano 8GB
Version: 2.0 - Enhanced for Research and Testing

Features:
- Improved RAG with PDF document processing
- Comprehensive logging for performance analysis
- Flexible configuration for testing
- Better error handling and retry mechanisms
"""

import whisper
import requests
import os
import sounddevice as sd
import numpy as np
import tempfile
import wave
import faiss
from sentence_transformers import SentenceTransformer
import torch
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

class Config:
    """Centralized configuration for the assistant"""
    
    # Disable HuggingFace telemetry for fully offline operation
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # Device configuration
    DEVICE = "cpu"  # More stable for Jetson Orin Nano
    
    # Model configurations
    WHISPER_MODEL = "tiny"  # Options: tiny, base, small
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    # Ollama configuration
    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    OLLAMA_MODEL = "llama3.2:3b"
    
    # Conversation settings
    MAX_HISTORY = 5
    MAX_TOKENS = 150  # Increased for better responses
    
    # Audio settings
    AUDIO_DURATION = 5  # seconds
    SAMPLE_RATE = 16000
    AUDIO_DEVICE = 25  # C270 HD WEBCAM
    
    # RAG settings
    RAG_TOP_K = 3  # Number of documents to retrieve
    RAG_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score
    RAG_CHUNK_SIZE = 500  # Characters per chunk
    RAG_CHUNK_OVERLAP = 50  # Overlap between chunks
    
    # File paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BIP_SOUND = os.path.join(CURRENT_DIR, "assets/bip.wav")
    BIP2_SOUND = os.path.join(CURRENT_DIR, "assets/bip2.wav")
    PIPER_MODEL = "/usr/local/share/piper/models/en_US-lessac-medium.onnx"
    PIPER_BIN = "/home/rangga/piper/build/piper"
    
    # Logging configuration
    LOG_DIR = os.path.join(CURRENT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # System prompt
    SYSTEM_PROMPT = """You are an intelligent AI assistant specialized in electronics, 
microcontrollers, embedded systems, and IoT technology. You provide accurate, 
concise, and helpful responses. When answering:
1. Be precise and technical when needed
2. Give practical examples when helpful
3. Keep responses under 3-4 sentences for voice interaction
4. Cite your sources from the knowledge base when available
5. If you don't know something, say so clearly"""

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging system"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(Config.LOG_DIR, f'assistant_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('VoiceAssistant')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class PerformanceTracker:
    """Track and log performance metrics"""
    
    def __init__(self):
        self.metrics = []
        self.current_query = {}
    
    def start_query(self, query_text: str):
        """Start tracking a new query"""
        self.current_query = {
            'query': query_text,
            'timestamp': datetime.now().isoformat(),
            'timings': {},
            'errors': []
        }
    
    def log_timing(self, component: str, duration: float):
        """Log timing for a component"""
        self.current_query['timings'][component] = duration
        logger.debug(f"{component} took {duration:.3f}s")
    
    def log_error(self, error: str):
        """Log an error"""
        self.current_query['errors'].append(error)
        logger.error(error)
    
    def log_result(self, response: str, rag_used: bool, docs_retrieved: int):
        """Log the final result"""
        self.current_query['response'] = response
        self.current_query['rag_used'] = rag_used
        self.current_query['docs_retrieved'] = docs_retrieved
        self.current_query['total_time'] = sum(self.current_query['timings'].values())
        
        # Add to metrics list
        self.metrics.append(self.current_query.copy())
        
        # Save to file periodically
        if len(self.metrics) % 5 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(Config.LOG_DIR, f'metrics_{timestamp}.json')
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")

# ============================================================================
# ENHANCED VECTOR DATABASE WITH BETTER RAG
# ============================================================================

class EnhancedVectorDatabase:
    """Improved vector database with document processing and scoring"""
    
    def __init__(self, embedding_model: SentenceTransformer, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
        self.metadata = []  # Store metadata about each chunk
        self.embedding_model = embedding_model
        logger.info("Enhanced Vector Database initialized")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:  # If found in latter half
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def add_documents(self, docs: List[str], source: str = "manual"):
        """Add documents with chunking and metadata"""
        for doc_idx, doc in enumerate(docs):
            chunks = self.chunk_text(doc, Config.RAG_CHUNK_SIZE, Config.RAG_CHUNK_OVERLAP)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Create metadata
                metadata = {
                    'source': source,
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'text': chunk
                }
                
                self.documents.append(chunk)
                self.metadata.append(metadata)
        
        # Re-index all documents
        if self.documents:
            embeddings = self.embedding_model.encode(self.documents)
            self.index = faiss.IndexFlatL2(Config.EMBEDDING_DIM)
            self.index.add(np.array(embeddings, dtype=np.float32))
            
            logger.info(f"Added {len(self.documents)} chunks from {source}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search with similarity scores and metadata"""
        if not self.documents:
            logger.warning("No documents in database")
            return []
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0].astype(np.float32)
        
        # Search
        distances, indices = self.index.search(
            np.array([query_embedding]), 
            min(top_k, len(self.documents))
        )
        
        # Prepare results with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert L2 distance to similarity score (0-1, higher is better)
            similarity = 1 / (1 + dist)
            
            if similarity >= Config.RAG_SIMILARITY_THRESHOLD:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'distance': float(dist)
                })
        
        return results
    
    def load_from_pdf(self, pdf_path: str):
        """Load and process PDF document"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                # Add as documents with source tracking
                self.add_documents([full_text], source=os.path.basename(pdf_path))
                logger.info(f"Successfully loaded PDF: {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_chunks': len(self.documents),
            'total_size_chars': sum(len(doc) for doc in self.documents),
            'sources': list(set(m['source'] for m in self.metadata))
        }

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

logger.info("="*60)
logger.info("Initializing Models...")
logger.info("="*60)

# Initialize embedding model
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
logger.info(f"✓ Embedding model loaded: {Config.EMBEDDING_MODEL}")

# Initialize Whisper ASR
whisper_model = whisper.load_model(Config.WHISPER_MODEL, device=Config.DEVICE)
logger.info(f"✓ Whisper model loaded: {Config.WHISPER_MODEL}")

# Initialize Vector Database
db = EnhancedVectorDatabase(embedding_model, dim=Config.EMBEDDING_DIM)

# Load documents from PDFs if available
pdf_dir = Path(Config.CURRENT_DIR)
for pdf_file in ['referensi.pdf', 'jurnal_tambahan.pdf']:
    pdf_path = pdf_dir / pdf_file
    if pdf_path.exists():
        logger.info(f"Loading {pdf_file}...")
        db.load_from_pdf(str(pdf_path))

# Add default documents if no PDFs loaded
if len(db.documents) == 0:
    default_docs = [
        "The Jetson Nano is a compact, powerful computer designed by NVIDIA for AI applications at the edge.",
        "Developers can create AI assistants in under 100 lines of Python code using open-source libraries.",
        "Retrieval Augmented Generation enhances AI responses by combining language models with external knowledge bases.",
        "Arduino is an open-source electronics platform based on easy-to-use hardware and software.",
        "A resistor is a passive component that limits or regulates the flow of electrical current in an electronic circuit.",
        "LED stands for Light Emitting Diode, a semiconductor device that emits light when current flows through it.",
        "PWM (Pulse Width Modulation) is a technique used to control the amount of power delivered to a load by rapidly switching it on and off.",
        "UART (Universal Asynchronous Receiver-Transmitter) is a serial communication protocol that uses two wires (TX and RX) for point-to-point communication.",
        "I2C (Inter-Integrated Circuit) is a multi-master, multi-slave serial communication protocol that uses two wires (SDA and SCL) and allows multiple devices on the same bus.",
        "An interrupt is a signal to the processor that temporarily stops the current program execution to handle a more urgent task.",
    ]
    db.add_documents(default_docs, source="default_knowledge")

# Print database statistics
stats = db.get_stats()
logger.info("Database Statistics:")
for key, value in stats.items():
    logger.info(f"  {key}: {value}")

# Initialize Performance Tracker
perf_tracker = PerformanceTracker()

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

conversation_history = []

def add_to_history(user_msg: str, assistant_msg: str):
    """Add exchange to conversation history"""
    conversation_history.append({
        "user": user_msg,
        "assistant": assistant_msg
    })
    
    # Keep only recent history
    if len(conversation_history) > Config.MAX_HISTORY:
        conversation_history.pop(0)

def get_history_context() -> str:
    """Get formatted conversation history"""
    if not conversation_history:
        return ""
    
    history_text = "Recent conversation:\n"
    for exchange in conversation_history[-3:]:
        history_text += f"User: {exchange['user']}\n"
        history_text += f"Assistant: {exchange['assistant']}\n"
    
    return history_text

# ============================================================================
# AUDIO FUNCTIONS
# ============================================================================

def play_sound(sound_file: str):
    """Play audio notification"""
    try:
        os.system(f"aplay -D plughw:0,0 {sound_file} 2>/dev/null")
    except Exception as e:
        logger.warning(f"Could not play sound: {e}")

def record_audio(filename: str, duration: int = None, fs: int = None) -> float:
    """Record audio and return recording time"""
    if duration is None:
        duration = Config.AUDIO_DURATION
    if fs is None:
        fs = Config.SAMPLE_RATE
    
    start_time = time.time()
    play_sound(Config.BIP_SOUND)
    logger.info(f"Recording for {duration} seconds...")
    
    try:
        audio = sd.rec(
            int(duration * fs), 
            samplerate=fs, 
            channels=1, 
            dtype='int16', 
            device=Config.AUDIO_DEVICE
        )
        sd.wait()
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        
        play_sound(Config.BIP2_SOUND)
        recording_time = time.time() - start_time
        logger.info(f"Recording completed in {recording_time:.2f}s")
        
        return recording_time
        
    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise

def transcribe_audio(filename: str) -> Tuple[str, float]:
    """Transcribe audio file and return (text, time_taken)"""
    start_time = time.time()
    
    try:
        result = whisper_model.transcribe(filename, language="en", fp16=False)
        transcription = result['text'].strip()
        transcription_time = time.time() - start_time
        
        logger.info(f"Transcribed: '{transcription}' ({transcription_time:.2f}s)")
        return transcription, transcription_time
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise

def text_to_speech(text: str) -> float:
    """Convert text to speech and return processing time"""
    start_time = time.time()
    response_file = os.path.join(Config.CURRENT_DIR, "response.wav")
    
    try:
        # Escape quotes in text
        safe_text = text.replace('"', '\\"').replace("'", "\\'")
        
        cmd = f'echo "{safe_text}" | {Config.PIPER_BIN} --model {Config.PIPER_MODEL} --output_file {response_file} && aplay -D plughw:0,0 {response_file} 2>/dev/null'
        os.system(cmd)
        
        tts_time = time.time() - start_time
        logger.info(f"TTS completed in {tts_time:.2f}s")
        
        return tts_time
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return 0.0

# ============================================================================
# LLM INTERACTION WITH RAG
# ============================================================================

def check_ollama_server() -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✓ Ollama server running ({len(models)} models available)")
            
            model_names = [m['name'] for m in models]
            if Config.OLLAMA_MODEL in model_names:
                logger.info(f"✓ Using model: {Config.OLLAMA_MODEL}")
            else:
                logger.warning(f"⚠ Model {Config.OLLAMA_MODEL} not found")
                logger.warning(f"  Available: {', '.join(model_names[:3])}")
            
            return True
    except Exception as e:
        logger.error(f"✗ Ollama server not running: {e}")
        return False

def ask_llm_with_rag(query: str, use_rag: bool = True) -> Tuple[str, float, Dict]:
    """
    Query LLM with optional RAG
    Returns: (response_text, inference_time, rag_info)
    """
    start_time = time.time()
    rag_info = {'used': use_rag, 'docs_retrieved': 0, 'docs': []}
    
    # Retrieve relevant documents if RAG is enabled
    context_text = ""
    if use_rag and len(db.documents) > 0:
        retrieved_docs = db.search(query, top_k=Config.RAG_TOP_K)
        rag_info['docs_retrieved'] = len(retrieved_docs)
        rag_info['docs'] = retrieved_docs
        
        if retrieved_docs:
            context_text = "Relevant information from knowledge base:\n"
            for i, doc_info in enumerate(retrieved_docs, 1):
                context_text += f"{i}. {doc_info['text']}\n"
            context_text += "\n"
            
            logger.debug(f"Retrieved {len(retrieved_docs)} documents for RAG")
    
    # Build prompt
    history_text = get_history_context()
    
    full_prompt = f"""{Config.SYSTEM_PROMPT}

{history_text}{context_text}User: {query}
Assistant:"""
    
    # Call Ollama
    data = {
        "model": Config.OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": Config.MAX_TOKENS,
            "top_k": 40,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(
            Config.OLLAMA_URL,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            inference_time = time.time() - start_time
            
            # Add to conversation history
            add_to_history(query, result)
            
            logger.info(f"LLM response generated ({inference_time:.2f}s)")
            return result, inference_time, rag_info
        else:
            error_msg = f"Ollama server error (status {response.status_code})"
            logger.error(error_msg)
            return error_msg, 0.0, rag_info
            
    except requests.exceptions.Timeout:
        error_msg = "Request timed out"
        logger.error(error_msg)
        return error_msg, 0.0, rag_info
    except Exception as e:
        error_msg = f"Error: {str(e)[:50]}"
        logger.error(error_msg)
        return error_msg, 0.0, rag_info

# ============================================================================
# MAIN ASSISTANT LOOP
# ============================================================================

def process_query(query_text: str, use_rag: bool = True) -> Dict:
    """
    Process a single query and return comprehensive results
    This function is designed for both interactive and testing use
    """
    results = {
        'query': query_text,
        'timestamp': datetime.now().isoformat(),
        'timings': {},
        'use_rag': use_rag
    }
    
    perf_tracker.start_query(query_text)
    
    try:
        # Get LLM response
        start = time.time()
        response, llm_time, rag_info = ask_llm_with_rag(query_text, use_rag=use_rag)
        results['llm_time'] = llm_time
        results['response'] = response
        results['rag_info'] = rag_info
        
        perf_tracker.log_timing('llm_inference', llm_time)
        
        # Log result
        perf_tracker.log_result(
            response, 
            rag_info['used'], 
            rag_info['docs_retrieved']
        )
        
        results['total_time'] = time.time() - start
        results['success'] = True
        
        logger.info(f"Query processed successfully")
        
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        logger.error(error_msg)
        perf_tracker.log_error(error_msg)
        results['success'] = False
        results['error'] = str(e)
    
    return results

def run_interactive_mode():
    """Run the assistant in interactive voice mode"""
    logger.info("="*60)
    logger.info("VOICE ASSISTANT - Interactive Mode")
    logger.info("="*60)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Whisper Model: {Config.WHISPER_MODEL}")
    logger.info(f"LLM Model: {Config.OLLAMA_MODEL}")
    logger.info(f"RAG Enabled: {len(db.documents)} chunks loaded")
    logger.info("="*60)
    
    if not check_ollama_server():
        logger.error("⚠ Please start Ollama server first!")
        return
    
    logger.info("\n✓ All systems ready! Listening for voice input...\n")
    logger.info("Press Ctrl+C to stop\n")
    
    while True:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                # Record audio
                start_total = time.time()
                rec_time = record_audio(tmpfile.name)
                perf_tracker.log_timing('recording', rec_time)
                
                # Transcribe
                transcribed_text, trans_time = transcribe_audio(tmpfile.name)
                perf_tracker.log_timing('transcription', trans_time)
                
                # Skip if empty
                if not transcribed_text or len(transcribed_text.strip()) < 2:
                    logger.warning("⚠ No speech detected, listening again...")
                    continue
                
                logger.info(f"You: {transcribed_text}")
                
                # Get response
                response, llm_time, rag_info = ask_llm_with_rag(transcribed_text, use_rag=True)
                perf_tracker.log_timing('llm_inference', llm_time)
                
                logger.info(f"Assistant: {response}")
                
                # Speak response
                if response and not response.startswith("Error:"):
                    tts_time = text_to_speech(response)
                    perf_tracker.log_timing('tts', tts_time)
                
                # Log complete interaction
                total_time = time.time() - start_total
                perf_tracker.log_result(
                    response,
                    rag_info['used'],
                    rag_info['docs_retrieved']
                )
                
                logger.info(f"Total interaction time: {total_time:.2f}s")
                logger.info("-"*60)
                
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Shutting down. Goodbye!")
            perf_tracker.save_metrics()
            break
        except Exception as e:
            logger.error(f"⚠ Error occurred: {e}")
            logger.info("Continuing...")

# ============================================================================
# TESTING INTERFACE
# ============================================================================

def run_text_test(test_queries: List[str], use_rag: bool = True) -> List[Dict]:
    """
    Test mode: process text queries without audio I/O
    Useful for benchmarking and comparison
    """
    logger.info("="*60)
    logger.info("VOICE ASSISTANT - Text Testing Mode")
    logger.info("="*60)
    
    if not check_ollama_server():
        logger.error("⚠ Please start Ollama server first!")
        return []
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n[Query {i}/{len(test_queries)}] {query}")
        result = process_query(query, use_rag=use_rag)
        results.append(result)
        logger.info(f"Response: {result.get('response', 'ERROR')}")
        logger.info(f"Time: {result.get('total_time', 0):.2f}s")
        
        # Small delay between queries
        time.sleep(0.5)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(Config.LOG_DIR, f'test_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTest results saved to: {results_file}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode with sample queries
        test_queries = [
            "What is the main function of a resistor?",
            "Explain how PWM works",
            "Is capacitor used to store charge? True or false?"
        ]
        run_text_test(test_queries, use_rag=True)
    else:
        # Interactive voice mode
        run_interactive_mode()
