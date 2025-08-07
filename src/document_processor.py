"""
Document Processor Module - PDF Text Extraction and Semantic Chunking

This module implements a comprehensive document processing pipeline designed for
efficient text extraction from PDF documents and intelligent semantic chunking.
It provides the foundation for converting unstructured PDF content into searchable
text chunks optimized for vector embedding and retrieval systems.

Processing Pipeline:
1. PDF Text Extraction: Extract raw text with page-level granularity
2. Text Cleaning and Normalization: Remove artifacts and standardize formatting
3. Semantic Chunking: Split text into meaningful chunks with configurable overlap
4. Metadata Preservation: Maintain page numbers, section titles, and positioning
5. Data Serialization: Save processed chunks in JSON format for downstream use

Key Features:
- High-quality PDF text extraction using PyMuPDF
- Intelligent text cleaning with artifact removal
- Semantic chunking with configurable size and overlap
- Page-level metadata preservation
- Progress tracking with rich console interface
- Comprehensive error handling and logging
- Flexible output formats (JSON, in-memory objects)

The module is designed to handle various PDF types including:
- Technical documentation with complex formatting
- Multi-page documents with headers and footers
- Documents with tables, images, and mixed content
- Large documents requiring efficient memory management

Author: Segun Oni
Version: 1.0.0
"""

import fitz  # PyMuPDF - high-performance PDF processing library
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Third-party imports for enhanced user interface and configuration
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
import argparse
from dotenv import load_dotenv

# Load environment variables for configuration
load_dotenv()

# Configure logging for document processing operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize console for rich text output and progress tracking
console = Console()

@dataclass
class TextChunk:
    """
    Represents a text chunk with comprehensive metadata for vector processing.
    
    This dataclass encapsulates a single text chunk along with its metadata,
    providing all necessary information for downstream processing including
    vector embedding generation and similarity search.
    
    Attributes:
        id: Unique identifier for the chunk within the document
        text: The actual text content of the chunk
        start_char: Starting character position in the original document
        end_char: Ending character position in the original document
        length: Length of the text chunk in characters
        page_number: Page number where the chunk originated (if available)
        section_title: Title of the section containing the chunk (if available)
        chunk_type: Type of chunk (e.g., "text", "header", "table")
    """
    id: int
    text: str
    start_char: int
    end_char: int
    length: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str = "text"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the TextChunk to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the TextChunk suitable for JSON encoding
        """
        return asdict(self)

class DocumentProcessor:
    """
    Handles document processing and semantic chunking for PDF documents.
    
    This class provides a complete pipeline for processing PDF documents from
    raw text extraction to semantic chunking. It implements intelligent text
    processing algorithms that preserve document structure while creating
    optimal chunks for vector embedding and retrieval systems.
    
    The processor supports configurable chunk sizes and overlap parameters,
    enabling fine-tuning for different document types and use cases. It also
    provides comprehensive progress tracking and error handling for robust
    document processing workflows.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document processor with configurable chunking parameters.
        
        Args:
            chunk_size: Target chunk size in characters (default: from environment or 512)
                       Larger chunks provide more context but may be less precise
            chunk_overlap: Overlap between consecutive chunks in characters (default: from environment or 50)
                          Overlap helps maintain context across chunk boundaries
        """
        # Use environment variables if not provided
        if chunk_size is None:
            import os
            self.chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
        else:
            self.chunk_size = chunk_size
            
        if chunk_overlap is None:
            import os
            self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
        else:
            self.chunk_overlap = chunk_overlap
        self.chunks: List[TextChunk] = []  # Store processed chunks in memory
        
        # Log initialization with configuration parameters
        console.print(f"DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})", style="blue")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF with page-level granularity and progress tracking.
        
        This method uses PyMuPDF (fitz) to extract text from each page of the PDF,
        maintaining page boundaries and providing detailed progress feedback.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Extracted text as a single string with page markers
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: For other PDF processing errors
        """
        try:
            # Open PDF document using PyMuPDF
            doc = fitz.open(pdf_path) # type: ignore
            text_parts = []
            total_pages = len(doc)
            
            # Set up progress tracking with rich console interface
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Extracting text from PDF...", total=total_pages)
                
                # Process each page individually to maintain structure
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Add page marker for reference and structure preservation
                    text_parts.append(f"[PAGE_{page_num + 1}]\n{page_text}")
                    progress.update(task, advance=1)
            
            # Close document after extraction is complete
            doc.close()
            full_text = "\n\n".join(text_parts)
            
            # Log extraction statistics for monitoring
            console.print(f"Extracted {len(full_text)} characters from {total_pages} pages", style="green")
            return full_text
            
        except Exception as e:
            # Log error and re-raise for proper error handling upstream
            console.print(f"Error extracting text from PDF: {e}", style="red")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text to improve quality for processing.
        
        This method applies a series of text cleaning operations to remove
        artifacts, standardize formatting, and prepare text for semantic
        chunking. It handles common PDF extraction issues like excessive
        whitespace, page markers, and formatting inconsistencies.
        
        Args:
            text: Raw extracted text from PDF
            
        Returns:
            Cleaned and normalized text ready for chunking
        """
        console.print("Cleaning text...", style="blue")
        
        # Remove page markers completely - these are internal markers from extraction
        text = re.sub(r'\[PAGE_\d+\]\s*', '', text)
        
        # Remove excessive whitespace - normalize to single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers - common PDF artifacts
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up special characters but preserve important punctuation
        # This removes unwanted characters while keeping essential punctuation marks
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']', '', text)
        
        # Normalize spacing around punctuation - remove extra spaces before punctuation
        text = re.sub(r'\s+([\.\,\!\?\;\:])', r'\1', text)
        
        # Remove multiple periods - common artifact from PDF extraction
        text = re.sub(r'\.{2,}', '.', text)
        
        # Log cleaning results for monitoring
        console.print(f"Text cleaned: {len(text)} characters remaining", style="green")
        return text.strip()
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract document sections based on headers and structural patterns.
        
        This method analyzes the document structure to identify sections based on
        common header patterns. It looks for all-caps headers, numbered sections,
        and title-case patterns to create a logical document structure.
        
        Args:
            text: Clean text to analyze for section structure
            
        Returns:
            List of sections with metadata including title, content, and page number
        """
        sections = []
        # Initialize with a default section for content before first header
        current_section = {"title": "Introduction", "content": "", "page": 1}
        
        # Split text into lines for analysis
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers using multiple pattern matching strategies
            # Look for all caps headers, numbered sections, and title case patterns
            if (re.match(r'^[A-Z\s]{5,}$', line) or  # All caps headers (5+ characters)
                re.match(r'^\d+\.\s+[A-Z]', line) or  # Numbered sections (1. Title)
                re.match(r'^[A-Z][a-z]+\s+[A-Z]', line)):  # Title case headers
                
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section with detected header
                current_section = {
                    "title": line,
                    "content": "",
                    "page": self._extract_page_number(line)
                }
            else:
                # Add line to current section content
                current_section["content"] += line + " "
        
        # Add final section if it contains content
        if current_section["content"].strip():
            sections.append(current_section)
        
        # Log section extraction results
        console.print(f"Extracted {len(sections)} sections", style="green")
        return sections
    
    def _extract_page_number(self, text: str) -> int:
        """
        Extract page number from text containing page markers.
        
        This helper method searches for page markers in the format [PAGE_X]
        and extracts the page number for metadata purposes.
        
        Args:
            text: Text that may contain page markers
            
        Returns:
            Page number as integer, defaults to 1 if no marker found
        """
        match = re.search(r'\[PAGE_(\d+)\]', text)
        return int(match.group(1)) if match else 1
    
    def _semantic_chunking(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Create semantic chunks with intelligent splitting based on document structure.
        
        This method implements intelligent chunking that respects natural text
        boundaries like sentences and paragraphs. It attempts to break text at
        meaningful points rather than arbitrary character positions, which
        improves the quality of downstream processing like vector embedding.
        
        Args:
            text: The text to chunk into semantic units
            chunk_size: Target chunk size in characters
            overlap: Overlap between consecutive chunks in characters
        
        Returns:
            List of TextChunk objects with semantic boundaries and metadata
        """
        console.print("Creating semantic chunks...", style="blue")
        
        # First, try to split by sections for better semantic coherence
        sections = self.extract_sections(text)
        
        chunks = []
        chunk_id = 0
        
        # Process each section individually to maintain semantic boundaries
        for section in sections:
            section_text = section["content"]
            section_title = section["title"]
            page_number = section["page"]
            
            # Split section into chunks using semantic chunking algorithm
            section_chunks = self._chunk_text_semantically(
                section_text, 
                chunk_id, 
                section_title, 
                page_number,
                chunk_size,
                overlap
            )
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        # If no sections found, fall back to simple chunking
        # This ensures we always produce chunks even for unstructured documents
        if not chunks:
            chunks = self._chunk_text_simple(text, 0, chunk_size, overlap)
        
        # Log chunking results for monitoring
        console.print(f"Created {len(chunks)} semantic chunks", style="green")
        return chunks
    
    def _chunk_text_semantically(self, text: str, start_id: int, section_title: str, page_number: int, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Create chunks with semantic boundaries for optimal text processing.
        
        This method implements intelligent chunking that respects natural text
        boundaries like sentences and paragraphs. It attempts to break text at
        meaningful points rather than arbitrary character positions, which
        improves the quality of downstream processing like vector embedding.
        
        Args:
            text: The text to chunk into semantic units
            start_id: The starting ID for the chunks (for unique identification)
            section_title: The title of the section containing the text
            page_number: The page number where the text originated
            chunk_size: Target chunk size in characters
            overlap: Overlap between consecutive chunks in characters
        
        Returns:
            List of TextChunk objects with semantic boundaries and metadata
        """
        chunks = []
        current_pos = 0
        
        # Process text in chunks while respecting semantic boundaries
        while current_pos < len(text):
            end_pos = current_pos + chunk_size
            
            # Try to find a good break point for semantic coherence
            if end_pos < len(text):
                # Look for natural text boundaries within the chunk range
                sentence_end = text.rfind('.', current_pos, end_pos)
                paragraph_end = text.rfind('\n\n', current_pos, end_pos)
                
                # Prefer paragraph breaks, then sentence breaks for better semantics
                # Only break at these points if they're within 70% of the target size
                if paragraph_end > current_pos + chunk_size * 0.7:
                    end_pos = paragraph_end + 2  # Include the paragraph break
                elif sentence_end > current_pos + chunk_size * 0.7:
                    end_pos = sentence_end + 1   # Include the sentence ending
            
            # Extract the chunk text and clean it
            chunk_text = text[current_pos:end_pos].strip()
            
            # Only create chunks for non-empty text
            if chunk_text:
                chunk = TextChunk(
                    id=start_id + len(chunks),
                    text=chunk_text,
                    start_char=current_pos,
                    end_char=end_pos,
                    length=len(chunk_text),
                    page_number=page_number,
                    section_title=section_title,
                    chunk_type="semantic"
                )
                chunks.append(chunk)
            
            # Move position with overlap for context preservation
            # This ensures consecutive chunks have some shared context
            current_pos = end_pos - overlap
            if current_pos >= len(text):
                break
        
        return chunks
    
    def _chunk_text_simple(self, text: str, start_id: int, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Simple fixed-size chunking as fallback for unstructured text.
        
        This method provides a basic chunking strategy when semantic chunking
        is not possible or when no document structure is detected. It creates
        fixed-size chunks with overlap for context preservation.
        
        Args:
            text: The text to chunk into fixed-size pieces
            start_id: The starting ID for the chunks (for unique identification)
            chunk_size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks in characters
        
        Returns:
            List of TextChunk objects with fixed-size boundaries
        """
        chunks = []
        current_pos = 0
        
        # Handle invalid chunk sizes gracefully
        if chunk_size <= 0:
            # If chunk size is invalid, create a single chunk with the entire text
            if text.strip():
                chunk = TextChunk(
                    id=start_id,
                    text=text.strip(),
                    start_char=0,
                    end_char=len(text),
                    length=len(text.strip()),
                    chunk_type="simple"
                )
                chunks.append(chunk)
            return chunks
        
        # Process text in fixed-size chunks with overlap
        while current_pos < len(text):
            end_pos = current_pos + chunk_size
            chunk_text = text[current_pos:end_pos].strip()
            
            # Only create chunks for non-empty text
            if chunk_text:
                chunk = TextChunk(
                    id=start_id + len(chunks),
                    text=chunk_text,
                    start_char=current_pos,
                    end_char=end_pos,
                    length=len(chunk_text),
                    chunk_type="simple"
                )
                chunks.append(chunk)
            
            # Move position with overlap for context preservation
            current_pos = end_pos - overlap
            if current_pos >= len(text):
                break
        
        return chunks
    
    def _simple_chunking(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Simple fixed-size chunking as fallback for unstructured text.
        
        This method provides a basic chunking strategy when semantic chunking
        is not possible or when no document structure is detected. It creates
        fixed-size chunks with overlap for context preservation.
        
        Args:
            text: The text to chunk into fixed-size pieces
            chunk_size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks in characters
        
        Returns:
            List of TextChunk objects with fixed-size boundaries
        """
        chunks = []
        current_pos = 0
        
        # Handle invalid chunk sizes gracefully
        if chunk_size <= 0:
            # If chunk size is invalid, create a single chunk with the entire text
            if text.strip():
                chunk = TextChunk(
                    id=0,
                    text=text.strip(),
                    start_char=0,
                    end_char=len(text),
                    length=len(text.strip()),
                    chunk_type="simple"
                )
                chunks.append(chunk)
            return chunks
        
        # Process text in fixed-size chunks with overlap
        while current_pos < len(text):
            end_pos = current_pos + chunk_size
            chunk_text = text[current_pos:end_pos].strip()
            
            # Only create chunks for non-empty text
            if chunk_text:
                chunk = TextChunk(
                    id=len(chunks),
                    text=chunk_text,
                    start_char=current_pos,
                    end_char=end_pos,
                    length=len(chunk_text),
                    chunk_type="simple"
                )
                chunks.append(chunk)
            
            # Move position with overlap for context preservation
            current_pos = end_pos - overlap
            if current_pos >= len(text):
                break
        
        return chunks
    
    def process_document(self, pdf_path: str, output_path: Optional[str] = None) -> List[TextChunk]:
        """
        Complete document processing pipeline from PDF to semantic chunks.
        
        This method orchestrates the entire document processing workflow,
        including text extraction, cleaning, chunking, and optional saving.
        It provides a convenient single-method interface for processing
        PDF documents into searchable text chunks.
        
        Args:
            pdf_path: Path to the PDF file to process
            output_path: Optional path to save processed chunks as JSON
            
        Returns:
            List of processed TextChunk objects ready for vector embedding
        """
        console.print(f"Processing document: {pdf_path}", style="bold blue")
        
        # Step 1: Extract text from PDF with page-level granularity
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Clean and normalize extracted text
        clean_text = self._clean_text(raw_text)
        
        # Step 3: Create semantic chunks with metadata
        self.chunks = self._semantic_chunking(clean_text, self.chunk_size, self.chunk_overlap)
        
        # Step 4: Save chunks to JSON if output path provided
        if output_path:
            self.save_chunks(self.chunks, output_path)
        
        # Step 5: Display processing statistics
        self._show_statistics()
        
        return self.chunks
    
    def save_chunks(self, chunks: List[TextChunk], output_path: str) -> None:
        """
        Save processed chunks to JSON file with metadata.
        
        This method serializes the processed chunks along with processing
        metadata to a JSON file for later use. The output includes chunk
        content, positioning information, and processing parameters.
        
        Args:
            chunks: List of TextChunk objects to save
            output_path: Path where the JSON file should be saved
        """
        output_file = Path(output_path)
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare comprehensive metadata for serialization
        chunk_data = {
            'metadata': {
                'total_chunks': len(chunks),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'chunk_types': list(set(chunk.chunk_type for chunk in chunks))
            },
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        
        # Write chunks to JSON file with proper encoding
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"Chunks saved to: {output_path}", style="green")
    
    def _show_statistics(self) -> None:
        """
        Display comprehensive processing statistics in a formatted panel.
        
        This method calculates and displays key metrics about the processing
        results, including chunk counts, character statistics, and chunk type
        distribution. It provides valuable insights into the processing quality.
        """
        if not self.chunks:
            return
        
        total_chars = sum(chunk.length for chunk in self.chunks)
        avg_chunk_size = total_chars / len(self.chunks)
        
        chunk_types = {}
        for chunk in self.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        stats_panel = Panel(
            f"Total Chunks: {len(self.chunks)}\n"
            f"Total Characters: {total_chars:,}\n"
            f"Average Chunk Size: {avg_chunk_size:.1f} chars\n"
            f"Chunk Types: {chunk_types}",
            title="Processing Statistics",
            border_style="green"
        )
        console.print(stats_panel)
    
    def get_sample_chunks(self, num_samples: int = 3) -> List[TextChunk]:
        """
        Get sample chunks for preview and debugging purposes.
        
        This method returns a subset of processed chunks that can be used
        for previewing the processing results or debugging chunk quality.
        
        Args:
            num_samples: Number of sample chunks to return (default: 3)
            
        Returns:
            List of sample TextChunk objects for preview
        """
        return self.chunks[:num_samples] if self.chunks else []

def main():
    """
    Main function for command-line usage of the document processor.
    
    This function provides a command-line interface for processing PDF documents
    into semantic chunks. It supports various configuration options and provides
    sample output for quality verification.
    """
    parser = argparse.ArgumentParser(description="Process PDF documents into chunks")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file for chunks")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in characters")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample chunks to show")
    
    args = parser.parse_args()
    
    try:
        processor = DocumentProcessor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        chunks = processor.process_document(args.pdf_path, args.output)
        
        # Show sample chunks
        if chunks:
            console.print(f"\nSample chunks:", style="bold")
            for i, chunk in enumerate(processor.get_sample_chunks(args.samples), 1):
                console.print(f"\nChunk {i}:")
                console.print(f"  Type: {chunk.chunk_type}")
                console.print(f"  Length: {chunk.length} chars")
                if chunk.section_title:
                    console.print(f" Section: {chunk.section_title}")
                console.print(f"  Text: {chunk.text[:100]}...")
        
        return 0
        
    except Exception as e:
        console.print(f"Processing failed: {e}", style="red")
        return 1

if __name__ == "__main__":
    main() 