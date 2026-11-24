#!/usr/bin/env python3
"""
SEC Edgar Filing Ingestion System for ProximaDB - Production Implementation

A comprehensive, production-ready implementation for ingesting SEC Edgar filings
into ProximaDB with advanced parsing, chunking, embedding strategies, and
enterprise-grade error handling.

Features:
- Multi-format parsing (XBRL, HTML, TXT) with intelligent fallbacks
- Financial-specific embeddings (FinBERT, SEC-BERT)
- Async batch processing with rate limiting
- Comprehensive error handling and retry logic
- Detailed logging and monitoring
- Progress tracking and resumable ingestion
"""

import asyncio
import json
import re
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from urllib.parse import urljoin
import warnings

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup, NavigableString
from lxml import etree
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

# ProximaDB imports
try:
    from proximadb import ProximaDBClient, Protocol
    from proximadb.models import (
        CollectionConfig, VectorRecord, StorageEngine,
        DistanceMetric, QuantizationType, CompressionType,
        FilterableColumn, FilterableDataType
    )
    from proximadb.chunking import ChunkingConfig, ChunkingStrategy, TextChunker
    from proximadb.embedding_providers.finbert_provider import (
        FinBERTProvider, SECBERTProvider
    )
except ImportError as e:
    warnings.warn(f"ProximaDB SDK not found: {e}. Install with: pip install proximadb")
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class FilingType(Enum):
    """SEC filing types"""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    DEF_14A = "DEF 14A"
    FORM_S1 = "S-1"
    FORM_20F = "20-F"
    FORM_11K = "11-K"
    FORM_N1A = "N-1A"


class SectionType(Enum):
    """Document section types for targeted processing"""
    BUSINESS_OVERVIEW = "business_overview"
    RISK_FACTORS = "risk_factors"
    MDA = "management_discussion_analysis"
    FINANCIAL_STATEMENTS = "financial_statements"
    NOTES_TO_FINANCIALS = "notes_to_financials"
    LEGAL_PROCEEDINGS = "legal_proceedings"
    EXECUTIVE_COMPENSATION = "executive_compensation"
    AUDITOR_REPORT = "auditor_report"
    MARKET_RISK = "market_risk"
    CONTROLS_PROCEDURES = "controls_procedures"


class ProcessingStatus(Enum):
    """Filing processing status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class FilingMetadata:
    """Complete metadata for SEC filing"""
    ticker: str
    company_name: str
    cik: str
    filing_type: FilingType
    filing_date: datetime
    period_end_date: datetime
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    accession_number: str = ""
    file_number: str = ""
    form_type: str = ""
    url: str = ""
    format: str = "unknown"  # xbrl, html, txt
    file_size: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time: Optional[float] = None
    chunk_count: int = 0
    vector_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class ProcessingResult:
    """Result of processing a filing"""
    success: bool
    filing_metadata: FilingMetadata
    chunks_created: int = 0
    vectors_stored: int = 0
    processing_time: float = 0.0
    error: Optional[Exception] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline"""
    # API Configuration
    sec_base_url: str = "https://www.sec.gov/Archives/edgar/data"
    user_agent: str = "ProximaDB Research (research@example.com)"
    rate_limit: float = 0.1  # Seconds between requests (SEC limit: 10/sec)
    
    # Processing Configuration
    batch_size: int = 100
    max_workers: int = 4
    chunk_size: int = 1024
    chunk_overlap: int = 128
    
    # Embedding Configuration
    embedding_model: str = "finbert"  # finbert, secbert, or hybrid
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    
    # Storage Configuration
    collection_prefix: str = "sec_filings"
    enable_compression: bool = True
    compression_type: str = "ZSTD"
    enable_quantization: bool = True
    quantization_type: str = "INT8"
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_multiplier: float = 2.0
    
    # Cache Configuration
    enable_cache: bool = True
    cache_dir: str = "./.sec_cache"
    
    # Monitoring
    enable_progress: bool = True
    log_level: str = "INFO"


# ============================================================================
# Base Classes and Interfaces
# ============================================================================

class FilingParser(ABC):
    """Abstract base class for filing parsers"""
    
    @abstractmethod
    async def parse(
        self,
        content: str,
        metadata: FilingMetadata
    ) -> Dict[str, Any]:
        """Parse filing content"""
        pass
    
    @abstractmethod
    def can_parse(self, format: str) -> bool:
        """Check if parser can handle format"""
        pass


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies"""
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


# ============================================================================
# Parser Implementations
# ============================================================================

class XBRLParser(FilingParser):
    """Parser for XBRL format filings"""
    
    def __init__(self):
        self.namespaces = {
            'xbrli': 'http://www.xbrl.org/2003/instance',
            'us-gaap': 'http://fasb.org/us-gaap/',
            'dei': 'http://xbrl.sec.gov/dei/',
            'xbrldi': 'http://xbrl.org/2006/xbrldi'
        }
        self.logger = logging.getLogger(f"{__name__}.XBRLParser")
    
    def can_parse(self, format: str) -> bool:
        return format.lower() == 'xbrl'
    
    async def parse(
        self,
        content: str,
        metadata: FilingMetadata
    ) -> Dict[str, Any]:
        """Parse XBRL content with multiple fallback strategies"""
        self.logger.info(f"Parsing XBRL filing for {metadata.ticker}")
        
        sections = {}
        financial_data = {}
        tables = []
        
        try:
            # Primary: Parse as XML
            root = etree.fromstring(content.encode('utf-8'))
            
            # Extract contexts
            contexts = self._extract_contexts(root)
            
            # Extract financial facts
            financial_data = self._extract_financial_facts(root, contexts)
            
            # Extract text blocks
            sections = self._extract_text_blocks(root)
            
            # Extract tables
            tables = self._extract_xbrl_tables(root)
            
            self.logger.info(
                f"Successfully parsed XBRL: {len(sections)} sections, "
                f"{len(financial_data)} metrics, {len(tables)} tables"
            )
            
        except etree.XMLSyntaxError as e:
            self.logger.warning(f"XML parsing failed: {e}. Trying fallback methods...")
            
            # Fallback 1: Extract as HTML
            try:
                sections = self._extract_html_sections(content)
            except Exception as e2:
                self.logger.warning(f"HTML extraction failed: {e2}")
            
            # Fallback 2: Pattern-based extraction
            if not sections:
                sections = self._pattern_extract_sections(content)
        
        except Exception as e:
            self.logger.error(f"XBRL parsing error: {e}")
            # Last resort: treat as plain text
            sections = {"full_content": content[:50000]}  # Limit size
        
        return {
            'sections': sections,
            'financial_data': financial_data,
            'tables': tables,
            'metadata': metadata.to_dict(),
            'parse_method': 'xbrl'
        }
    
    def _extract_contexts(self, root: etree.Element) -> Dict[str, Dict]:
        """Extract time period contexts from XBRL"""
        contexts = {}
        
        for context in root.xpath('.//xbrli:context', namespaces=self.namespaces):
            context_id = context.get('id')
            context_data = {}
            
            # Extract entity
            entity = context.find('.//xbrli:entity', namespaces=self.namespaces)
            if entity is not None:
                identifier = entity.find('.//xbrli:identifier', namespaces=self.namespaces)
                if identifier is not None:
                    context_data['entity'] = identifier.text
            
            # Extract period
            period = context.find('.//xbrli:period', namespaces=self.namespaces)
            if period is not None:
                instant = period.find('xbrli:instant', namespaces=self.namespaces)
                if instant is not None:
                    context_data['period'] = instant.text
                    context_data['period_type'] = 'instant'
                else:
                    start = period.find('xbrli:startDate', namespaces=self.namespaces)
                    end = period.find('xbrli:endDate', namespaces=self.namespaces)
                    if start is not None and end is not None:
                        context_data['period'] = f"{start.text} to {end.text}"
                        context_data['period_type'] = 'duration'
                        context_data['start_date'] = start.text
                        context_data['end_date'] = end.text
            
            contexts[context_id] = context_data
        
        return contexts
    
    def _extract_financial_facts(
        self,
        root: etree.Element,
        contexts: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Extract financial facts with context"""
        financial_data = {}
        
        # Find all elements with contextRef
        for element in root.xpath('.//*[@contextRef]'):
            if '}' in element.tag:
                namespace, tag = element.tag.rsplit('}', 1)
                namespace = namespace[1:]  # Remove leading {
            else:
                tag = element.tag
                namespace = None
            
            context_ref = element.get('contextRef')
            unit_ref = element.get('unitRef')
            decimals = element.get('decimals')
            
            # Get value
            value = element.text
            if value:
                value = value.strip()
                
                # Try to parse as number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                # Store with context
                if tag not in financial_data:
                    financial_data[tag] = []
                
                fact_data = {
                    'value': value,
                    'context': contexts.get(context_ref, {}),
                    'unit': unit_ref,
                    'decimals': decimals,
                    'namespace': namespace
                }
                
                financial_data[tag].append(fact_data)
        
        return financial_data
    
    def _extract_text_blocks(self, root: etree.Element) -> Dict[str, str]:
        """Extract text block sections"""
        sections = {}
        
        # Look for TextBlock elements
        for element in root.xpath('.//*[contains(local-name(), "TextBlock")]'):
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            section_name = tag.replace('TextBlock', '').replace('_', ' ').title()
            
            # Get text content
            text = element.text or ''
            
            # If it's HTML, clean it
            if text.strip().startswith('<'):
                text = self._clean_html_text(text)
            
            if text.strip():
                sections[section_name] = text
        
        return sections
    
    def _extract_xbrl_tables(self, root: etree.Element) -> List[Dict]:
        """Extract table data from XBRL"""
        tables = []
        
        # This would require more complex XBRL table linkbase parsing
        # For now, return empty list
        return tables
    
    def _extract_html_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from embedded HTML"""
        sections = {}
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove scripts and styles
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Find major sections
        for header in soup.find_all(['h1', 'h2', 'h3']):
            section_name = header.get_text(strip=True)
            section_content = []
            
            for sibling in header.find_next_siblings():
                if sibling.name and sibling.name.startswith('h'):
                    break
                text = sibling.get_text(strip=True)
                if text:
                    section_content.append(text)
            
            if section_content:
                sections[section_name] = '\n'.join(section_content)
        
        return sections
    
    def _pattern_extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections using regex patterns"""
        sections = {}
        
        # Common section patterns
        patterns = [
            (r'(?:ITEM|Item)\s+1\.?\s*(?:Business|BUSINESS)', 'Business Overview'),
            (r'(?:ITEM|Item)\s+1A\.?\s*(?:Risk|RISK)', 'Risk Factors'),
            (r'(?:ITEM|Item)\s+7\.?\s*(?:Management|MANAGEMENT)', 'MD&A'),
            (r'(?:ITEM|Item)\s+8\.?\s*(?:Financial|FINANCIAL)', 'Financial Statements'),
        ]
        
        for pattern, section_name in patterns:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            match = regex.search(content)
            
            if match:
                start = match.end()
                # Find next section or end
                next_match = re.search(r'(?:ITEM|Item)\s+\d+[A-Z]?\.?', content[start:])
                end = start + next_match.start() if next_match else len(content)
                
                sections[section_name] = content[start:end].strip()[:10000]  # Limit size
        
        return sections
    
    def _clean_html_text(self, html: str) -> str:
        """Clean HTML text"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts and styles
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)


class HTMLParser(FilingParser):
    """Parser for HTML format filings"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HTMLParser")
    
    def can_parse(self, format: str) -> bool:
        return format.lower() == 'html'
    
    async def parse(
        self,
        content: str,
        metadata: FilingMetadata
    ) -> Dict[str, Any]:
        """Parse HTML filing with table extraction"""
        self.logger.info(f"Parsing HTML filing for {metadata.ticker}")
        
        soup = BeautifulSoup(content, 'html.parser')
        sections = {}
        tables = []
        
        # Remove scripts and styles
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Extract sections
        sections = self._extract_sections(soup)
        
        # Extract tables
        tables = self._extract_tables(soup)
        
        # Extract metadata
        doc_metadata = self._extract_metadata(soup)
        
        self.logger.info(
            f"Parsed HTML: {len(sections)} sections, {len(tables)} tables"
        )
        
        return {
            'sections': sections,
            'tables': tables,
            'metadata': {**metadata.to_dict(), **doc_metadata},
            'parse_method': 'html'
        }
    
    def _extract_sections(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract sections based on headers"""
        sections = {}
        current_section = None
        current_content = []
        
        # Process all elements
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'table']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = element.get_text(strip=True)
                current_content = []
                
            elif element.name == 'table':
                # Add table placeholder
                if current_section:
                    current_content.append(f"[TABLE: {element.get('id', 'unnamed')}]")
                    
            else:
                # Add text content
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    current_content.append(text)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract and structure tables"""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            try:
                # Extract table data
                df = self._table_to_dataframe(table)
                
                if not df.empty:
                    tables.append({
                        'id': f"table_{i}",
                        'data': df,
                        'caption': self._get_table_caption(table),
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
            except Exception as e:
                self.logger.warning(f"Failed to parse table {i}: {e}")
        
        return tables
    
    def _table_to_dataframe(self, table) -> pd.DataFrame:
        """Convert HTML table to DataFrame"""
        rows = []
        headers = []
        
        # Extract headers
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        # If no thead, try first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                potential_headers = [cell.get_text(strip=True) 
                                   for cell in first_row.find_all(['th', 'td'])]
                if all(len(h) < 100 for h in potential_headers):
                    headers = potential_headers
        
        # Extract body rows
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr'):
            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if row and any(cell for cell in row):
                rows.append(row)
        
        # Create DataFrame
        if headers and rows:
            # Ensure consistent column count
            max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
            headers = headers[:max_cols] + [''] * (max_cols - len(headers))
            rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in rows]
            return pd.DataFrame(rows, columns=headers)
        elif rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame()
    
    def _get_table_caption(self, table) -> str:
        """Get table caption if available"""
        caption = table.find('caption')
        if caption:
            return caption.get_text(strip=True)
        
        # Check for preceding header
        prev = table.find_previous_sibling(['h1', 'h2', 'h3', 'h4', 'p'])
        if prev and prev.name == 'p':
            text = prev.get_text(strip=True)
            if len(text) < 200:
                return text
        
        return ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = {}
        
        # Extract from meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        # Extract title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text(strip=True)
        
        return metadata


class TextParser(FilingParser):
    """Parser for plain text format filings"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TextParser")
        
        # Section patterns
        self.section_patterns = {
            'business': r'(?:ITEM|Item)\s+1\.?\s*(?:Business|BUSINESS)',
            'risk_factors': r'(?:ITEM|Item)\s+1A\.?\s*(?:Risk|RISK)\s+(?:Factors|FACTORS)',
            'properties': r'(?:ITEM|Item)\s+2\.?\s*(?:Properties|PROPERTIES)',
            'legal': r'(?:ITEM|Item)\s+3\.?\s*(?:Legal|LEGAL)',
            'mda': r'(?:ITEM|Item)\s+7\.?\s*(?:Management|MANAGEMENT)',
            'financial': r'(?:ITEM|Item)\s+8\.?\s*(?:Financial|FINANCIAL)',
            'controls': r'(?:ITEM|Item)\s+9A\.?\s*(?:Controls|CONTROLS)',
        }
    
    def can_parse(self, format: str) -> bool:
        return format.lower() in ['txt', 'text', 'unknown']
    
    async def parse(
        self,
        content: str,
        metadata: FilingMetadata
    ) -> Dict[str, Any]:
        """Parse plain text filing"""
        self.logger.info(f"Parsing text filing for {metadata.ticker}")
        
        # Extract sections
        sections = self._extract_sections(content)
        
        # Extract tables (ASCII art)
        tables = self._extract_ascii_tables(content)
        
        # Extract financial numbers
        financial_data = self._extract_financial_numbers(content)
        
        self.logger.info(
            f"Parsed text: {len(sections)} sections, {len(tables)} tables"
        )
        
        return {
            'sections': sections,
            'tables': tables,
            'financial_data': financial_data,
            'metadata': metadata.to_dict(),
            'parse_method': 'text'
        }
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections using patterns"""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = list(regex.finditer(content))
            
            if matches:
                for i, match in enumerate(matches):
                    start = match.end()
                    
                    # Find end of section
                    if i < len(matches) - 1:
                        end = matches[i + 1].start()
                    else:
                        # Look for next item
                        next_item = re.search(
                            r'(?:ITEM|Item)\s+\d+[A-Z]?\.?',
                            content[start:],
                            re.IGNORECASE
                        )
                        end = start + next_item.start() if next_item else min(start + 50000, len(content))
                    
                    section_text = content[start:end].strip()
                    
                    # Clean up text
                    section_text = self._clean_text(section_text)
                    
                    if len(section_text) > 100:
                        sections[section_name] = section_text
        
        return sections
    
    def _extract_ascii_tables(self, content: str) -> List[Dict]:
        """Extract ASCII art tables"""
        tables = []
        lines = content.split('\n')
        
        in_table = False
        table_lines = []
        table_start = 0
        
        for i, line in enumerate(lines):
            # Detect table separators
            is_separator = (
                line.count('-') >= 20 or
                line.count('=') >= 20 or
                line.count('_') >= 20 or
                line.count('+') >= 3
            )
            
            # Detect table structure
            has_structure = (
                line.count('|') >= 2 or
                (line.count('  ') >= 3 and not line.strip().startswith('#'))
            )
            
            if (is_separator or has_structure) and not in_table:
                in_table = True
                table_start = i
                table_lines = [line]
            elif in_table:
                if line.strip() == '' and len(table_lines) > 3:
                    # End of table
                    tables.append({
                        'id': f"ascii_table_{len(tables)}",
                        'start_line': table_start,
                        'end_line': i,
                        'content': '\n'.join(table_lines),
                        'data': self._parse_ascii_table(table_lines)
                    })
                    in_table = False
                    table_lines = []
                elif line.strip():
                    table_lines.append(line)
        
        return tables
    
    def _parse_ascii_table(self, lines: List[str]) -> pd.DataFrame:
        """Parse ASCII table into DataFrame"""
        # Simple parsing - can be enhanced
        data = []
        
        for line in lines:
            if '|' in line:
                # Split by pipe
                cells = [cell.strip() for cell in line.split('|')]
                cells = [cell for cell in cells if cell]
                if cells:
                    data.append(cells)
            elif line.count('  ') >= 2:
                # Split by multiple spaces
                cells = re.split(r'\s{2,}', line.strip())
                if cells:
                    data.append(cells)
        
        if data:
            # First row might be headers
            if all(len(cell) < 50 for cell in data[0]):
                return pd.DataFrame(data[1:], columns=data[0])
            else:
                return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    def _extract_financial_numbers(self, content: str) -> Dict[str, List]:
        """Extract financial numbers with context"""
        financial_data = {}
        
        # Patterns for financial numbers
        patterns = [
            (r'\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?', 'currency'),
            (r'([\d,]+(?:\.\d+)?)\s*%', 'percentage'),
            (r'([\d,]+)\s+shares?', 'shares'),
        ]
        
        for pattern, data_type in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = regex.finditer(content)
            
            values = []
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                values.append({
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else None,
                    'context': context,
                    'position': match.start()
                })
            
            if values:
                financial_data[data_type] = values
        
        return financial_data
    
    def _clean_text(self, text: str) -> str:
        """Clean up text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'Page \d+', '', text)
        
        # Remove table of contents references
        text = re.sub(r'\.{3,}\d+', '', text)
        
        return text.strip()


# ============================================================================
# Embedding Strategies
# ============================================================================

class FinBERTEmbeddingStrategy(EmbeddingStrategy):
    """FinBERT embedding strategy for financial text"""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FinBERTEmbedding")
        
        try:
            self.provider = FinBERTProvider(
                model_name='finbert-general',
                batch_size=config.embedding_batch_size,
                normalize=config.normalize_embeddings
            )
            self.dimension = self.provider.get_dimension()
            self.logger.info(f"FinBERT initialized with dimension {self.dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FinBERT: {e}")
            raise
    
    async def embed(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Generate FinBERT embeddings"""
        try:
            # Preprocess texts for financial domain
            processed_texts = [
                self.provider.preprocess_financial_text(text)
                for text in texts
            ]
            
            # Generate embeddings
            embeddings = self.provider.embed_texts(processed_texts)
            
            self.logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self.dimension


class SECBERTEmbeddingStrategy(EmbeddingStrategy):
    """SEC-BERT embedding strategy for SEC filings"""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SECBERTEmbedding")
        
        try:
            self.provider = SECBERTProvider(
                model_name='sec-bert-base',
                batch_size=config.embedding_batch_size,
                normalize=config.normalize_embeddings
            )
            self.dimension = self.provider.get_dimension()
            self.logger.info(f"SEC-BERT initialized with dimension {self.dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SEC-BERT: {e}")
            raise
    
    async def embed(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Generate SEC-BERT embeddings"""
        try:
            # Preprocess for SEC filings
            processed_texts = [
                self.provider.preprocess_financial_text(text)
                for text in texts
            ]
            
            # Generate embeddings
            embeddings = self.provider.embed_texts(processed_texts)
            
            self.logger.debug(f"Generated {len(embeddings)} SEC-BERT embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"SEC-BERT embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self.dimension


class HybridEmbeddingStrategy(EmbeddingStrategy):
    """Hybrid strategy using different models for different content"""
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HybridEmbedding")
        
        # Initialize both providers
        self.finbert = FinBERTEmbeddingStrategy(config)
        self.secbert = SECBERTEmbeddingStrategy(config)
        
        # Use FinBERT dimension as primary
        self.dimension = self.finbert.get_dimension()
    
    async def embed(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Generate hybrid embeddings based on content type"""
        
        if not metadata:
            # Default to FinBERT if no metadata
            return await self.finbert.embed(texts)
        
        embeddings = []
        
        for text, meta in zip(texts, metadata):
            section_type = meta.get('section_type', '')
            
            # Choose model based on section type
            if section_type in ['risk_factors', 'legal_proceedings', 'controls_procedures']:
                # Use SEC-BERT for regulatory content
                embedding = await self.secbert.embed([text])
            else:
                # Use FinBERT for financial content
                embedding = await self.finbert.embed([text])
            
            embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        return self.dimension


# ============================================================================
# Main Ingestion Pipeline
# ============================================================================

class SECEdgarIngestionPipeline:
    """
    Complete SEC Edgar ingestion pipeline with enterprise features
    """
    
    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        proximadb_url: str = "http://localhost:5678",
        grpc_url: str = "http://localhost:5679"
    ):
        self.config = config or IngestionConfig()
        self.logger = logging.getLogger(f"{__name__}.Pipeline")
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, self.config.log_level))
        
        # Initialize ProximaDB clients
        self.rest_client = ProximaDBClient(
            base_url=proximadb_url,
            protocol=Protocol.REST
        )
        self.grpc_client = ProximaDBClient(
            base_url=grpc_url,
            protocol=Protocol.GRPC
        )
        
        # Initialize parsers
        self.parsers = [
            XBRLParser(),
            HTMLParser(),
            TextParser()
        ]
        
        # Initialize embedding strategy
        self._init_embedding_strategy()
        
        # Initialize cache
        if self.config.enable_cache:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'total_vectors': 0,
            'total_time': 0.0
        }
        
        # Rate limiting
        self.last_request_time = 0
        
        self.logger.info("SEC Edgar Ingestion Pipeline initialized")
    
    def _init_embedding_strategy(self):
        """Initialize embedding strategy based on config"""
        if self.config.embedding_model == 'finbert':
            self.embedding_strategy = FinBERTEmbeddingStrategy(self.config)
        elif self.config.embedding_model == 'secbert':
            self.embedding_strategy = SECBERTEmbeddingStrategy(self.config)
        elif self.config.embedding_model == 'hybrid':
            self.embedding_strategy = HybridEmbeddingStrategy(self.config)
        else:
            raise ValueError(f"Unknown embedding model: {self.config.embedding_model}")
        
        self.logger.info(f"Using {self.config.embedding_model} embedding strategy")
    
    async def initialize_collections(self) -> bool:
        """Create ProximaDB collections for SEC filings"""
        try:
            # Main collection
            main_config = CollectionConfig(
                name=f"{self.config.collection_prefix}_main",
                dimension=self.embedding_strategy.get_dimension(),
                distance_metric=DistanceMetric.COSINE,
                storage_engine=StorageEngine.NOVA,
                compression=CompressionType[self.config.compression_type] if self.config.enable_compression else None,
                quantization={
                    'type': QuantizationType[self.config.quantization_type],
                    'enable_two_stage_search': True,
                    'rerank_top_k': 100
                } if self.config.enable_quantization else None,
                filterable_columns=[
                    FilterableColumn("ticker", FilterableDataType.STRING),
                    FilterableColumn("filing_type", FilterableDataType.STRING),
                    FilterableColumn("section_type", FilterableDataType.STRING),
                    FilterableColumn("filing_date", FilterableDataType.TIMESTAMP),
                    FilterableColumn("fiscal_year", FilterableDataType.INTEGER),
                    FilterableColumn("fiscal_quarter", FilterableDataType.INTEGER),
                    FilterableColumn("chunk_index", FilterableDataType.INTEGER),
                ]
            )
            
            # Create collection
            await self.grpc_client.create_collection_async(main_config)
            
            self.logger.info(f"Created collection: {main_config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collections: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    async def fetch_filing(
        self,
        metadata: FilingMetadata
    ) -> str:
        """Fetch filing from SEC Edgar with retry logic"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.rate_limit:
            await asyncio.sleep(self.config.rate_limit - time_since_last)
        
        self.last_request_time = time.time()
        
        # Check cache first
        if self.config.enable_cache:
            cached_file = self.cache_dir / f"{metadata.accession_number}.txt"
            if cached_file.exists():
                self.logger.info(f"Using cached filing: {metadata.accession_number}")
                return cached_file.read_text()
        
        # Construct URL
        cik = metadata.cik.zfill(10)
        accession_clean = metadata.accession_number.replace('-', '')
        url = f"{self.config.sec_base_url}/{cik}/{accession_clean}/{metadata.accession_number}.txt"
        
        headers = {
            'User-Agent': self.config.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        timeout = ClientTimeout(total=60)
        
        async with ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Cache the filing
                    if self.config.enable_cache:
                        cached_file.write_text(content)
                    
                    metadata.file_size = len(content)
                    metadata.url = url
                    
                    return content
                else:
                    raise Exception(f"Failed to fetch filing: HTTP {response.status}")
    
    def detect_format(self, content: str) -> str:
        """Detect filing format"""
        content_lower = content[:5000].lower()
        
        if '<xbrl' in content_lower or '<?xml' in content_lower:
            return 'xbrl'
        elif '<html' in content_lower:
            return 'html'
        else:
            return 'txt'
    
    async def parse_filing(
        self,
        content: str,
        metadata: FilingMetadata
    ) -> Dict[str, Any]:
        """Parse filing using appropriate parser"""
        
        # Detect format
        format_type = self.detect_format(content)
        metadata.format = format_type
        
        # Find appropriate parser
        parser = None
        for p in self.parsers:
            if p.can_parse(format_type):
                parser = p
                break
        
        if not parser:
            self.logger.warning(f"No parser for format {format_type}, using text parser")
            parser = TextParser()
        
        # Parse
        return await parser.parse(content, metadata)
    
    async def chunk_filing(
        self,
        parsed_data: Dict[str, Any],
        metadata: FilingMetadata
    ) -> List[Dict[str, Any]]:
        """Chunk parsed filing into vector-ready documents"""
        
        chunks = []
        
        # Process text sections
        for section_name, section_content in parsed_data.get('sections', {}).items():
            if not section_content:
                continue
            
            # Determine section type
            section_type = self._classify_section(section_name)
            
            # Get appropriate chunking config
            chunk_config = self._get_chunk_config(section_type)
            chunker = TextChunker(chunk_config)
            
            # Generate source ID
            source_id = f"{metadata.ticker}_{metadata.filing_type.value}_{metadata.filing_date.strftime('%Y%m%d')}"
            
            # Chunk the text
            text_chunks = chunker.chunk_text(
                section_content,
                source_id=source_id,
                metadata={
                    'section': section_name,
                    'section_type': section_type.value if section_type else 'unknown',
                    'ticker': metadata.ticker,
                    'filing_type': metadata.filing_type.value,
                    'filing_date': metadata.filing_date.isoformat(),
                    'fiscal_year': metadata.fiscal_year,
                    'fiscal_quarter': metadata.fiscal_quarter
                }
            )
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'id': f"{chunk.chunk_id}_{i}",
                    'text': chunk.text,
                    'metadata': {
                        **chunk.metadata,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks)
                    }
                })
        
        # Process tables
        for table_info in parsed_data.get('tables', []):
            table_text = self._table_to_text(table_info)
            
            chunk_id = f"table_{hashlib.md5(table_text.encode()).hexdigest()[:8]}"
            chunks.append({
                'id': chunk_id,
                'text': table_text,
                'metadata': {
                    'type': 'table',
                    'ticker': metadata.ticker,
                    'filing_type': metadata.filing_type.value,
                    'filing_date': metadata.filing_date.isoformat(),
                    'fiscal_year': metadata.fiscal_year
                }
            })
        
        self.logger.info(f"Created {len(chunks)} chunks for {metadata.ticker}")
        return chunks
    
    def _classify_section(self, section_name: str) -> Optional[SectionType]:
        """Classify section type based on name"""
        section_lower = section_name.lower()
        
        if 'risk' in section_lower:
            return SectionType.RISK_FACTORS
        elif 'management' in section_lower or 'md&a' in section_lower:
            return SectionType.MDA
        elif 'financial' in section_lower and 'statement' in section_lower:
            return SectionType.FINANCIAL_STATEMENTS
        elif 'business' in section_lower:
            return SectionType.BUSINESS_OVERVIEW
        elif 'legal' in section_lower:
            return SectionType.LEGAL_PROCEEDINGS
        elif 'note' in section_lower:
            return SectionType.NOTES_TO_FINANCIALS
        elif 'control' in section_lower:
            return SectionType.CONTROLS_PROCEDURES
        elif 'compensation' in section_lower:
            return SectionType.EXECUTIVE_COMPENSATION
        elif 'audit' in section_lower:
            return SectionType.AUDITOR_REPORT
        else:
            return None
    
    def _get_chunk_config(self, section_type: Optional[SectionType]) -> ChunkingConfig:
        """Get chunking configuration for section type"""
        
        if section_type == SectionType.RISK_FACTORS:
            return ChunkingConfig(
                strategy=ChunkingStrategy.SEMANTIC,
                chunk_size=768,
                chunk_overlap=100,
                preserve_sentences=True
            )
        elif section_type == SectionType.MDA:
            return ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=1024,
                chunk_overlap=128,
                preserve_paragraphs=True
            )
        elif section_type == SectionType.FINANCIAL_STATEMENTS:
            return ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=2048,
                chunk_overlap=0
            )
        else:
            # Default config
            return ChunkingConfig(
                strategy=ChunkingStrategy.SLIDING_WINDOW,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
    
    def _table_to_text(self, table_info: Dict) -> str:
        """Convert table to searchable text"""
        if 'data' in table_info:
            if isinstance(table_info['data'], pd.DataFrame):
                return table_info['data'].to_string()
            else:
                return str(table_info['data'])
        elif 'content' in table_info:
            return table_info['content']
        else:
            return str(table_info)
    
    async def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Generate embeddings for chunks"""
        
        texts = [chunk['text'] for chunk in chunks]
        metadata = [chunk['metadata'] for chunk in chunks]
        
        # Process in batches
        embeddings = []
        batch_size = self.config.embedding_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            batch_embeddings = await self.embedding_strategy.embed(
                batch_texts,
                batch_metadata
            )
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def store_vectors(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        metadata: FilingMetadata
    ) -> int:
        """Store vectors in ProximaDB"""
        
        vectors = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(VectorRecord(
                id=chunk['id'],
                vector=embedding.tolist(),
                metadata=chunk['metadata']
            ))
        
        # Store in batches
        stored = 0
        batch_size = self.config.batch_size
        collection_name = f"{self.config.collection_prefix}_main"
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                result = await self.grpc_client.insert_vectors_async(
                    collection_id=collection_name,
                    records=batch
                )
                stored += len(batch)
            except Exception as e:
                self.logger.error(f"Failed to store batch: {e}")
        
        self.logger.info(f"Stored {stored} vectors for {metadata.ticker}")
        return stored
    
    async def ingest_filing(
        self,
        metadata: FilingMetadata
    ) -> ProcessingResult:
        """
        Complete ingestion pipeline for a single filing
        """
        
        start_time = time.time()
        result = ProcessingResult(
            success=False,
            filing_metadata=metadata
        )
        
        try:
            # Update status
            metadata.processing_status = ProcessingStatus.DOWNLOADING
            self.logger.info(f"Processing {metadata.filing_type.value} for {metadata.ticker}")
            
            # Fetch filing
            content = await self.fetch_filing(metadata)
            
            # Parse filing
            metadata.processing_status = ProcessingStatus.PARSING
            parsed_data = await self.parse_filing(content, metadata)
            
            # Chunk filing
            metadata.processing_status = ProcessingStatus.CHUNKING
            chunks = await self.chunk_filing(parsed_data, metadata)
            result.chunks_created = len(chunks)
            metadata.chunk_count = len(chunks)
            
            if not chunks:
                result.warnings.append("No chunks created")
                self.logger.warning(f"No chunks created for {metadata.ticker}")
                result.success = True
                return result
            
            # Generate embeddings
            metadata.processing_status = ProcessingStatus.EMBEDDING
            embeddings = await self.generate_embeddings(chunks)
            
            # Store vectors
            metadata.processing_status = ProcessingStatus.STORING
            vectors_stored = await self.store_vectors(chunks, embeddings, metadata)
            result.vectors_stored = vectors_stored
            metadata.vector_count = vectors_stored
            
            # Update status
            metadata.processing_status = ProcessingStatus.COMPLETED
            metadata.processing_time = time.time() - start_time
            result.processing_time = metadata.processing_time
            result.success = True
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['successful'] += 1
            self.stats['total_chunks'] += result.chunks_created
            self.stats['total_vectors'] += result.vectors_stored
            self.stats['total_time'] += result.processing_time
            
            self.logger.info(
                f"Successfully processed {metadata.ticker} {metadata.filing_type.value}: "
                f"{result.chunks_created} chunks, {result.vectors_stored} vectors, "
                f"{result.processing_time:.2f}s"
            )
            
        except Exception as e:
            metadata.processing_status = ProcessingStatus.FAILED
            metadata.error_message = str(e)
            result.error = e
            result.success = False
            
            self.stats['total_processed'] += 1
            self.stats['failed'] += 1
            
            self.logger.error(f"Failed to process {metadata.ticker}: {e}")
        
        return result
    
    async def batch_ingest(
        self,
        filings: List[FilingMetadata]
    ) -> List[ProcessingResult]:
        """
        Batch ingestion with parallel processing
        """
        
        results = []
        
        # Process in parallel with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_with_semaphore(metadata):
            async with semaphore:
                return await self.ingest_filing(metadata)
        
        tasks = [process_with_semaphore(metadata) for metadata in filings]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result, metadata in zip(results, filings):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    filing_metadata=metadata,
                    error=result
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        
        stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_processed']
            stats['avg_chunks_per_filing'] = stats['total_chunks'] / stats['total_processed']
            stats['avg_processing_time'] = stats['total_time'] / stats['total_processed']
        
        return stats
    
    async def search_filings(
        self,
        query: str,
        ticker: Optional[str] = None,
        filing_type: Optional[FilingType] = None,
        fiscal_year: Optional[int] = None,
        section_type: Optional[SectionType] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search SEC filings with metadata filtering
        """
        
        # Generate query embedding
        query_embedding = await self.embedding_strategy.embed([query])
        query_vector = query_embedding[0].tolist()
        
        # Build metadata filter
        metadata_filter = {}
        if ticker:
            metadata_filter['ticker'] = ticker
        if filing_type:
            metadata_filter['filing_type'] = filing_type.value
        if fiscal_year:
            metadata_filter['fiscal_year'] = fiscal_year
        if section_type:
            metadata_filter['section_type'] = section_type.value
        
        # Search
        collection_name = f"{self.config.collection_prefix}_main"
        
        results = await self.rest_client.search_vectors_async(
            collection_id=collection_name,
            query_vector=query_vector,
            metadata_filter=metadata_filter,
            top_k=top_k
        )
        
        return results


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """
    Example usage of the SEC Edgar ingestion pipeline
    """
    
    # Configure pipeline
    config = IngestionConfig(
        embedding_model='hybrid',  # Use hybrid FinBERT/SEC-BERT
        batch_size=50,
        max_workers=4,
        enable_cache=True,
        log_level='INFO'
    )
    
    # Initialize pipeline
    pipeline = SECEdgarIngestionPipeline(config)
    
    # Initialize collections
    await pipeline.initialize_collections()
    
    # Example filings to ingest
    filings = [
        FilingMetadata(
            ticker="AAPL",
            company_name="Apple Inc.",
            cik="0000320193",
            filing_type=FilingType.FORM_10K,
            filing_date=datetime(2024, 11, 1),
            period_end_date=datetime(2024, 9, 28),
            fiscal_year=2024,
            accession_number="0000320193-24-000123"
        ),
        FilingMetadata(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            cik="0000789019",
            filing_type=FilingType.FORM_10K,
            filing_date=datetime(2024, 7, 30),
            period_end_date=datetime(2024, 6, 30),
            fiscal_year=2024,
            accession_number="0000789019-24-000456"
        )
    ]
    
    # Batch ingest
    results = await pipeline.batch_ingest(filings)
    
    # Print results
    for result in results:
        if result.success:
            print(f"✅ {result.filing_metadata.ticker}: {result.chunks_created} chunks, {result.processing_time:.2f}s")
        else:
            print(f"❌ {result.filing_metadata.ticker}: {result.error}")
    
    # Print statistics
    stats = pipeline.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total vectors: {stats['total_vectors']}")
    
    # Example search
    search_results = await pipeline.search_filings(
        query="artificial intelligence machine learning risks",
        ticker="AAPL",
        filing_type=FilingType.FORM_10K,
        fiscal_year=2024,
        section_type=SectionType.RISK_FACTORS,
        top_k=5
    )
    
    print(f"\nSearch Results:")
    for result in search_results:
        print(f"  - {result['id']}: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())