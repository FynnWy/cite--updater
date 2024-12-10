import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Iterator
import os
import logging
import pickle
from difflib import SequenceMatcher
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import fuzz, process
from retriv import SparseRetriever
from retriv.paths import sr_state_path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DblpParser:
    """Parser for DBLP XML database with in-memory indexing for fast searches"""
    
    def __init__(self, xml_path: str, cache_dir: str = "dblp_cache", index_name: str = "dblp_index"):
        """Initialize the parser with path to DBLP XML file and optional cache directory"""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"DBLP XML file not found at: {xml_path}")
        self.xml_path = xml_path
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".retriv", "collections", cache_dir)
        self.publications_by_title = {}  # In-memory index
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Try to load from cache first
        self.search_engine = None
        self.index_path = os.path.join(self.cache_dir, index_name)
        self._load_or_build_index()
    

    
    def _load_or_build_index(self):
        """Load index from cache if it exists and is newer than XML file, otherwise build it"""
        xml_mtime = os.path.getmtime(self.xml_path)
        
        if os.path.exists(self.index_path):
            index_mtime = os.path.getmtime(self.index_path)
            if index_mtime > xml_mtime:  # Only load if index is newer than XML
                try:
                    logger.info("Loading Retriv index from cache...")
                    self.search_engine = SparseRetriever.load(self.index_path)
                    logger.info("Successfully loaded index from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load index: {e}")
            else:
                logger.info("XML file is newer than index, rebuilding...")
        else:
            logger.info("No existing index found, building new one...")
        
        self._build_index()
    
    def _create_xml_parser(self) -> ET.XMLParser:
        """Create XML parser with all necessary entity definitions"""
        parser = ET.XMLParser()
        
        # Basic entities from DTD
        parser.entity['reg'] = '®'
        parser.entity['micro'] = 'µ'
        parser.entity['times'] = '×'
        
        # Latin-1 entities from DTD
        latin1_entities = {
            # Uppercase letters
            'Agrave': 'À', 'Aacute': 'Á', 'Acirc': 'Â', 'Atilde': 'Ã', 'Auml': 'Ä', 'Aring': 'Å',
            'AElig': 'Æ', 'Ccedil': 'Ç', 'Egrave': 'È', 'Eacute': 'É', 'Ecirc': 'Ê', 'Euml': 'Ë',
            'Igrave': 'Ì', 'Iacute': 'Í', 'Icirc': 'Î', 'Iuml': 'Ï', 'ETH': 'Ð', 'Ntilde': 'Ñ',
            'Ograve': 'Ò', 'Oacute': 'Ó', 'Ocirc': 'Ô', 'Otilde': 'Õ', 'Ouml': 'Ö', 'Oslash': 'Ø',
            'Ugrave': 'Ù', 'Uacute': 'Ú', 'Ucirc': 'Û', 'Uuml': 'Ü', 'Yacute': 'Ý', 'THORN': 'Þ',
            
            # Lowercase letters
            'szlig': 'ß', 'agrave': 'à', 'aacute': 'á', 'acirc': 'â', 'atilde': 'ã', 'auml': 'ä',
            'aring': 'å', 'aelig': 'æ', 'ccedil': 'ç', 'egrave': 'è', 'eacute': 'é', 'ecirc': 'ê',
            'euml': 'ë', 'igrave': 'ì', 'iacute': 'í', 'icirc': 'î', 'iuml': 'ï', 'eth': 'ð',
            'ntilde': 'ñ', 'ograve': 'ò', 'oacute': 'ó', 'ocirc': 'ô', 'otilde': 'õ', 'ouml': 'ö',
            'oslash': 'ø', 'ugrave': 'ù', 'uacute': 'ú', 'ucirc': 'û', 'uuml': 'ü', 'yacute': 'ý',
            'thorn': 'þ', 'yuml': 'ÿ'
        }
        
        # Add all entities to parser
        for entity, char in latin1_entities.items():
            parser.entity[entity] = char
            
        return parser
    
    def _extract_text(self, element: ET.Element) -> str:
        """Extract text from an element, handling special characters"""
        return ''.join(element.itertext()).strip()
    
    def _parse_publication(self, elem: ET.Element) -> Dict:
        """Parse a publication element into a dictionary"""
        publication = {
            'type': elem.tag,
            'key': elem.get('key', ''),
            'authors': [],
            'title': '',
            'year': '',
            'venue': '',
            'url': '',
            'doi': ''
        }
        
        for child in elem:
            if child.tag == 'author':
                publication['authors'].append(self._extract_text(child))
            elif child.tag == 'title':
                publication['title'] = self._extract_text(child)
            elif child.tag == 'year':
                publication['year'] = self._extract_text(child)
            elif child.tag in ['booktitle', 'journal']:
                publication['venue'] = self._extract_text(child)
            elif child.tag == 'ee' and child.text and 'doi.org' in child.text:
                publication['doi'] = self._extract_text(child)
            elif child.tag == 'url':
                publication['url'] = self._extract_text(child)
        
        return publication
    
    def _build_index(self):
        """Build search index using Retriv"""
        # Initialize the sparse retriever with custom settings
        self.search_engine = SparseRetriever(
            index_name=self.index_path,
            model="bm25",
            min_df=1,
            tokenizer="whitespace",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )

        publication_types = {'article', 'inproceedings', 'proceedings', 'book', 
                           'incollection', 'phdthesis', 'mastersthesis'}
        
        parser = self._create_xml_parser()
        context = ET.iterparse(self.xml_path, events=('end',), parser=parser)
        
        documents = []
        total_publications = 0
        
        for event, elem in context:
            if elem.tag in publication_types:
                pub = self._parse_publication(elem)
                if pub['title']:
                    documents.append({
                        "id": str(total_publications),
                        "text": pub['title'],
                        "metadata": pub
                    })
                    total_publications += 1
                    
                    if total_publications % 100000 == 0:
                        logger.info(f"Processed {total_publications} publications...")
                
                elem.clear()

        # Index the documents
        logger.info("Indexing documents...")
        self.search_engine.index(documents)
        
        # Get and print the actual save path
        index_path = sr_state_path(self.index_path)
        logger.info(f"Index will be saved to: {index_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save the state explicitly
        state = {
            "init_args": self.search_engine.init_args,
            "id_mapping": self.search_engine.id_mapping,
            "doc_count": self.search_engine.doc_count,
            "inverted_index": self.search_engine.inverted_index,
            "vocabulary": self.search_engine.vocabulary,
            "doc_lens": self.search_engine.doc_lens,
            "relative_doc_lens": self.search_engine.relative_doc_lens,
            "hyperparams": self.search_engine.hyperparams,
        }
        
        # Save using numpy's savez_compressed
        logger.info("Saving index...")
        np.savez_compressed(index_path, state=state)
        logger.info(f"Index saved to {index_path}")
    
    def search_by_title(self, search_title: str, threshold: float = 5.0, top_k: int = 1) -> Optional[Dict]:
        """
        Search for publications by title using BM25 ranking
        
        Args:
            search_title (str): The title to search for
            threshold (float): Minimum BM25 score to consider a match
            top_k (int): Number of results to return
        
        Returns:
            Optional[Dict]: Best matching publication or None if no matches above threshold
        """
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
            
        results = self.search_engine.search(
            query=search_title,
            return_docs=True,
            cutoff=top_k
        )
        
        # Return best match above threshold
        if results and results[0]["score"] >= threshold:
            logger.info(f"Found match with BM25 score: {results[0]['score']}")
            return results[0]["metadata"]
            
        return None
