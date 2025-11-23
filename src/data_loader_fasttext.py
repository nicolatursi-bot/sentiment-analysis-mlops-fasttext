"""
DATA LOADER MODULE 
========================================
Gestisce il caricamento e il preprocessing dei dati per l'analisi del sentiment.

Flusso di elaborazione:
1. Caricamento dataset da HuggingFace Datasets
2. Preprocessing e pulizia testi
3. Preparazione dati in formato FastText (__label__ prefix)
4. Creazione di file train/test per FastText
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datasets import load_dataset, DatasetDict
import logging
import re
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentDataLoader:
    """
    Classe responsabile del caricamento e preprocessing dei dati di sentiment per FastText.
    
    FastText richiede un formato specifico:
    __label__positive Questo prodotto è fantastico!
    __label__negative Non sono soddisfatto della qualità
    __label__neutral Il prodotto è okay
    """
    
    def __init__(self):
        """Inizializza il DataLoader per FastText."""
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.reverse_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        logger.info("SentimentDataLoader inizializzato per FastText")
    
    def load_twitter_sentiment_dataset(self) -> DatasetDict:
        """
        Carica il dataset di sentiment dai tweet usando HuggingFace Datasets.
        Dataset: tweet_eval contenente ~60k tweet classificati per sentiment.
        """
        logger.info("Caricamento dataset Twitter Sentiment (tweet_eval)...")
        try:
            dataset = load_dataset("tweet_eval", "sentiment")
            logger.info(f"Dataset caricato. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
            return dataset
        except Exception as e:
            logger.error(f"Errore nel caricamento del dataset: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Esegue il preprocessing del testo per migliorare la qualità dell'input.
        
        Passaggi:
        1. Conversione a minuscolo
        2. Rimozione di URL
        3. Rimozione di menzioni
        4. Pulizia spazi bianchi duplicati
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text or text.isspace():
            text = "neutral"
        
        return text
    
    def convert_to_fasttext_format(self, text: str, label: int) -> str:
        """Converte un esempio nel formato richiesto da FastText."""
        label_name = self.label_mapping.get(label, 'neutral')
        fasttext_line = f"__label__{label_name} {text}"
        return fasttext_line
    
    def prepare_dataset(self, dataset: DatasetDict, output_dir: str = "./data/processed") -> Tuple[str, str]:
        """Prepara il dataset in formato FastText e salva in file di testo."""
        logger.info("Inizio del preprocessing del dataset...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        train_data = []
        val_data = []
        
        for example in dataset['train']:
            text = self.preprocess_text(example['text'])
            label = example['label']
            fasttext_line = self.convert_to_fasttext_format(text, label)
            train_data.append(fasttext_line)
        
        for example in dataset['validation']:
            text = self.preprocess_text(example['text'])
            label = example['label']
            fasttext_line = self.convert_to_fasttext_format(text, label)
            val_data.append(fasttext_line)
        
        train_path = f"{output_dir}/train.txt"
        val_path = f"{output_dir}/val.txt"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_data))
        
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_data))
        
        logger.info(f"Dataset preparato. Train: {len(train_data)}, Val: {len(val_data)}")
        
        return train_path, val_path
    
    def get_label_names(self) -> Dict[int, str]:
        """Restituisce il mapping tra indici numerici e nomi di sentiment."""
        return self.label_mapping
    
    def get_reverse_label_mapping(self) -> Dict[str, int]:
        """Restituisce il mapping inverso da nomi sentiment a indici."""
        return self.reverse_mapping


def load_and_prepare_data(output_dir: str = "./data/processed") -> Tuple[str, str]:
    """Funzione di convenienza per caricare e preparare il dataset in una sola chiamata."""
    loader = SentimentDataLoader()
    raw_dataset = loader.load_twitter_sentiment_dataset()
    train_path, val_path = loader.prepare_dataset(raw_dataset, output_dir)
    
    logger.info(f"✅ Dataset preparato:")
    logger.info(f"   Train file: {train_path}")
    logger.info(f"   Val file: {val_path}")
    
    return train_path, val_path