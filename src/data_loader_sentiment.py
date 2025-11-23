"""
DATA LOADER - Dataset Pubblici per Sentiment
==============================================
Caricamento e preprocessing di dataset pubblici per sentiment analysis.

Dataset supportati:
- Tweet Eval: Dataset di tweet classificati (positivo, neutro, negativo)
- Fallback: Dati di esempio per testing

Fonte: https://huggingface.co/datasets/tweet_eval
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentDataLoader:
    """
    Carica e preprocessa dataset pubblici per sentiment analysis.
    
    Classi supportate: positivo, neutro, negativo (3 classi)
    """
    
    def __init__(self):
        """Inizializza il data loader."""
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.reverse_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        logger.info("SentimentDataLoader inizializzato")
    
    def load_public_dataset(self) -> Dict:
        """
        Carica dataset pubblici di sentiment.
        
        Usa Tweet Eval dataset da HuggingFace.
        In caso di errore di connessione, usa dati di fallback.
        
        Returns:
            Dict con train e validation data
        """
        logger.info("Caricamento dataset pubblico di sentiment...")
        
        try:
            # Prova a caricare da HuggingFace
            from datasets import load_dataset
            
            logger.info("Scaricamento Tweet Eval dataset da HuggingFace...")
            dataset = load_dataset("tweet_eval", "sentiment")
            
            logger.info(f"✅ Dataset caricato: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")
            
            return {
                'train': dataset['train'],
                'validation': dataset['validation']
            }
            
        except Exception as e:
            logger.warning(f"Impossibile scaricare dataset da HuggingFace: {e}")
            logger.info("Usando dati di fallback per testing...")
            
            # Dati di fallback - dataset pubblico di esempio
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> Dict:
        """
        Dataset di fallback per testing quando la connessione fallisce.
        
        Dati di esempio basati su Tweet Eval structure.
        """
        train_data = {
            'text': [
                'Love this product! Best purchase ever',
                'This is amazing, highly recommend',
                'Excellent quality and fast shipping',
                'I hate this, waste of money',
                'Terrible experience, never again',
                'Very disappointed with quality',
                'It is okay, nothing special',
                'Average product, does the job',
                'Neither good nor bad, just okay',
                'Great service and product quality',
                'Horrible, complete disappointment',
                'Just what I needed, perfect',
                'Not bad, could be better',
                'Absolutely fantastic experience',
                'Really poor quality'
            ],
            'label': [2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 2, 1, 2, 0]
        }
        
        val_data = {
            'text': [
                'This is wonderful',
                'I am not satisfied',
                'It is fine',
                'Best ever',
                'Worst product'
            ],
            'label': [2, 0, 1, 2, 0]
        }
        
        # Crea struttura compatibile con HuggingFace
        class SimpleDataset:
            def __init__(self, data_dict):
                self.data = data_dict
                self.indices = list(range(len(data_dict['text'])))
            
            def __len__(self):
                return len(self.data['text'])
            
            def __getitem__(self, idx):
                return {
                    'text': self.data['text'][idx],
                    'label': self.data['label'][idx]
                }
            
            def __iter__(self):
                for idx in self.indices:
                    yield self[idx]
        
        logger.info(f"✅ Dataset di fallback caricato: Train={len(train_data['text'])}, Val={len(val_data['text'])}")
        
        return {
            'train': SimpleDataset(train_data),
            'validation': SimpleDataset(val_data)
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocessa testo per sentiment analysis.
        
        Operazioni:
        - Conversione a minuscolo
        - Rimozione URL
        - Rimozione menzioni e hashtag
        - Pulizia spazi multipli
        """
        import re
        
        # Minuscolo
        text = text.lower()
        
        # Rimuovi URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Rimuovi menzioni
        text = re.sub(r'@\w+', '', text)
        
        # Rimuovi hashtag (mantieni il testo)
        text = re.sub(r'#', '', text)
        
        # Pulizia spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text or text.isspace():
            text = "neutral text"
        
        return text
    
    def prepare_for_inference(self, texts: List[str]) -> List[str]:
        """
        Prepara testi per inference con il modello RoBERTa.
        
        Args:
            texts: Lista di testi raw
            
        Returns:
            Lista di testi preprocessati
        """
        preprocessed = [self.preprocess_text(text) for text in texts]
        return preprocessed
    
    def get_label_info(self) -> Dict:
        """Restituisce informazioni sulle etichette di sentiment."""
        return {
            'labels': self.label_mapping,
            'reverse_mapping': self.reverse_mapping,
            'num_classes': 3,
            'class_names': ['negative', 'neutral', 'positive']
        }


class DatasetAnalyzer:
    """Analizza caratteristiche del dataset di sentiment."""
    
    @staticmethod
    def analyze_dataset(dataset) -> Dict:
        """
        Analizza distribuzione e caratteristiche del dataset.
        
        Args:
            dataset: Dataset da analizzare
            
        Returns:
            Dict con statistiche
        """
        labels = []
        text_lengths = []
        
        for sample in dataset:
            labels.append(sample['label'])
            text_lengths.append(len(sample['text'].split()))
        
        label_counts = {
            0: labels.count(0),
            1: labels.count(1),
            2: labels.count(2)
        }
        
        return {
            'total_samples': len(labels),
            'label_distribution': label_counts,
            'label_percentages': {
                'negative': label_counts[0] / len(labels) * 100,
                'neutral': label_counts[1] / len(labels) * 100,
                'positive': label_counts[2] / len(labels) * 100
            },
            'avg_text_length': float(np.mean(text_lengths)),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths)
        }