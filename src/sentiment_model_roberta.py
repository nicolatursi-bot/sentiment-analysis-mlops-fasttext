"""
SENTIMENT MODEL - RoBERTa Preaddestrato
========================================
Implementazione del modello RoBERTa per sentiment analysis da HuggingFace.
Modello: twitter-roberta-base-sentiment-latest (cardiffnlp)

Classificazione: positivo, neutro, negativo
Fonte dati: Tweet Eval dataset (pubblico)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from typing import List, Dict, Tuple
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoBERTaSentimentModel:
    """
    Wrapper per il modello RoBERTa preaddestrato per sentiment analysis.
    
    Modello: cardiffnlp/twitter-roberta-base-sentiment-latest
    Fonte: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    """
    
    def __init__(self):
        """Inizializza il modello RoBERTa preaddestrato da HuggingFace."""
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.reverse_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        logger.info(f"Caricamento modello RoBERTa: {self.model_name}")
        
        try:
            # Carica tokenizer e modello da HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Crea pipeline per sentiment analysis
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # -1 = CPU, 0+ = GPU
            )
            
            logger.info("✅ Modello RoBERTa caricato con successo")
            logger.info(f"Classi supportate: {list(self.label_mapping.values())}")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            raise
    
    def inference(self, texts: List[str]) -> List[Dict]:
        """
        Esegue inference su una lista di testi.
        
        Args:
            texts: Lista di testi da classificare
            
        Returns:
            Lista di dict con predictions
        """
        if not texts:
            logger.warning("Lista di testi vuota")
            return []
        
        predictions = []
        start_time = time.time()
        
        logger.info(f"Inference su {len(texts)} testi...")
        
        try:
            # Esegui predictions con pipeline
            results = self.pipeline(texts)
            
            for text, result in zip(texts, results):
                # Estrai label e score
                label_text = result['label']  # Es: "POSITIVE", "NEGATIVE", "NEUTRAL"
                score = float(result['score'])
                
                # Normalizza label a minuscolo
                label_normalized = label_text.lower()
                
                predictions.append({
                    'text': text,
                    'label': label_normalized,
                    'confidence': score,
                    'model': 'twitter-roberta-base-sentiment-latest'
                })
            
            inference_time = time.time() - start_time
            logger.info(f"✅ Inference completata in {inference_time:.3f}s ({len(texts)} testi)")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Errore durante l'inference: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Restituisce informazioni sul modello."""
        return {
            'model_name': self.model_name,
            'model_type': 'RoBERTa (preaddestrato)',
            'labels': list(self.label_mapping.values()),
            'num_labels': 3,
            'source': 'https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest',
            'use_case': 'Twitter/Social Media Sentiment Analysis'
        }
    
    def batch_inference(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Esegue inference batch per testi numerosi.
        
        Args:
            texts: Lista di testi
            batch_size: Dimensione del batch
            
        Returns:
            Lista di predictions
        """
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_predictions = self.inference(batch)
            all_predictions.extend(batch_predictions)
        
        return all_predictions


class SentimentAnalyzer:
    """Classe per analisi di sentiment end-to-end."""
    
    def __init__(self):
        """Inizializza l'analyzer con il modello RoBERTa."""
        self.model = RoBERTaSentimentModel()
        logger.info("SentimentAnalyzer inizializzato")
    
    def analyze(self, texts: List[str]) -> Dict:
        """
        Analizza sentiment di una lista di testi.
        
        Args:
            texts: Lista di testi da analizzare
            
        Returns:
            Dict con risultati aggregati
        """
        # Esegui inference
        predictions = self.model.inference(texts)
        
        # Aggregazioni statistiche
        labels = [p['label'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        sentiment_counts = {
            'positive': labels.count('positive'),
            'neutral': labels.count('neutral'),
            'negative': labels.count('negative')
        }
        
        sentiment_percentages = {
            label: (count / len(labels) * 100) if labels else 0
            for label, count in sentiment_counts.items()
        }
        
        result = {
            'total_texts': len(texts),
            'predictions': predictions,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'model_info': self.model.get_model_info()
        }
        
        logger.info(f"Analisi completata: Positive={sentiment_counts['positive']}, "
                   f"Neutral={sentiment_counts['neutral']}, Negative={sentiment_counts['negative']}")
        
        return result