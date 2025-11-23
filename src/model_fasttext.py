"""
MODEL MODULE - Versione FastText
=================================
Gestisce il caricamento, training e inference del modello FastText.

FastText è un modello leggero e veloce sviluppato da Facebook Research.
Combina word embeddings con logistic regression per la classificazione.
"""

import fasttext
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastTextSentimentModel:
    """Wrapper per il modello FastText di sentiment analysis."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Inizializza il modello FastText."""
        self.model = None
        self.model_path = model_path
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.reverse_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("FastTextSentimentModel inizializzato (modello non ancora caricato)")
    
    def train(self, train_file: str, 
              lr: float = 0.5,
              epoch: int = 25,
              wordNgrams: int = 2,
              dim: int = 100,
              loss: str = 'softmax') -> Dict:
        """
        Addestra il modello FastText su un file di training.
        
        Parametri FastText:
        - lr: Learning rate (default: 0.5)
        - epoch: Numero di epoche (default: 25)
        - wordNgrams: N-gramma di parole (default: 2)
        - dim: Dimensione dei vettori (default: 100)
        - loss: Funzione di loss ('softmax' per multi-class)
        """
        logger.info(f"Inizio training FastText da file: {train_file}")
        logger.info(f"Parametri: lr={lr}, epoch={epoch}, dim={dim}, loss={loss}")
        
        start_time = time.time()
        
        try:
            self.model = fasttext.train_supervised(
                input=train_file,
                lr=lr,
                epoch=epoch,
                wordNgrams=wordNgrams,
                dim=dim,
                loss=loss,
                verbose=2
            )
            
            training_time = time.time() - start_time
            
            logger.info(f"✅ Training completato in {training_time:.2f} secondi")
            
            stats = {
                'training_time': training_time,
                'lr': lr,
                'epoch': epoch,
                'wordNgrams': wordNgrams,
                'dim': dim,
                'loss': loss,
                'model_type': 'FastText'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Errore durante il training: {e}")
            raise
    
    def inference(self, texts: List[str], k: int = 1) -> List[Dict]:
        """
        Esegue l'inference su una lista di testi.
        
        Processo:
        1. Per ogni testo, ottieni predizione dal modello
        2. Estrai label e confidence
        3. Converti in formato leggibile
        """
        if self.model is None:
            logger.error("Modello non è stato addestrato o caricato")
            raise ValueError("Model non inizializzato")
        
        predictions = []
        start_time = time.time()
        
        for text in texts:
            try:
                labels, scores = self.model.predict(text, k=k)
                
                top_label = labels[0].replace('__label__', '')
                top_score = float(scores[0])
                
                all_predictions = {}
                for label, score in zip(labels, scores):
                    label_name = label.replace('__label__', '')
                    all_predictions[label_name] = float(score)
                
                predictions.append({
                    'text': text,
                    'label': top_label,
                    'confidence': top_score,
                    'all_predictions': all_predictions
                })
                
            except Exception as e:
                logger.warning(f"Errore nell'inference di '{text[:50]}': {e}")
                predictions.append({
                    'text': text,
                    'label': 'neutral',
                    'confidence': 0.33,
                    'all_predictions': {'negative': 0.33, 'neutral': 0.33, 'positive': 0.33}
                })
        
        inference_time = time.time() - start_time
        logger.info(f"Inference su {len(texts)} testi completata in {inference_time:.3f}s")
        
        return predictions
    
    def evaluate(self, val_file: str) -> Dict:
        """Valuta il modello su un file di validazione."""
        if self.model is None:
            logger.error("Modello non è stato addestrato o caricato")
            raise ValueError("Model non inizializzato")
        
        logger.info(f"Valutazione del modello su: {val_file}")
        
        try:
            N, precision, recall = self.model.test(val_file)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            metrics = {
                'num_samples': N,
                'accuracy': precision,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"Valutazione completata:")
            logger.info(f"  - Accuracy: {precision:.4f}")
            logger.info(f"  - F1-Score: {f1:.4f}")
            logger.info(f"  - Samples: {N}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Errore nella valutazione: {e}")
            raise
    
    def save_model(self, save_path: str) -> None:
        """Salva il modello FastText su disco."""
        if self.model is None:
            logger.error("Nessun modello da salvare")
            raise ValueError("Model non inizializzato")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Salvataggio del modello in {save_path}")
        
        model_path_str = str(save_path)
        self.model.save_model(model_path_str)
        
        metadata = {
            'model_type': 'FastText',
            'label_mapping': self.label_mapping,
            'model_file': f"{model_path_str}.bin"
        }
        
        metadata_path = f"{model_path_str}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"✅ Modello salvato: {model_path_str}.bin")
        logger.info(f"✅ Metadati salvati: {metadata_path}")
    
    def load_model(self, model_path: str) -> None:
        """Carica un modello FastText da disco."""
        logger.info(f"Caricamento del modello da {model_path}")
        
        try:
            self.model = fasttext.load_model(model_path)
            self.model_path = model_path
            logger.info("✅ Modello caricato con successo")
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            raise
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Ottiene il vettore di embedding per una parola."""
        if self.model is None:
            raise ValueError("Model non inizializzato")
        
        return self.model.get_word_vector(word)
    
    def get_model_info(self) -> Dict:
        """Restituisce informazioni sul modello."""
        if self.model is None:
            return {'status': 'Model not initialized'}
        
        info = {
            'model_type': 'FastText',
            'labels': self.model.labels,
            'num_labels': len(self.model.labels),
            'dimension': self.model.dim,
            'label_mapping': self.label_mapping
        }
        
        return info


class ModelTrainer:
    """Classe per gestire il ciclo completo di training e validazione."""
    
    def __init__(self):
        """Inizializza il trainer."""
        self.best_model = None
        self.best_metrics = None
        self.training_history = []
        logger.info("ModelTrainer inizializzato")
    
    def train_and_evaluate(self, train_file: str, val_file: str,
                          hyperparameters: Optional[Dict] = None) -> Dict:
        """Esegue training e validazione del modello."""
        if hyperparameters is None:
            hyperparameters = {
                'lr': 0.5,
                'epoch': 25,
                'wordNgrams': 2,
                'dim': 100,
                'loss': 'softmax'
            }
        
        logger.info(f"Training con hyperparameters: {hyperparameters}")
        
        model = FastTextSentimentModel()
        train_stats = model.train(train_file, **hyperparameters)
        
        val_metrics = model.evaluate(val_file)
        
        results = {
            'train_stats': train_stats,
            'val_metrics': val_metrics,
            'hyperparameters': hyperparameters
        }
        
        self.best_model = model
        self.best_metrics = val_metrics
        self.training_history.append(results)
        
        return results
    
    def get_best_model(self) -> FastTextSentimentModel:
        """Restituisce il miglior modello addestrato."""
        return self.best_model
    
    def get_training_history(self) -> List[Dict]:
        """Restituisce lo storico di training."""
        return self.training_history
    
