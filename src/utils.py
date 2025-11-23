"""
UTILS MODULE - Funzioni di Utilità
====================================
Contiene funzioni helper per il progetto:
- Gestione configurazione
- Setup logging
- Gestione percorsi directory
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import os


class ConfigManager:
    """Gestisce la configurazione del progetto."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Inizializza il config manager.
        
        Args:
            config_path: Percorso del file di configurazione
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Carica la configurazione dal file JSON."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Errore nel caricamento config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Restituisce la configurazione di default."""
        return {
            "project_name": "Sentiment Analysis MLOps",
            "version": "1.0.0",
            "fasttext": {
                "lr": 0.5,
                "epoch": 25,
                "wordNgrams": 2,
                "dim": 100,
                "loss": "softmax"
            },
            "monitoring": {
                "window_size": 100,
                "drift_threshold": 0.05
            },
            "data_paths": {
                "raw": "data/raw",
                "processed": "data/processed",
                "models": "models",
                "outputs": "outputs"
            }
        }
    
    def get(self, key: str, default: Optional[any] = None) -> any:
        """
        Ottiene un valore dalla configurazione.
        
        Args:
            key: Chiave della configurazione (supporta notazione punto: "fasttext.lr")
            default: Valore di default se non trovato
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self, filepath: str = None) -> None:
        """Salva la configurazione su file."""
        filepath = filepath or self.config_path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def update(self, key: str, value: any) -> None:
        """
        Aggiorna un valore nella configurazione.
        
        Args:
            key: Chiave della configurazione (supporta notazione punto)
            value: Nuovo valore
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


def setup_logging(log_level: int = logging.INFO, 
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura il logging per l'applicazione.
    
    Args:
        log_level: Livello di logging (default: INFO)
        log_file: Percorso del file di log (opzionale)
        
    Returns:
        Logger configurato
    """
    logger = logging.getLogger(__name__)
    
    # Crea formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (opzionale)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(log_level)
    
    return logger


def get_project_dirs() -> Dict[str, Path]:
    """
    Restituisce i percorsi delle directory principali del progetto.
    
    Returns:
        Dict con percorsi: root, src, tests, data, models, outputs
    """
    root = Path(__file__).parent.parent  # Cartella principale del progetto
    
    return {
        'root': root,
        'src': root / 'src',
        'tests': root / 'tests',
        'data': root / 'data',
        'data_raw': root / 'data' / 'raw',
        'data_processed': root / 'data' / 'processed',
        'models': root / 'models',
        'outputs': root / 'outputs',
        'logs': root / 'logs'
    }


def ensure_directories() -> None:
    """Assicura che tutte le directory principali esistano."""
    dirs = get_project_dirs()
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Restituisce un timestamp formattato per i file."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_filename(model_name: str = "sentiment_model") -> str:
    """Restituisce un nome file per il modello con timestamp."""
    timestamp = get_timestamp()
    return f"{model_name}_{timestamp}"


class PathHelper:
    """Helper per gestire percorsi del progetto."""
    
    @staticmethod
    def get_data_processed_dir() -> Path:
        """Restituisce la directory dei dati processati."""
        dirs = get_project_dirs()
        return dirs['data_processed']
    
    @staticmethod
    def get_models_dir() -> Path:
        """Restituisce la directory dei modelli."""
        dirs = get_project_dirs()
        return dirs['models']
    
    @staticmethod
    def get_outputs_dir() -> Path:
        """Restituisce la directory degli output."""
        dirs = get_project_dirs()
        return dirs['outputs']
    
    @staticmethod
    def get_logs_dir() -> Path:
        """Restituisce la directory dei log."""
        dirs = get_project_dirs()
        return dirs['logs']


# Inizializza directory al caricamento del modulo
ensure_directories()