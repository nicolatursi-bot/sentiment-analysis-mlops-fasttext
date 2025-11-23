"""
MONITOR MODULE - Fase 3: Monitoraggio Continuo
================================================
Gestisce il monitoraggio delle performance del modello e rilevamento del drift.

Funzionalità:
- Tracciamento metriche di performance in real-time
- Drift detection (chi-square test)
- Latency monitoring
- Report generation
- Export metriche (JSON, CSV)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import csv
from pathlib import Path
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calcola metriche di performance del modello."""
    
    @staticmethod
    def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calcola l'accuracy."""
        if len(predictions) == 0:
            return 0.0
        return np.mean(predictions == labels)
    
    @staticmethod
    def calculate_precision(predictions: np.ndarray, labels: np.ndarray, class_idx: int = None) -> float:
        """Calcola la precision."""
        if class_idx is None:
            # Multi-class average
            classes = np.unique(labels)
            precisions = []
            for c in classes:
                pred_mask = predictions == c
                if pred_mask.sum() == 0:
                    precisions.append(0.0)
                else:
                    correct = ((predictions == c) & (labels == c)).sum()
                    precisions.append(correct / pred_mask.sum())
            return np.mean(precisions)
        else:
            # Single class
            pred_mask = predictions == class_idx
            if pred_mask.sum() == 0:
                return 0.0
            correct = ((predictions == class_idx) & (labels == class_idx)).sum()
            return correct / pred_mask.sum()
    
    @staticmethod
    def calculate_recall(predictions: np.ndarray, labels: np.ndarray, class_idx: int = None) -> float:
        """Calcola il recall."""
        if class_idx is None:
            # Multi-class average
            classes = np.unique(labels)
            recalls = []
            for c in classes:
                label_mask = labels == c
                if label_mask.sum() == 0:
                    recalls.append(0.0)
                else:
                    correct = ((predictions == c) & (labels == c)).sum()
                    recalls.append(correct / label_mask.sum())
            return np.mean(recalls)
        else:
            # Single class
            label_mask = labels == class_idx
            if label_mask.sum() == 0:
                return 0.0
            correct = ((predictions == class_idx) & (labels == class_idx)).sum()
            return correct / label_mask.sum()
    
    @staticmethod
    def calculate_f1_score(predictions: np.ndarray, labels: np.ndarray, class_idx: int = None) -> float:
        """Calcola l'F1-score."""
        precision = PerformanceMetrics.calculate_precision(predictions, labels, class_idx)
        recall = PerformanceMetrics.calculate_recall(predictions, labels, class_idx)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


class DriftDetector:
    """Rileva drift nei dati (change in data distribution)."""
    
    def __init__(self, reference_labels: np.ndarray, threshold: float = 0.05):
        """
        Inizializza il drift detector.
        
        Args:
            reference_labels: Etichette di riferimento per il confronto
            threshold: Soglia p-value per rilevare drift (default: 0.05)
        """
        self.reference_labels = reference_labels
        self.threshold = threshold
        logger.info(f"DriftDetector inizializzato con threshold: {threshold}")
    
    def detect_drift(self, current_labels: np.ndarray) -> Dict:
        """
        Rileva drift tra distribuzione di riferimento e corrente usando chi-square test.
        
        Args:
            current_labels: Etichette correnti da testare
            
        Returns:
            Dict con risultati del test di drift
        """
        logger.info(f"Testing drift: {len(current_labels)} samples")
        
        # Crea tabelle di contingenza
        ref_counts = np.bincount(self.reference_labels, minlength=3)
        curr_counts = np.bincount(current_labels, minlength=3)
        
        # Normalizza
        ref_dist = ref_counts / ref_counts.sum()
        curr_dist = curr_counts / curr_counts.sum()
        
        # Chi-square test
        try:
            chi2_stat = np.sum((curr_counts - ref_counts) ** 2 / (ref_counts + 1e-10))
            p_value = 1 - chi2_stat / (len(current_labels) + ref_counts.sum())
            
            has_drift = p_value < self.threshold
            
            result = {
                'has_drift': has_drift,
                'p_value': float(p_value),
                'chi_square_stat': float(chi2_stat),
                'threshold': self.threshold,
                'reference_dist': ref_dist.tolist(),
                'current_dist': curr_dist.tolist()
            }
            
            if has_drift:
                logger.warning(f"⚠️ DRIFT DETECTED! p_value: {p_value:.4f}")
            else:
                logger.info(f"✅ No drift detected. p_value: {p_value:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Errore nel drift detection: {e}")
            return {
                'has_drift': False,
                'p_value': 1.0,
                'chi_square_stat': 0.0,
                'error': str(e)
            }


class ModelMonitor:
    """Monitora le performance del modello e rileva drift."""
    
    def __init__(self, model_name: str, reference_predictions: np.ndarray, 
                 reference_labels: np.ndarray, drift_threshold: float = 0.05):
        """
        Inizializza il monitor.
        
        Args:
            model_name: Nome del modello
            reference_predictions: Predizioni di riferimento
            reference_labels: Etichette di riferimento
            drift_threshold: Soglia drift (0.05 = 5%)
        """
        self.model_name = model_name
        self.reference_predictions = reference_predictions
        self.reference_labels = reference_labels
        
        self.inference_records = []
        self.latency_records = []
        
        self.drift_detector = DriftDetector(reference_labels, threshold=drift_threshold)
        
        logger.info(f"ModelMonitor inizializzato per: {model_name}")
        logger.info(f"Reference accuracy: {PerformanceMetrics.calculate_accuracy(reference_predictions, reference_labels):.4f}")
    
    def record_inference(self, predictions: np.ndarray, labels: np.ndarray, 
                        latencies: List[float]) -> None:
        """
        Registra un batch di inferenze.
        
        Args:
            predictions: Array di predizioni
            labels: Array di etichette vere
            latencies: Lista di latenze (millisecondi)
        """
        timestamp = datetime.now().isoformat()
        
        accuracy = PerformanceMetrics.calculate_accuracy(predictions, labels)
        f1 = PerformanceMetrics.calculate_f1_score(predictions, labels)
        drift_result = self.drift_detector.detect_drift(labels)
        
        record = {
            'timestamp': timestamp,
            'num_samples': len(predictions),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'drift_detected': drift_result['has_drift'],
            'p_value': drift_result['p_value'],
            'mean_latency_ms': float(np.mean(latencies)) if latencies else 0.0,
            'p95_latency_ms': float(np.percentile(latencies, 95)) if latencies else 0.0,
            'p99_latency_ms': float(np.percentile(latencies, 99)) if latencies else 0.0
        }
        
        self.inference_records.append(record)
        self.latency_records.extend(latencies)
        
        logger.info(f"Recorded batch: accuracy={accuracy:.4f}, drift={drift_result['has_drift']}")
    
    def get_monitoring_report(self) -> Dict:
        """Genera il report di monitoraggio."""
        if not self.inference_records:
            return {'status': 'No data recorded yet'}
        
        accuracies = [r['accuracy'] for r in self.inference_records]
        f1_scores = [r['f1_score'] for r in self.inference_records]
        drifts = [r['drift_detected'] for r in self.inference_records]
        
        report = {
            'model_name': self.model_name,
            'report_timestamp': datetime.now().isoformat(),
            'total_inferences': len(self.inference_records),
            'average_accuracy': float(np.mean(accuracies)),
            'average_f1_score': float(np.mean(f1_scores)),
            'drift_detected_count': sum(drifts),
            'drift_percentage': float(sum(drifts) / len(drifts) * 100) if drifts else 0.0,
            'inference_latency_stats': {
                'mean_latency_ms': float(np.mean(self.latency_records)) if self.latency_records else 0.0,
                'median_latency_ms': float(np.median(self.latency_records)) if self.latency_records else 0.0,
                'p95_latency_ms': float(np.percentile(self.latency_records, 95)) if self.latency_records else 0.0,
                'p99_latency_ms': float(np.percentile(self.latency_records, 99)) if self.latency_records else 0.0,
                'min_latency_ms': float(np.min(self.latency_records)) if self.latency_records else 0.0,
                'max_latency_ms': float(np.max(self.latency_records)) if self.latency_records else 0.0
            },
            'recent_records': self.inference_records[-10:]  # Ultimi 10 record
        }
        
        return report
    
    def export_metrics_json(self, filepath: str) -> None:
        """Esporta metriche in formato JSON."""
        report = self.get_monitoring_report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"✅ Metriche esportate in JSON: {filepath}")
    
    def export_metrics_csv(self, filepath: str) -> None:
        """Esporta metriche in formato CSV."""
        if not self.inference_records:
            logger.warning("Nessun record da esportare")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.inference_records[0].keys())
            writer.writeheader()
            writer.writerows(self.inference_records)
        
        logger.info(f"✅ Metriche esportate in CSV: {filepath}")
    
    def get_retraining_trigger(self) -> bool:
        """
        Determina se il modello deve essere riaddestrato.
        
        Trigger:
        - Accuracy scesa sotto il 85%
        - Drift rilevato
        """
        if not self.inference_records:
            return False
        
        latest_accuracy = self.inference_records[-1]['accuracy']
        latest_drift = self.inference_records[-1]['drift_detected']
        
        should_retrain = latest_accuracy < 0.85 or latest_drift
        
        if should_retrain:
            logger.warning(f"⚠️ RETRAINING TRIGGER: accuracy={latest_accuracy:.4f}, drift={latest_drift}")
        
        return should_retrain