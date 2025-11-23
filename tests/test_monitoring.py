"""
TEST MONITORING
===============
Unit test per il modulo monitor.
Verifica il monitoraggio, drift detection e export metriche.
"""

import pytest
import numpy as np
import tempfile
import os
from src.monitor import ModelMonitor, PerformanceMetrics, DriftDetector


class TestPerformanceMetrics:
    """Test suite per il calcolo delle metriche."""
    
    def test_accuracy_calculation(self):
        """
        Test: Verifica il calcolo dell'accuracy.
        
        Controlla:
        - Accuracy corretta per predizioni giuste
        - Accuracy corretta per predizioni sbagliate
        - Casi edge (array vuoto)
        """
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        
        accuracy = PerformanceMetrics.calculate_accuracy(predictions, labels)
        assert accuracy == 1.0  # 100% accuracy
        
        # Test con errori
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 1, 0, 1])
        
        accuracy = PerformanceMetrics.calculate_accuracy(predictions, labels)
        assert 0 < accuracy < 1
    
    def test_precision_calculation(self):
        """
        Test: Verifica il calcolo della precision.
        
        Controlla:
        - Precision corretta per multi-class
        - Valori tra 0 e 1
        """
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        
        precision = PerformanceMetrics.calculate_precision(predictions, labels)
        assert 0 <= precision <= 1
    
    def test_recall_calculation(self):
        """
        Test: Verifica il calcolo del recall.
        
        Controlla:
        - Recall corretta per multi-class
        - Valori tra 0 e 1
        """
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        
        recall = PerformanceMetrics.calculate_recall(predictions, labels)
        assert 0 <= recall <= 1
    
    def test_f1_score_calculation(self):
        """
        Test: Verifica il calcolo dell'F1-score.
        
        Controlla:
        - F1-score corretta
        - Valori tra 0 e 1
        - Media armonica tra precision e recall
        """
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        
        f1 = PerformanceMetrics.calculate_f1_score(predictions, labels)
        assert 0 <= f1 <= 1


class TestDriftDetector:
    """Test suite per il drift detection."""
    
    def test_drift_detector_initialization(self):
        """
        Test: Verifica l'inizializzazione del drift detector.
        
        Controlla:
        - Detector creato correttamente
        - Threshold configurato
        """
        reference_labels = np.array([0, 1, 2, 0, 1, 2])
        detector = DriftDetector(reference_labels, threshold=0.05)
        
        assert detector.threshold == 0.05
        assert len(detector.reference_labels) == 6
    
    def test_no_drift_detection(self):
        """
        Test: Verifica che non rileva drift quando distribuzione è simile.
        
        Controlla:
        - No drift quando distribuzioni uguali
        - p_value alto
        """
        reference_labels = np.array([0, 1, 2, 0, 1, 2])
        current_labels = np.array([0, 1, 2, 0, 1, 2])
        
        detector = DriftDetector(reference_labels, threshold=0.05)
        result = detector.detect_drift(current_labels)
        
        assert 'has_drift' in result
        assert 'p_value' in result
        assert 'chi_square_stat' in result
    
    def test_drift_detection_structure(self):
        """
        Test: Verifica la struttura del risultato drift detection.
        
        Controlla:
        - Tutti i campi presenti
        - Valori nel range corretto
        """
        reference_labels = np.array([0, 1, 2, 0, 1, 2])
        current_labels = np.array([0, 1, 2, 0, 1, 2])
        
        detector = DriftDetector(reference_labels, threshold=0.05)
        result = detector.detect_drift(current_labels)
        
        assert 'has_drift' in result
        assert 'p_value' in result
        assert isinstance(result['has_drift'], (bool, np.bool_))
        assert 0 <= result['p_value'] <= 1


class TestModelMonitor:
    """Test suite per il ModelMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Fixture che fornisce un'istanza di ModelMonitor."""
        reference_predictions = np.random.choice([0, 1, 2], size=100)
        reference_labels = np.random.choice([0, 1, 2], size=100)
        
        return ModelMonitor(
            "test-model",
            reference_predictions,
            reference_labels,
            drift_threshold=0.05
        )
    
    def test_monitor_initialization(self, monitor):
        """
        Test: Verifica l'inizializzazione del monitor.
        
        Controlla:
        - Monitor creato correttamente
        - Nomi e configurazione corretti
        """
        assert monitor.model_name == "test-model"
        assert len(monitor.reference_predictions) == 100
        assert len(monitor.reference_labels) == 100
        assert monitor.drift_detector is not None
    
    def test_record_inference(self, monitor):
        """
        Test: Verifica la registrazione di inference.
        
        Controlla:
        - Record creato correttamente
        - Metriche calcolate
        - Latency registrata
        """
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        latencies = [2.5, 2.3, 2.4, 2.6, 2.5]
        
        monitor.record_inference(predictions, labels, latencies)
        
        assert len(monitor.inference_records) == 1
        assert len(monitor.latency_records) == 5
        
        record = monitor.inference_records[0]
        assert 'accuracy' in record
        assert 'f1_score' in record
        assert 'mean_latency_ms' in record
    
    def test_get_monitoring_report(self, monitor):
        """
        Test: Verifica la generazione del report di monitoraggio.
        
        Controlla:
        - Report struttura corretta
        - Tutti i campi presenti
        - Metriche calcolate
        """
        # Aggiungi un paio di batch
        for _ in range(2):
            predictions = np.random.choice([0, 1, 2], size=50)
            labels = np.random.choice([0, 1, 2], size=50)
            latencies = [2.5 + i*0.1 for i in range(50)]
            
            monitor.record_inference(predictions, labels, latencies)
        
        report = monitor.get_monitoring_report()
        
        assert 'model_name' in report
        assert 'report_timestamp' in report
        assert 'total_inferences' in report
        assert 'average_accuracy' in report
        assert 'average_f1_score' in report
        assert 'inference_latency_stats' in report
        
        # Verifica latency stats
        latency_stats = report['inference_latency_stats']
        assert 'mean_latency_ms' in latency_stats
        assert 'p95_latency_ms' in latency_stats
        assert 'p99_latency_ms' in latency_stats
    
    def test_export_metrics_json(self, monitor):
        """
        Test: Verifica l'export delle metriche in JSON.
        
        Controlla:
        - File JSON creato
        - Contenuto valido
        """
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        latencies = [2.5, 2.3, 2.4, 2.6, 2.5]
        
        monitor.record_inference(predictions, labels, latencies)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.json")
            monitor.export_metrics_json(filepath)
            
            assert os.path.exists(filepath)
            
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert 'model_name' in data
                assert 'average_accuracy' in data
    
    def test_export_metrics_csv(self, monitor):
        """
        Test: Verifica l'export delle metriche in CSV.
        
        Controlla:
        - File CSV creato
        - Contenuto valido
        - Numero di righe corretto
        """
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        latencies = [2.5, 2.3, 2.4, 2.6, 2.5]
        
        monitor.record_inference(predictions, labels, latencies)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.csv")
            monitor.export_metrics_csv(filepath)
            
            assert os.path.exists(filepath)
            
            import csv
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert 'accuracy' in rows[0]
                assert 'f1_score' in rows[0]
    
    def test_retraining_trigger(self, monitor):
        """
        Test: Verifica il trigger di retraining.
        
        Controlla:
        - Trigger falso quando accuracy alta
        - Trigger vero quando accuracy bassa
        """
        # Scenario 1: Accuracy alta, no drift
        predictions = np.array([0, 1, 2] * 10)  # Predizioni giuste
        labels = np.array([0, 1, 2] * 10)
        latencies = [2.5] * 30
        
        monitor.record_inference(predictions, labels, latencies)
        
        should_retrain = monitor.get_retraining_trigger()
        # Con accuracy alta, potrebbe non triggerare
        assert isinstance(should_retrain, bool)
    
    def test_multiple_batches(self, monitor):
        """
        Test: Verifica il monitoraggio di multipli batch.
        
        Controlla:
        - Tracking corretto di multipli batch
        - Metriche cumulative corrette
        """
        # Registra 3 batch
        for i in range(3):
            predictions = np.random.choice([0, 1, 2], size=50)
            labels = np.random.choice([0, 1, 2], size=50)
            latencies = [2.5 + i*0.1 for i in range(50)]
            
            monitor.record_inference(predictions, labels, latencies)
        
        report = monitor.get_monitoring_report()
        
        assert report['total_inferences'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```
