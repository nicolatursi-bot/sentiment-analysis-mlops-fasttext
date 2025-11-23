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
    
    def test_accuracy_calculation(self):
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        accuracy = PerformanceMetrics.calculate_accuracy(predictions, labels)
        assert accuracy == 1.0


class TestDriftDetector:
    
    def test_drift_detector_initialization(self):
        reference_labels = np.array([0, 1, 2, 0, 1, 2])
        detector = DriftDetector(reference_labels, threshold=0.05)
        assert detector.threshold == 0.05
        assert len(detector.reference_labels) == 6


class TestModelMonitor:
    
    @pytest.fixture
    def monitor(self):
        reference_predictions = np.random.choice([0, 1, 2], size=100)
        reference_labels = np.random.choice([0, 1, 2], size=100)
        return ModelMonitor("test-model", reference_predictions, reference_labels)
    
    def test_monitor_initialization(self, monitor):
        assert monitor.model_name == "test-model"
        assert len(monitor.reference_predictions) == 100
        assert monitor.drift_detector is not None
    
    def test_record_inference(self, monitor):
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        latencies = [2.5, 2.3, 2.4, 2.6, 2.5]
        monitor.record_inference(predictions, labels, latencies)
        assert len(monitor.inference_records) == 1