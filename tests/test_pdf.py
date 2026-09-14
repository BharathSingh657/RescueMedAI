"""
Unit test suite for Hospital Pre-Arrival PDF Report Generator.
"""
import pytest
from src.backend.state import IncidentState, PatientRecord
from src.backend.hospital.notification import HospitalNotificationGenerator
from src.backend.hospital.pdf_generator import HospitalPDFReportGenerator


def test_pdf_generation_basic():
    state = IncidentState()
    state.patient = PatientRecord(
        patient_id="PT-TEST-01",
        age=35,
        systolic_bp=90,
        diastolic_bp=60,
        spo2=91,
        heart_rate=115,
        respiratory_rate=26,
        consciousness="Voice",
        symptoms=["Massive External Bleeding", "Difficulty breathing"]
    )
    notif = HospitalNotificationGenerator.generate(state)
    pdf_bytes = HospitalPDFReportGenerator.generate(state, notif)

    assert pdf_bytes is not None
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")
