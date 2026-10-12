"""CardioAI Phase 2 — cohort builder and validation toolkit for Mayo Clinic Platform.

Builds the five agent-validation cohorts of study CA-AGENTS-VAL-PHASE2
(Echo, ECG, CAD, Risk Stratification, Longitudinal) from OMOP CDM data inside a
Mayo Clinic Platform Discover workspace (SparkSQL), and computes the
pre-specified validation endpoints.
"""
__version__ = "0.1.0"
COHORTS = ["echo", "ecg", "cad", "risk", "longitudinal"]
