import pandas as pd


def detect_drift(reference_data, current_data, threshold=0.1):
    """
    Compare reference data with current data.

    Returns:
        drift_detected: True if significant drift is found
        drift_results: details for each feature
    """

    drift_results = {}

    common_columns = reference_data.columns.intersection(current_data.columns)

    for column in common_columns:
        reference_mean = reference_data[column].mean()
        current_mean = current_data[column].mean()

        if reference_mean == 0:
            drift_score = 0
        else:
            drift_score = abs(current_mean - reference_mean) / abs(reference_mean)

        drift_results[column] = drift_score

    drift_detected = any(
        score > threshold for score in drift_results.values()
    )

    return drift_detected, drift_results