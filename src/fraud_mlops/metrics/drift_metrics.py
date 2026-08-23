from prometheus_client import Gauge


# 0 = no drift
# 1 = drift detected
data_drift_detected = Gauge(
    "data_drift_detected",
    "Whether data drift has been detected"
)


def update_drift_metric(drift_detected):
    """Update the Prometheus drift metric."""

    if drift_detected:
        data_drift_detected.set(1)
    else:
        data_drift_detected.set(0)