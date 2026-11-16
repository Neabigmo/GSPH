import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DIR = os.path.join(ROOT, "configs")
DATASETS_DIR = os.path.join(ROOT, "datasets")
RESULTS_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(ROOT, "logs")
REPORTS_DIR = os.path.join(ROOT, "reports")
STATUS_DIR = os.path.join(ROOT, "status")


def ensure_layout():
    for path in [
        DATASETS_DIR,
        os.path.join(ROOT, "third_party"),
        os.path.join(ROOT, "environments"),
        os.path.join(ROOT, "runners"),
        RESULTS_DIR,
        LOG_DIR,
        REPORTS_DIR,
        STATUS_DIR,
        os.path.join(ROOT, "research", "gsph"),
    ]:
        os.makedirs(path, exist_ok=True)
