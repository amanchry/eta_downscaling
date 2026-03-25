import ee
import json
from pathlib import Path


_SERVICE_ACCOUNT_JSON = Path(__file__).parent / "waterinfor-service-account.json"


def ee_authenticate():
    """
    Authenticate and initialise the GEE Python API.

    Priority:
    1. Service-account JSON at helpers/waterinfor-service-account.json
       (non-interactive, suitable for automated / server runs).
    2. Interactive browser-based OAuth flow via ee.Authenticate()
       (used when the JSON key is absent — typical for local dev).
    """
    if _SERVICE_ACCOUNT_JSON.exists():
        with open(_SERVICE_ACCOUNT_JSON) as f:
            key = json.load(f)
        credentials = ee.ServiceAccountCredentials(
            key["client_email"],
            key_data=json.dumps(key),
        )
        ee.Initialize(credentials)
        print("GEE authenticated via service account.")
    else:
        print(
            f"Service account key not found at:\n  {_SERVICE_ACCOUNT_JSON}\n"
            "Falling back to interactive authentication (browser will open) ..."
        )
        ee.Authenticate()
        ee.Initialize()
        print("GEE authenticated interactively.")
