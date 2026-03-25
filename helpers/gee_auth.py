import ee
import json
from pathlib import Path


_SERVICE_ACCOUNT_JSON = Path(__file__).parent / "waterinfor-service-account.json"


def ee_authenticate():
    with open(_SERVICE_ACCOUNT_JSON) as f:
        service_account_key = json.load(f)

    credentials = ee.ServiceAccountCredentials(
        service_account_key["client_email"],
        key_data=json.dumps(service_account_key),
    )
    ee.Initialize(credentials)
    print("GEE Initialized")


