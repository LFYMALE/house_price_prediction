import requests

URL = "http://localhost:9696/predict"

sample_house = {
    "square_footage": 2500,
    "num_bedrooms": 4,
    "num_bathrooms": 3,
    "year_built": 2015,
    "lot_size": 2.5,
    "garage_size": 2,
    "neighborhood_quality": 8,
}

response = requests.post(URL, json=sample_house, timeout=10)
print(f"Status Code : {response.status_code}")
print(f"Response    : {response.json()}")