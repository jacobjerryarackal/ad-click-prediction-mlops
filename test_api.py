import requests

url = "http://127.0.0.1:8000/predict"

# Sample raw user behavior payload (simulating a single row from an ad exchange, without 'click' or 'id')
sample_data = {
    "C1": 1005,
    "banner_pos": 0,
    "site_id": "1fbe01fe",
    "site_domain": "f3845767",
    "site_category": "28905ebd",
    "app_id": "ecad2386",
    "app_domain": "7801e8d9",
    "app_category": "07d7df22",
    "device_ip": "eb34399b",
    "device_model": "8a4875bd",
    "device_type": 1,
    "device_conn_type": 0,
    "C14": 15701,
    "C15": 320,
    "C16": 50,
    "C17": 2333,
    "C18": 157,
    "C19": 35,
    "C20": 100075,
    "C21": 23
}

print("Sending real-time request to Ad Click Prediction API...")
response = requests.post(url, json=sample_data)

if response.status_code == 200:
    print(f"Success! Response: {response.json()}")
else:
    print(f"Failed with status code {response.status_code}: {response.text}")