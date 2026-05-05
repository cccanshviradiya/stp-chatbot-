import requests
import re

def test_mcp():
    url = "http://localhost:8000/sse"
    print(f"Connecting to {url}...")
    with requests.get(url, stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"Received: {decoded_line}")
                if decoded_line.startswith("event: endpoint"):
                    continue
                if decoded_line.startswith("data: "):
                    endpoint = decoded_line[6:]
                    print(f"Found endpoint: {endpoint}")
                    full_url = f"http://localhost:8000{endpoint}" if endpoint.startswith('/') else endpoint
                    print(f"Posting to {full_url}...")
                    resp = requests.post(full_url, json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "execute_sql",
                            "arguments": {"query": 'SELECT 1'}
                        }
                    })
                    print(f"Status: {resp.status_code}")
                    print(f"Response: {resp.text}")
                    # Keep listening for the result
                elif decoded_line.startswith("data: ") and session_url is not None:
                    print(f"Received Message: {decoded_line}")

if __name__ == "__main__":
    test_mcp()
