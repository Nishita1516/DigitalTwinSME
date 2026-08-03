import base64
import zlib
import urllib.request
import os

code = '''%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}, "flowchart": {"curve": "linear"}}}%%
flowchart TD
    A([ Start ]) --> B[ Load Dataset ]
    B --> C[ Preprocess Data ]
    C --> D[ Generate Sequences ]
    D --> E[ Train ML Model ]
    E --> F[ Predict RUL ]
    F --> G[" Process Logs (NLP) "]
    G --> H[ Display Results ]
    H --> I([ End ])'''

out_dir = r"c:\WORKFILES\dissertation\Digital Twin\ARCHITECTURE\diagrams_png"
os.makedirs(out_dir, exist_ok=True)

compressed = zlib.compress(code.encode('utf-8'), 9)
encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
url = f"https://kroki.io/mermaid/png/{encoded}"
target_path = os.path.join(out_dir, "simple_flowchart.png")

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Successfully generated simple_flowchart.png")
except Exception as e:
    print(f"Error generating simple_flowchart.png: {e}")
