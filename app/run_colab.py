import subprocess, time, os
from pyngrok import ngrok

ngrok.kill()
ngrok.set_auth_token("35dOIHJThOmsVpcJtJuMsODZAi4_6hkT1PByThnZ7B4KrgWjS")

PORT = 8501

process = subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.address", "0.0.0.0",
     "--server.port", str(PORT)],
)

time.sleep(8)
public_url = ngrok.connect(PORT)

print("Streamlit URL:", public_url)
