# Modified main.py with better API initialization
import subprocess
import sys
import asyncio
import threading
from ai_race_engineer.module_agent import agent
import time
from tinypedal.tinypedal.api_control import api
# from tinypedal.tinypedal.regex_pattern import API_NAME_RF2, API_NAME_LMU

def initialize_api():
    # Explicitly initialize the API
    api.connect("Le Mans Ultimate")  # or "rFactor 2" depending on your simulation
    api.start()
    return api.state  # Return if API is connected successfully

def run_tinypedal():
    # This runs TinyPedal in a separate process
    subprocess.run([sys.executable, r"./TinyPedal/run.py"])

if __name__ == "__main__":
    # Start TinyPedal in a separate thread
    tinypedal_thread = threading.Thread(target=run_tinypedal)
    tinypedal_thread.daemon = True  # This makes the thread exit when the main program exits
    tinypedal_thread.start()
    
    print("Waiting for TinyPedal to initialize...")
    time.sleep(10)  # Wait for TinyPedal to load
    
    # Initialize the API specifically for your agent
    api_ready = initialize_api()
    print(f"API state: {'Connected' if api_ready else 'Not connected'}")
    
    # Wait until the API is ready before continuing
    attempts = 0
    max_attempts = 10
    while not api_ready and attempts < max_attempts:
        print(f"Waiting for API to connect... Attempt {attempts+1}/{max_attempts}")
        time.sleep(2)
        api_ready = api.state
        attempts += 1
    
    if not api_ready:
        print("WARNING: Failed to connect to API after multiple attempts")
    
    try:
        asyncio.run(agent.start_interface())
    except KeyboardInterrupt:
        api.stop()  # Properly close the API connection
        asyncio.run(agent.stop_interface())