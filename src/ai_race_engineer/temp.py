import google.generativeai as genai
import os

# Replace 'YOUR_API_KEY' with your actual API key
genai.configure(api_key=os.environ["API_KEY"])
# model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with your desired model
# prompt = "Write a story about a robot who dreams of becoming a chef."
# response = model.generate_content(prompt)
# print(response.text)
prompt = "I have a lot of sliding for the throttle input."
query = f"""
    You are a professional Race Engineer. Your task is to respond to the user's query 
    based on real-time data from rFactor 2's shared memory map (rF2 Memory Map).
    
    **Instructions:**
    1. Always categorize the user query into one of two categories:
       - **PITSTOP:** Queries related to pit strategy, refueling, tire changes, mechanical issues, etc.
       - **DRIVING:** Queries related to improving on-track performance, such as braking, corner exits, lap times, telemetry analysis, etc.
    
    2. Always base your responses on the provided rFactor 2 Memory Map data. The data contains telemetry, scoring, and force feedback information. Use this data to form accurate, data-driven responses.
    
    **Query from the user:**
    "{prompt}"

    **Categories:**
    - PITSTOP: Pit strategy, fuel level, tire wear, and mechanical adjustments.
    - DRIVING: Speed, cornering, braking, acceleration, and telemetry feedback.

    **Action:**
    - First, categorize the user query as either "PITSTOP" or "DRIVING".
    - Second, respond using relevant rF2 Memory Map data, including telemetry, vehicle scoring, or force feedback information.
    """
sample_file = genai.upload_file(path="suspension_data.log", display_name="Sample drawing",mime_type="text/plain")
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
response = model.generate_content(
    [query, sample_file]
)

print(response.text)