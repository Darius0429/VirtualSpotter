import sounddevice as sd

def list_audio_devices():
    """List all available audio devices"""
    print(sd.query_devices())


list_audio_devices()