import pygame
import threading
import speech_recognition as sr

class Speech2Text:
    def __init__(self) -> None:
        self.listening = False
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
    def transcribe_speech(self):
        """Listen to microphone input and transcribe speech"""
        try:
            print("Listening...")
            self.listening = True

            # Adjust for ambient noise with a reduced duration for quicker response
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
            self.listening = False
            print("Processing audio...")
            speech_text = self.recognizer.recognize_google(audio)
            print(f"You said: {speech_text}")
            return speech_text

        except sr.WaitTimeoutError:
            print("Listening timed out, no speech detected.")
            return None

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
            return None

        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None
