import tkinter as tk
from tkinter import scrolledtext, messagebox
import random
import json
import pyttsx3
import threading
import os
import webbrowser
from datetime import datetime
from PIL import Image, ImageTk  # <-- Added for background image

# Machine Learning and AI Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import google.generativeai as genai

# Optional speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Speech recognition not available. Install with: pip install SpeechRecognition pyaudio")

class dreamzoneAssistant:
    def __init__(self):
        self.is_speaking = False
        self.voice_enabled = True
        self.genai_model = None

        self.setup_generative_model()
        self.load_intents()
        self.train_model()
        self.setup_voice_engine()
        self.setup_speech_recognition()
        
        # GUI must be set up last as it relies on other components
        self.setup_gui()

    def setup_generative_model(self):
        """Initializes the Google Generative AI model."""
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("API Key not found. Please set the 'GEMINI_API_KEY' environment variable.")
                messagebox.showwarning("API Key Missing", "Gemini API key not found. The assistant will rely on local responses only.")
                return
            
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel('gemini-2.0-flash') # <-- Updated model
            print("✓ Gemini model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            messagebox.showwarning("Gemini Error", f"Could not initialize Gemini: {e}\nFalling back to local responses only.")

    def load_intents(self):
        """Load intents from JSON file or create a default one."""
        try:
            if os.path.exists("intents.json"):
                with open("intents.json", "r", encoding='utf-8') as f:
                    self.intents = json.load(f)
                print("✓ Intents loaded from intents.json")
            else:
                print("intents.json not found. Creating a default file...")
                self.intents = self.create_default_intents()
                self.save_intents()
        except Exception as e:
            print(f"Error loading intents: {e}")
            messagebox.showerror("Error", f"Could not load intents: {e}")
            self.intents = self.create_default_intents()

    def create_default_intents(self):
        """Creates a default intents structure if the JSON file is missing."""
        return {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["hi", "hello", "hey", "how are you", "what's up"],
                    "responses": ["Hello!", "Hi there, how can I help?", "Hey!", "I'm doing great, thanks for asking!"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["bye", "goodbye", "see you later", "exit", "quit"],
                    "responses": ["Goodbye!", "See you later!", "Have a great day!"]
                },
                {
                    "tag": "thanks",
                    "patterns": ["thanks", "thank you", "that's helpful"],
                    "responses": ["You're welcome!", "Anytime!", "Glad I could help!"]
                },
                {
                    "tag": "about",
                    "patterns": ["who are you", "what are you", "tell me about yourself"],
                    "responses": ["I am dream zone, a voice assistant created to help you.", "I'm your friendly neighborhood assistant, dream zone!"]
                }
            ]
        }

    def save_intents(self):
        """Save intents to JSON file."""
        try:
            with open("intents.json", "w", encoding='utf-8') as f:
                json.dump(self.intents, f, indent=4, ensure_ascii=False)
            print("✓ Intents saved to intents.json")
        except Exception as e:
            print(f"Error saving intents: {e}")

    def train_model(self):
        """Train the local machine learning model for intent recognition."""
        try:
            patterns, tags = [], []
            for intent in self.intents["intents"]:
                for pattern in intent["patterns"]:
                    if pattern.strip():
                        patterns.append(pattern.lower())
                        tags.append(intent["tag"])
            
            if not patterns:
                raise ValueError("No training patterns found in intents!")
                
            self.vectorizer = CountVectorizer()
            X = self.vectorizer.fit_transform(patterns)
            self.model = MultinomialNB()
            self.model.fit(X, tags)
            print(f"✓ Local model trained with {len(patterns)} patterns")
        except Exception as e:
            print(f"Error training model: {e}")
            messagebox.showerror("Training Error", f"Could not train the model: {e}")
            if hasattr(self, 'root'): self.root.destroy()

    def _handle_hardcoded_commands(self, user_input):
        """Handles simple, hardcoded commands for speed and reliability."""
        t = user_input.lower().strip()
        
        if "what time" in t:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."
        if "what's the date" in t:
            return f"Today is {datetime.now().strftime('%B %d, %Y')}."
        if "open youtube" in t:
            webbrowser.open(r"https://www.youtube.com/@CDCKasaragod/videos")
            return "Opening YouTube."
        if "open google" in t or "open website" in t:
            webbrowser.open(r"https://www.dreamzone.co.in/")
            return "Opening website."
        if "open instagram" in t:
            webbrowser.open(r"https://www.instagram.com/dreamzone_kasaragod/?hl=en")
            return "Opening Instagram."
        if "open location" in t or "open map" in t or "where is dream zone" in t or "where is the located" in t or "where is the location" in t:
            webbrowser.open(r"https://www.google.com/maps/dir//3rd+Floor,+Square+Nine+mall,+New+Busstand+Junction,+NH+66,+Kasaragod,+Kerala+671121/@12.5069479,74.9131774,12z/data=!4m8!4m7!1m0!1m5!1m1!1s0x3ba4825cecc298f5:0x781fa60b9fc5edc0!2m2!1d74.9955793!2d12.5069604?entry=ttu&g_ep=EgoyMDI1MTAwMS4wIKXMDSoASAFQAw%3D%3D")
            return "Opening Maps."

        return None # No hardcoded command was matched

    def chatbot_response(self, user_input):
        """
        Generates a response by checking hardcoded commands, then the local model,
        and finally falling back to the Gemini AI model.
        """
        if not user_input.strip():
            return "Please say something!"

        # 1. Check for simple, hardcoded commands first
        hardcoded_response = self._handle_hardcoded_commands(user_input)
        if hardcoded_response:
            return hardcoded_response

        # 2. Try the local ML model for predefined intents
        try:
            X_test = self.vectorizer.transform([user_input.lower()])
            probabilities = self.model.predict_proba(X_test)[0]
            max_prob = max(probabilities)
            
            if max_prob > 0.65: # Confidence threshold
                tag = self.model.classes_[probabilities.argmax()]
                for intent in self.intents["intents"]:
                    if intent["tag"] == tag:
                        return random.choice(intent["responses"])
        except Exception as e:
            print(f"Local model prediction error: {e}")

        # 3. If no high-confidence local match, fall back to Gemini
        if self.genai_model:
            try:
                print("Querying Gemini for a response...")
                prompt = f"You are dream zone, a friendly and helpful voice assistant. Respond to the user's message concisely: '{user_input}'"
                response = self.genai_model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API error: {e}")
                return "I'm having trouble connecting to my advanced brain right now. Please try again."
        
        # 4. Final fallback if Gemini is not available
        return "I'm not sure how to respond to that. Can you try rephrasing?"
    y=10
    def send_message(self, event=None):
        """Handles sending the user message and getting a response in a thread."""
        user_input = self.entry.get().strip()
        if not user_input:
            return
        
        self.chat_window.config(state=tk.NORMAL)
        self.canvas.create_text(10, self.y, text=f"You : {user_input}\n", anchor="nw",width=740)
        self.y += 20
        self.entry.delete(0, tk.END)
        self.chat_window.config(state=tk.DISABLED)
        self.update_status("Thinking...")

        # Run response generation in a thread to avoid freezing the GUI
        threading.Thread(target=self.get_and_display_response, args=(user_input,), daemon=True).start()
        return "break" # Prevents the default tkinter event behavior

    def get_and_display_response(self, user_input):
        """Gets response and schedules GUI update from the main thread."""
        try:
            response = self.chatbot_response(user_input)
            self.root.after(0, self.display_response, response)
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            self.root.after(0, self.display_response, error_msg)
            print(error_msg)
    y = 30  # Initial y-coordinate for text placement on canvas
    def display_response(self, response):
        """Displays the chatbot's response in the chat window."""
        self.canvas.config(state=tk.NORMAL)
        self.canvas.create_text(20, self.y, anchor="nw", text=f"dream zone : {response}\n\n", fill="#333333")
        #self.canvas.yview(tk.END)
        self.y += 40
        self.canvas.update_idletasks()
        self.canvas.config(state=tk.DISABLED)
        self.speak_text(response)
        self.update_status("Ready")

    def speak_text(self, text):
        """Convert text to speech in a separate thread."""
        if not self.voice_enabled or not self.engine or self.is_speaking:
            return
        
        def speak():
            try:
                self.is_speaking = True
                clean_text = text.replace("*", "").strip()
                if clean_text:
                    self.engine.say(clean_text)
                    self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.is_speaking = False
        
        threading.Thread(target=speak, daemon=True).start()
    
    def setup_voice_engine(self):
        """Initialize text-to-speech engine."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 0.9)
            print("✓ Voice engine initialized")
        except Exception as e:
            print(f"Voice engine error: {e}")
            self.engine = None
            messagebox.showwarning("Voice Warning", "Text-to-speech not available.")
    
    def setup_speech_recognition(self):
        """Initialize speech recognition."""
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = None
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("✓ Speech recognition initialized")
        except Exception as e:
            print(f"Speech recognition setup error: {e}")
            self.recognizer = None

    def listen_for_speech(self):
        """Listen for voice input in a thread."""
        if not self.recognizer:
            messagebox.showwarning("Not Available", "Speech recognition is not set up.")
            return
        
        self.update_status("Listening...")
        self.listen_button.config(state="disabled", text="🎤 Listening...", bg="#FF6B6B")
        
        def listen():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, self.process_speech_input, text)
            except sr.WaitTimeoutError:
                self.root.after(0, self.update_status, "No speech detected.")
            except sr.UnknownValueError:
                self.root.after(0, self.update_status, "Could not understand speech.")
            except Exception as e:
                self.root.after(0, self.update_status, f"Error: {e}")
            finally:
                self.root.after(0, self.reset_listen_button)
        
        threading.Thread(target=listen, daemon=True).start()

    def process_speech_input(self, text):
        """Process recognized speech by putting it in the entry box and sending it."""
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.send_message()

    def reset_listen_button(self):
        self.listen_button.config(state="normal", text="🎤 Listen", bg="#4CAF50")
        self.update_status("Ready")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def toggle_voice(self):
        """Toggle voice output on/off."""
        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self.voice_button.config(text="🔊 Voice ON", bg="#4CAF50")
            self.speak_text("Voice is now on")
        else:
            if self.is_speaking:
                self.engine.stop()
            self.voice_button.config(text="🔇 Voice OFF", bg="#FF6B6B")
            
    def clear_chat(self):
        """Clear the chat window and show welcome message."""
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.delete(1.0, tk.END)
        welcome = ("Hi there! 👋 I'm dream zone, your voice assistant. How can I help you today?")
        self.canvas.create_text(305, 35, text=f"dream zone: {welcome}\n\n")
        self.chat_window.config(state=tk.DISABLED)
        self.speak_text(welcome)

    def setup_gui(self):
        """Set up the graphical user interface."""
        self.root = tk.Tk()
        self.root.title("dream zone Voice Assistant 🤖")
        self.root.geometry("800x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#FF99E2")
        
        main_frame = tk.Frame(self.root, bg="#FF99E2")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        title_label = tk.Label(main_frame, text="dream zone Voice Assistant 🤖",
                               font=("Arial", 16, "bold"), bg="#FF99E2", fg="#8B008B")
        title_label.pack(pady=(0, 10))
        
        self.chat_window = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, state=tk.DISABLED, 
            font=("Arial", 11), fg="#333333", bg="#ffe9f6", height=20)
        self.chat_window.pack(fill=tk.BOTH, expand=True, pady=10)
        self.canvas = tk.Canvas(self.chat_window)
        self.canvas.config(bg="#FFC4F6", bd=2, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        input_frame = tk.Frame(main_frame, bg="#FF99E2")
        input_frame.pack(fill=tk.X, pady=(0, 10))

         # Add background image
        try:
            # Make sure 'background.png' is in the same directory as the script
            bg_image = Image.open("bg.png") 
            # Use Image.Resampling.LANCZOS which is the modern replacement for ANTIALIAS
            bg_image = bg_image.resize((800, 700), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            # bg_label = tk.Label(self.chat_window, image=self.bg_photo)
            # bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
            self.canvas.image = self.bg_photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Background image error: {e}. Ensure 'bg.png' exists.")
        
        self.entry = tk.Entry(input_frame, font=("Arial", 12), relief=tk.RAISED, bd=2)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.send_message)
        self.entry.focus_set()
        
        self.send_button = tk.Button(
            input_frame, text="Send 📤", command=self.send_message, 
            font=("Arial", 11, "bold"), bg="#F650C7", fg="white",
            relief=tk.RAISED, bd=2, cursor="hand2")
        self.send_button.pack(side=tk.RIGHT)
        
        control_frame = tk.Frame(main_frame, bg="#FF99E2")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.listen_button = tk.Button(
                control_frame, text="🎤 Listen", command=self.listen_for_speech, 
                font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                relief=tk.RAISED, bd=2, cursor="hand2")
            self.listen_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.voice_button = tk.Button(
            control_frame, text="🔊 Voice ON", command=self.toggle_voice, 
            font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
            relief=tk.RAISED, bd=2, cursor="hand2")
        self.voice_button.pack(side=tk.LEFT, padx=(0, 5))
        
        clear_button = tk.Button(
            control_frame, text="🗑️ Clear", command=self.clear_chat, 
            font=("Arial", 10, "bold"), bg="#FF6B6B", fg="white",
            relief=tk.RAISED, bd=2, cursor="hand2")
        clear_button.pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(
            main_frame, text="Status: Initializing...", 
            font=("Arial", 10), bg="#FF99E2", fg="#8B008B")
        self.status_label.pack(pady=(5, 0))
        
        self.root.after(200, self.clear_chat)

    def run(self):
        """Start the assistant's main GUI loop."""
        self.update_status("Ready")
        self.root.mainloop()

if __name__ == "__main__":
    assistant = dreamzoneAssistant()
    assistant.run()