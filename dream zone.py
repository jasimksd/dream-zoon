import tkinter as tk
from tkinter import scrolledtext, messagebox
import random
import json
import pyttsx3
import threading
import os
import webbrowser
from datetime import datetime
from PIL import Image, ImageTk

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
        self.chat_history = []  # store all messages
        self.setup_generative_model()
        self.load_intents()
        self.train_model()
        self.setup_voice_engine()
        self.setup_speech_recognition()
        self.setup_gui()

    # ==========================================================
    # SETUP METHODS
    # ==========================================================
    def setup_generative_model(self):
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("API Key not found. Please set GEMINI_API_KEY.")
                messagebox.showwarning("API Key Missing", "Gemini API key not found. Using local model only.")
                return
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel("gemini-2.5-flash")
            print("✓ Gemini model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            messagebox.showwarning("Gemini Error", f"Could not initialize Gemini: {e}")

    def load_intents(self):
        try:
            if os.path.exists("intents.json"):
                with open("intents.json", "r", encoding="utf-8") as f:
                    self.intents = json.load(f)
                print("✓ Intents loaded.")
            else:
                self.intents = self.create_default_intents()
                self.save_intents()
        except Exception as e:
            print(f"Error loading intents: {e}")
            self.intents = self.create_default_intents()

    def create_default_intents(self):
        return {
            "intents": [
                {"tag": "greeting", "patterns": ["hi", "hello", "hey"], "responses": ["Hey!", "Hi there!", "Hello!"]},
                {"tag": "goodbye", "patterns": ["bye", "goodbye"], "responses": ["Goodbye!", "See you later!"]},
                {"tag": "thanks", "patterns": ["thanks", "thank you"], "responses": ["You're welcome!", "No problem!"]},
                {"tag": "about", "patterns": ["who are you", "what are you"], "responses": ["I'm Dream Zone Assistant."]}
            ]
        }

    def save_intents(self):
        with open("intents.json", "w", encoding="utf-8") as f:
            json.dump(self.intents, f, indent=4, ensure_ascii=False)

    def train_model(self):
        patterns, tags = [], []
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                patterns.append(pattern.lower())
                tags.append(intent["tag"])

        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(patterns)
        self.model = MultinomialNB()
        self.model.fit(X, tags)
        print("✓ Local model trained.")

    def setup_voice_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)
            self.engine.setProperty("volume", 0.9)
            print("✓ Voice engine ready.")
        except Exception as e:
            print(f"Voice engine error: {e}")
            self.engine = None

    def setup_speech_recognition(self):
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = None
            return
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("✓ Speech recognition ready.")
        except Exception as e:
            print(f"Speech recognition error: {e}")
            self.recognizer = None

    # ==========================================================
    # CHAT / RESPONSE HANDLING
    # ==========================================================
    def _handle_hardcoded_commands(self, text):
        t = text.lower()
        if "time" in t:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."
        if "date" in t:
            return f"Today is {datetime.now().strftime('%B %d, %Y')}."
        if "open youtube" in t:
            webbrowser.open("https://www.youtube.com/@CDCKasaragod/videos")
            return "Opening YouTube..."
        if "open web" in t:
            webbrowser.open("https://www.dreamzone.co.in/")
            return "Opening website..."
        if "open instagram" in t:
            webbrowser.open("https://www.instagram.com/dreamzone_kasaragod/?hl=en")
            return "Opening Instagram..."
        if "open location" in t or "where is dream zone" in t:
            webbrowser.open("https://www.google.com/maps/dir//3rd+Floor,+Square+Nine+mall,+New+Busstand+Junction,+NH+66,+Kasaragod,+Kerala+671121/@12.5168109,74.9576239,7052m/data=!3m2!1e3!5s0x3ba4825c93621d93:0x828a928d106d5dc3!4m8!4m7!1m0!1m5!1m1!1s0x3ba4825cecc298f5:0x781fa60b9fc5edc0!2m2!1d74.9955793!2d12.5069604?entry=ttu&g_ep=EgoyMDI1MTIwOS4wIKXMDSoASAFQAw%3D%3D")
            return "Opening Dream Zone location..."
        if "open facebook" in t :
            webbrowser.open("https://www.facebook.com/dreamzone.ksd/")
        return None

    def chatbot_response(self, user_input):
        if not user_input.strip():
            return "Please say something!"

        # 1️⃣ Hardcoded commands
        quick = self._handle_hardcoded_commands(user_input)
        if quick:
            return quick

        # 2️⃣ Local model
        X_test = self.vectorizer.transform([user_input.lower()])
        probs = self.model.predict_proba(X_test)[0]
        max_prob = max(probs)
        if max_prob > 0.65:
            tag = self.model.classes_[probs.argmax()]
            for intent in self.intents["intents"]:
                if intent["tag"] == tag:
                    return random.choice(intent["responses"])

        # 3️⃣ Gemini AI
        if self.genai_model:
            try:
                prompt = f"You are Dream Zone, a helpful assistant. Respond concisely to: {user_input}"
                resp = self.genai_model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                print(f"Gemini error: {e}")
                return "I'm having trouble connecting right now."

        # 4️⃣ Fallback
        return "I'm not sure how to respond to that."

    # ==========================================================
    # VOICE + CHAT DISPLAY
    # ==========================================================
    def speak_text(self, text):
        if not self.voice_enabled or not self.engine:
            return
        def speak():
            try:
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.is_speaking = False
        threading.Thread(target=speak, daemon=True).start()

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)
        self.display_text(f"You: {user_input}")
        self.chat_history.append({"user": user_input})
        self.update_status("Thinking...")

        threading.Thread(target=self._process_message, args=(user_input,), daemon=True).start()
        return "break"

    def _process_message(self, user_input):
        response = self.chatbot_response(user_input)
        self.root.after(0, self.display_response, response)

    def display_text(self, text):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, text + "\n")
        self.chat_window.yview(tk.END)
        self.chat_window.config(state=tk.DISABLED)

    def display_response(self, response):
        self.display_text(f"Dream Zone: {response}\n")
        self.chat_history.append({"assistant": response})
        self.speak_text(response)
        self.update_status("Ready")

    def clear_chat(self):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.delete("1.0", tk.END)
        self.chat_window.config(state=tk.DISABLED)
        self.chat_history.clear()
        self.display_response("Hi there! 👋 I'm Dream Zone. How can I help you today?")

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self.voice_button.config(text="🔊 Voice ON", bg="#4CAF50")
            self.speak_text("Voice enabled")
        else:
            if self.is_speaking:
                self.engine.stop()
            self.voice_button.config(text="🔇 Voice OFF", bg="#FF6B6B")

    def listen_for_speech(self):
        if not self.recognizer:
            messagebox.showwarning("Unavailable", "Speech recognition not available.")
            return
        self.listen_button.config(state="disabled", text="🎤 Listening...", bg="#FF6B6B")
        self.update_status("Listening...")
        def listen():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, lambda: self.entry.insert(0, text))
                self.root.after(0, self.send_message)
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"Error: {e}"))
            finally:
                self.root.after(0, self.reset_listen_button)
        threading.Thread(target=listen, daemon=True).start()

    def reset_listen_button(self):
        self.listen_button.config(state="normal", text="🎤 Listen", bg="#4CAF50")
        self.update_status("Ready")

    # ==========================================================
    # UTILITIES
    # ==========================================================
    def get_all_variables(self):
        """Return all key variables as a single dictionary."""
        return {
            "intents": self.intents,
            "voice_enabled": self.voice_enabled,
            "is_speaking": self.is_speaking,
            "genai_model": self.genai_model,
            "vectorizer": self.vectorizer,
            "model": self.model,
            "recognizer": getattr(self, "recognizer", None),
            "microphone": getattr(self, "microphone", None),
            "chat_history": self.chat_history,
        }

    def export_chat_history(self):
        """Save the chat history to a file."""
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            for entry in self.chat_history:
                for role, msg in entry.items():
                    f.write(f"{role}: {msg}\n")
        messagebox.showinfo("Exported", "Chat history saved to chat_history.txt")

    def update_status(self, msg):
        self.status_label.config(text=f"Status: {msg}")

    # ==========================================================
    # GUI SETUP
    # ==========================================================
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Dream Zone Voice Assistant 🤖")
        self.root.geometry("800x700")
        self.root.configure(bg="#FF99E2")

        # --- ICON HANDLING ---
        ico_path = "D:/dream zoon/image/removebg.ico"
        png_path = "D:/dream zoon/image/removebg.png"

        # Convert PNG → ICO if missing
        if os.path.exists(png_path) and not os.path.exists(ico_path):
            try:
                img = Image.open(png_path)
                img.save(ico_path, format="ICO")
            except Exception as e:
                print(f"⚠️ Could not convert PNG to ICO: {e}")

        # Try loading icon safely
        try:
            if os.name == "nt" and os.path.exists(ico_path):
                self.root.iconbitmap(ico_path)
            elif os.path.exists(png_path):
                logo_img = Image.open(png_path)
                logo_img = logo_img.resize((32, 32), Image.Resampling.LANCZOS)
                self.logo_icon = ImageTk.PhotoImage(logo_img)
                self.root.iconphoto(False, self.logo_icon)
        except Exception as e:
            print(f"⚠️ Icon load error: {e}")

        # --- GUI WIDGETS ---
        title = tk.Label(self.root, text="Dream Zone Voice Assistant 🤖",
                        font=("Arial", 18, "bold"), bg="#FF99E2", fg="#8B008B")
        title.pack(pady=10)

        self.chat_window = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, bg="#FFEAF9", font=("Arial", 12)
        )
        self.chat_window.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.chat_window.config(state=tk.DISABLED)

        input_frame = tk.Frame(self.root, bg="#FF99E2")
        input_frame.pack(fill=tk.X, padx=15, pady=5)

        self.entry = tk.Entry(input_frame, font=("Arial", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        send_btn = tk.Button(input_frame, text="Send 📤", command=self.send_message,
                            bg="#F650C7", fg="white", font=("Arial", 11, "bold"))
        send_btn.pack(side=tk.RIGHT)

        ctrl_frame = tk.Frame(self.root, bg="#FF99E2")
        ctrl_frame.pack(fill=tk.X, padx=15, pady=5)

        if SPEECH_RECOGNITION_AVAILABLE:
            self.listen_button = tk.Button(ctrl_frame, text="🎤 Listen", command=self.listen_for_speech,
                                        bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
            self.listen_button.pack(side=tk.LEFT, padx=5)

        self.voice_button = tk.Button(ctrl_frame, text="🔊 Voice ON", command=self.toggle_voice,
                                    bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.voice_button.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(ctrl_frame, text="🗑️ Clear", command=self.clear_chat,
                            bg="#FF6B6B", fg="white", font=("Arial", 10, "bold"))
        clear_btn.pack(side=tk.RIGHT, padx=5)

        export_btn = tk.Button(ctrl_frame, text="💾 Export Chat", command=self.export_chat_history,
                            bg="#9370DB", fg="white", font=("Arial", 10, "bold"))
        export_btn.pack(side=tk.RIGHT, padx=5)

        self.status_label = tk.Label(self.root, text="Status: Ready",
                                    bg="#FF99E2", fg="#8B008B", font=("Arial", 10))
        self.status_label.pack(pady=(0, 10))

        self.clear_chat()

    # ==========================================================
    # RUN APP
    # ==========================================================
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    assistant = dreamzoneAssistant()
    assistant.run()
