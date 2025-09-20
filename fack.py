import tkinter as tk
from tkinter import scrolledtext, messagebox
import random
import json
import pyttsx3
import threading
# import os <-- FIX 3: Removed unused import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Optional speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

class DreamZoonAssistant:
    def __init__(self):
        self.setup_gui()
        self.load_intents()
        self.train_model()
        self.setup_voice_engine()
        self.setup_speech_recognition()
        self.is_speaking = False
        self.voice_enabled = True

    # --- FIX 1: ADDED THE MISSING METHOD ---

    def load_intents(self):
        """Load intents from JSON file or create default ones"""
        try:
            with open("intents.json", "r", encoding='utf-8') as f:
                self.intents = json.load(f)
        except FileNotFoundError:
            # Create default intents if file doesn't exist
            self.intents = self.create_default_intents()
            self.save_intents()
    
    def save_intents(self):
        """Save intents to JSON file"""
        try:
            with open("intents.json", "w", encoding='utf-8') as f:
                json.dump(self.intents, f, indent=2)
        except Exception as e:
            print(f"Error saving intents: {e}")

    def train_model(self):
        """Train the machine learning model"""
        patterns, tags = [], []
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                if pattern.strip():  # Skip empty patterns
                    patterns.append(pattern.lower())
                    tags.append(intent["tag"])
        
        if not patterns:
            messagebox.showerror("Error", "No training patterns found in intents.json!")
            self.root.destroy()
            return
            
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(patterns)
        self.model = MultinomialNB()
        self.model.fit(X, tags)

    def setup_voice_engine(self):
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            
            # --- FIX 2: REMOVED UNNECESSARY BLOCKING CALLS ---
            # self.engine.runAndWait() # This can block or cause issues on init
            # self.engine.stop()       # This is not needed here
            
            # Configure voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.engine.setProperty('rate', 170)
            self.engine.setProperty('volume', 1.0)
            
        except Exception as e:
            print(f"Voice engine initialization error: {e}")
            self.engine = None

    def setup_speech_recognition(self):
        """Initialize speech recognition"""
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                print(f"Speech recognition setup error: {e}")
                self.recognizer = None
                self.microphone = None
        else:
            self.recognizer = None
            self.microphone = None

    def chatbot_response(self, user_input):
        # ... code to predict the intent ...
        try:
            X_test = self.vectorizer.transform([user_input.lower()])
            tag = self.model.predict(X_test)[0]
            
            for intent in self.intents["intents"]:
                if intent["tag"] == tag:
                    # If a match is found, it returns a response here
                    return random.choice(intent["responses"])
        
        except Exception as e:
            print(f"Error generating response: {e}")
        
        # If the 'try' block fails or finds no match, this line is executed
        return "Sorry, I didn't understand that. Could you rephrase?"

    def speak_text(self,text):
        """Convert text to speech in a separate thread"""
      
        self.engine = pyttsx3.init()
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()
        print('called')
        
            # finally:
            #     self.is_speaking = False
        
        # speech_thread = threading.Thread(target=speak, daemon=True)
        # speech_thread.start()

    def listen_for_speech(self):
        """Listen for speech input"""
        if not self.recognizer or not self.microphone:
            messagebox.showwarning("Speech Recognition", "Speech recognition is not available. Please install 'SpeechRecognition' and 'PyAudio' packages.")
            return
        
        self.update_status("Listening...")
        self.listen_button.config(state="disabled", text="Listening...", bg="#FF6B6B")
        
        def listen():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, lambda: self.process_speech_input(text))
            except sr.WaitTimeoutError:
                self.root.after(0, lambda: self.update_status("No speech detected. Try again."))
            except sr.UnknownValueError:
                self.root.after(0, lambda: self.update_status("Could not understand audio. Try again."))
            except sr.RequestError as e:
                self.root.after(0, lambda: self.update_status(f"Speech service error: {e}"))
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"Error: {e}"))
            finally:
                self.root.after(0, lambda: self.reset_listen_button())
        
        listen_thread = threading.Thread(target=listen, daemon=True)
        listen_thread.start()

    def process_speech_input(self, text):
        """Process recognized speech input"""
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.update_status(f"Recognized: {text}")
        self.send_message()

    def reset_listen_button(self):
        """Reset the listen button state"""
        self.listen_button.config(state="normal", text="🎤 Listen", bg="#4CAF50")
        self.update_status("Ready")

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=f"Status: {message}")

    def send_message(self):
        """Send message and get response"""
        user_input = self.entry.get()
        if not user_input.strip():
            return
        
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"You: {user_input}\n")
        self.entry.delete(0, tk.END)
        
        response = self.chatbot_response(user_input)
        self.chat_window.insert(tk.END, f"Dream Zoon: {response}\n\n")
        self.chat_window.yview(tk.END)
        self.chat_window.config(state=tk.DISABLED)

        self.speak_text(response)
        self.update_status("Ready")

    def toggle_voice(self):
        """Toggle voice output on/off"""
        self.voice_enabled = not self.voice_enabled
        voice_text = "🔊 Voice ON" if self.voice_enabled else "🔇 Voice OFF"
        voice_color = "#4CAF50" if self.voice_enabled else "#FF6B6B"
        self.voice_button.config(text=voice_text, bg=voice_color)
        
        status = "Voice enabled" if self.voice_enabled else "Voice disabled"
        self.update_status(status)

    def clear_chat(self):
        """Clear the chat window"""
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.delete(1.0, tk.END)
        welcome = "Hi 👋 I'm Dream Zoon Assistant. How can I help you?"
        self.chat_window.insert(tk.END, f"Dream Zoon: {welcome}\n\n")
        self.chat_window.config(state=tk.DISABLED)
        if self.voice_enabled:
            self.speak_text(welcome)

    def on_enter(self, event):
        """Handle Enter key press"""
        self.send_message()

    def setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Dream Zoon Voice Assistant 🤖")
        self.root.geometry("700x700")
        self.root.configure(bg="#FF99E2")
        self.root.resizable(True, True)
        
        main_frame = tk.Frame(self.root, bg="#FF99E2")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = tk.Label(main_frame, text="Dream Zoon Voice Assistant 🤖", 
                               font=("Arial", 16, "bold"), bg="#FF99E2", fg="#FFFFFF")
        title_label.pack(pady=(0, 10))
        
        self.chat_window = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=80, height=20, 
            font=("Arial", 11), bg="#FFE8F7", fg="#333333",
            borderwidth=2, relief="groove"
        )
        self.chat_window.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        input_frame = tk.Frame(main_frame, bg="#FF99E2")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.entry = tk.Entry(input_frame, width=50, font=("Arial", 12), 
                              bg="white", relief="groove", borderwidth=2)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.on_enter)
        
        self.send_button = tk.Button(input_frame, text="Send 📤", command=self.send_message,
                                     font=("Arial", 11, "bold"), bg="#F650C7", fg="white",
                                     relief="raised", borderwidth=2, cursor="hand2")
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        control_frame = tk.Frame(main_frame, bg="#FF99E2")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.listen_button = tk.Button(control_frame, text="🎤 Listen", command=self.listen_for_speech,
                                           font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                                           relief="raised", borderwidth=2, cursor="hand2")
            self.listen_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.voice_button = tk.Button(control_frame, text="🔊 Voice ON", command=self.toggle_voice,
                                      font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                                      relief="raised", borderwidth=2, cursor="hand2")
        self.voice_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(control_frame, text="🗑️ Clear", command=self.clear_chat,
                                 font=("Arial", 10, "bold"), bg="#FF6B6B", fg="white",
                                 relief="raised", borderwidth=2, cursor="hand2")
        clear_button.pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(main_frame, text="Status: Initializing...", 
                                     font=("Arial", 10), bg="#FF99E2", fg="#333333")
        self.status_label.pack(pady=(10, 0))
        
        welcome = "Hi 👋 I'm Dream Zoon Assistant. How can I help you?"
        self.chat_window.insert(tk.END, f"Dream Zoon: {welcome}\n\n")
        self.chat_window.config(state=tk.DISABLED) # Make chat window read-only

    def run(self):
        """Start the assistant"""
        self.update_status("Ready")
        
        # Consistent welcome message
        welcome = "Hi! I'm Dream Zoon Assistant. How can I help you?"
        self.speak_text(welcome)
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Assistant shutting down...")
        finally:
            if self.engine:
                self.engine.stop()

def main():
    """Main function to run the assistant"""
    print("Starting Dream Zoon Voice Assistant...")
    
    if not SPEECH_RECOGNITION_AVAILABLE:
        print("\nOptional dependency 'SpeechRecognition' not found.")
        print("You can install it with: pip install SpeechRecognition PyAudio")
        print("The assistant will work with text input only.\n")
    
    try:
        assistant = DreamZoonAssistant()
        assistant.run()
    except Exception as e:
        print(f"An error occurred while starting the assistant: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


















