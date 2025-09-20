import tkinter as tk
from tkinter import scrolledtext, messagebox
import random
import json
import pyttsx3
import threading
import os
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
        self.setup_gui()
        self.load_intents()
        self.train_model()
        self.setup_voice_engine()
        self.setup_speech_recognition()
        self.setup_generative_model()
        
        self.is_speaking = False
        self.voice_enabled = True

    def setup_generative_model(self):
        """Initializes the Google Generative AI model."""
        try:
            # Try multiple environment variable names

            # Make sure to set your GEMINI_API_KEY environment variable first
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("API Key not found. Please set the 'GEMINI_API_KEY' environment variable.")
            else:
                genai.configure(api_key=api_key)

                print("Available models that support 'generateContent':")
                # for m in genai.list_models():
                #     if 'generateContent' in m.supported_generation_methods:
                #         print(f"- {m.name}")

                genai.configure(api_key=api_key)
                self.genai_model = genai.GenerativeModel('gemini-2.0-flash')
                print("✓ Gemini model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            messagebox.showwarning("Gemini Error", f"Could not initialize Gemini: {e}\nFalling back to local responses only.")
            self.genai_model = None

    def load_intents(self):
        """Load intents from JSON file or create default ones."""
        try:
            if os.path.exists("intents.json"):
                with open("intents.json", "r", encoding='utf-8') as f:
                    self.intents = json.load(f)
                print("✓ Intents loaded from intents.json")
            else:
                print("Creating default intents.json file...")
                self.intents = self.create_default_intents()
                self.save_intents()
        except Exception as e:
            print(f"Error loading intents: {e}")
            messagebox.showerror("Error", f"Could not load intents: {e}")
            self.intents = self.create_default_intents()
    
    

    def save_intents(self):
        """Save intents to JSON file."""
        try:
            with open("intents.json", "w", encoding='utf-8') as f:
                json.dump(self.intents, f, indent=2, ensure_ascii=False)
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
            self.root.destroy()

    def chatbot_response(self, user_input):
        """
        First, try the local model. If confidence is low, fall back to Gemini.
        """
        if not user_input.strip():
            return "Please say something!"
        
        try:
            # Try local model first
            X_test = self.vectorizer.transform([user_input.lower()])
            probabilities = self.model.predict_proba(X_test)[0]
            max_prob = max(probabilities)
            
            # If confidence is high, use local response
            if max_prob > 0.6:  # Lowered threshold for better coverage
                tag = self.model.classes_[probabilities.argmax()]
                for intent in self.intents["intents"]:
                    if intent["tag"] == tag:
                        return random.choice(intent["responses"])
        except Exception as e:
            print(f"Local model prediction error: {e}")

        # Fall back to Gemini if available
        if self.genai_model:
            try:
                print("Using Gemini for response...")
                # Create a more conversational prompt
                prompt = f"""You are dream zone, a friendly voice assistant. 
                Respond to this message in a helpful, conversational way: {user_input}
                Keep your response concise and friendly."""
                
                response = self.genai_model.generate_content(prompt)
                return response.text if response.text else "I'm thinking, but nothing came to mind. Can you try rephrasing?"
            except Exception as e:
                print(f"Gemini API error: {e}")
                return "I'm having trouble with my advanced features right now. Could you try asking something simpler?"
        
        # Final fallback
        fallback_responses = [
            "That's interesting! Could you tell me more?",
            "I'm not sure about that, but I'm here to chat!",
            "Could you rephrase that? I'd like to help!",
            "That's a good question! I'm still learning about that topic.",
            "Hmm, I'm not quite sure how to respond to that. What else is on your mind?"
        ]
        return random.choice(fallback_responses)

    def send_message(self):
        """Handles sending the user message and displaying the response."""
        user_input = self.entry.get().strip()
        if not user_input:
            return
        
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"You: {user_input}\n")
        self.entry.delete(0, tk.END)
        self.chat_window.config(state=tk.DISABLED)
        self.response = self.chatbot_response(user_input)
        self.display_response(self.response)
        self.update_status("Thinking...")
        
        # Get response in a separate thread to avoid freezing the GUI
        def get_response():
            try:
                response = self.chatbot_response(user_input)
                
            except Exception as e:
                error_msg = f"Error getting response: {e}"
                self.root.after(0, self.display_response, "Sorry, I encountered an error. Please try again!")
                print(error_msg)

        threading.Thread(target=get_response, daemon=True).start()

    def display_response(self, response):
        """Displays the chatbot's response in the chat window."""
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"dream zone: {response}\n\n")
        self.chat_window.yview(tk.END)
        self.chat_window.config(state=tk.DISABLED)
        self.speak_text(response)
        self.update_status("Ready")

    def speak_text(self, text):
        """Convert text to speech in a separate thread."""
        if not self.voice_enabled or not self.engine or self.is_speaking:
            return
        
        def speak():
            try:
                self.is_speaking = True
                # Clean text for better speech
                clean_text = text.replace("✓", "").replace("🤖", "").strip()
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
            voices = self.engine.getProperty('voices')
            
            # Try to set a pleasant voice
            if voices:
                for voice in voices:
                    if any(word in voice.name.lower() for word in ['female', 'zira', 'hazel', 'samantha']):
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.engine.setProperty('rate', 160)  # Slightly slower for clarity
            self.engine.setProperty('volume', 0.9)
            print("✓ Voice engine initialized")
        except Exception as e:
            print(f"Voice engine error: {e}")
            self.engine = None
            messagebox.showwarning("Voice Warning", "Text-to-speech not available. Install with: pip install pyttsx3")
    
    def setup_speech_recognition(self):
        """Initialize speech recognition."""
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = None
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("✓ Speech recognition initialized")
        except Exception as e:
            print(f"Speech recognition setup error: {e}")
            self.recognizer = None

    def listen_for_speech(self):
        """Listen for voice input."""
        if not self.recognizer:
            messagebox.showwarning("Not Available", "Speech recognition is not available. Make sure you have a microphone connected and the required packages installed.")
            return
        
        self.update_status("Listening... (speak now)")
        self.listen_button.config(state="disabled", text="🎤 Listening...", bg="#FF6B6B")
        
        def listen():
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, self.process_speech_input, text)
                
            except sr.WaitTimeoutError:
                self.root.after(0, lambda: self.update_status("No speech detected. Try again."))
            except sr.UnknownValueError:
                self.root.after(0, lambda: self.update_status("Could not understand speech. Try again."))
            except sr.RequestError as e:
                self.root.after(0, lambda: self.update_status(f"Speech service error: {e}"))
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"Error: {e}"))
            finally:
                self.root.after(0, self.reset_listen_button)
        
        threading.Thread(target=listen, daemon=True).start()

    def process_speech_input(self, text):
        """Process recognized speech."""
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.update_status(f"Heard: {text}")
        # Automatically send the message
        self.root.after(1000, self.send_message)  # Small delay to show what was heard

    def reset_listen_button(self):
        """Reset the listen button to normal state."""
        if hasattr(self, 'listen_button'):
            self.listen_button.config(state="normal", text="🎤 Listen", bg="#4CAF50")
        self.update_status("Ready")

    def update_status(self, message):
        """Update the status label."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Status: {message}")

    def toggle_voice(self):
        """Toggle voice output on/off."""
        self.voice_enabled = not self.voice_enabled
        voice_text = "🔊 Voice ON" if self.voice_enabled else "🔇 Voice OFF"
        voice_color = "#4CAF50" if self.voice_enabled else "#FF6B6B"
        self.voice_button.config(text=voice_text, bg=voice_color)
        self.update_status("Voice enabled" if self.voice_enabled else "Voice disabled")
        
        if self.voice_enabled:
            self.speak_text("Voice is now on")
        else:
            # Stop any current speech
            if self.engine and self.is_speaking:
                try:
                    self.engine.stop()
                except:
                    pass

    def clear_chat(self):
        """Clear the chat window and show welcome message."""
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.delete(1.0, tk.END)
        welcome = "Hi there! 👋 I'm dream zone, your voice assistant. How can I help you today?"
        self.chat_window.insert(tk.END, f"dream zone: {welcome}\n\n")
        self.chat_window.config(state=tk.DISABLED)
        self.speak_text("Hi there! I'm dream zone. How can I help you today?")

    def on_enter(self, event):
        """Handle Enter key press."""
        self.send_message()
        return "break"  # Prevent default behavior

    def setup_gui(self):
        """Set up the graphical user interface."""
        self.root = tk.Tk()
        self.root.title("dream zone Voice Assistant 🤖")
        self.root.geometry("800x700")
        self.root.configure(bg="#FF99E2")
        self.root.resizable(True, True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#FF99E2")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_label = tk.Label(main_frame, text="dream zone Voice Assistant 🤖", 
                              font=("Arial", 16, "bold"), bg="#FF99E2", fg="#8B008B")
        title_label.pack(pady=(0, 10))
        
        # Chat window
        self.chat_window = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, state=tk.DISABLED, 
            font=("Arial", 11), bg="#FFE8F7", fg="#333333",
            height=20
        )
        self.chat_window.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg="#FF99E2")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.entry = tk.Entry(input_frame, font=("Arial", 12), relief=tk.RAISED, bd=2)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.on_enter)
        self.entry.focus_set()
        
        self.send_button = tk.Button(
            input_frame, text="Send 📤", command=self.send_message, 
            font=("Arial", 11, "bold"), bg="#F650C7", fg="white",
            relief=tk.RAISED, bd=2, cursor="hand2"
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Control frame
        control_frame = tk.Frame(main_frame, bg="#FF99E2")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Speech recognition button (if available)
        if SPEECH_RECOGNITION_AVAILABLE:
            self.listen_button = tk.Button(
                control_frame, text="🎤 Listen", command=self.listen_for_speech, 
                font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                relief=tk.RAISED, bd=2, cursor="hand2"
            )
            self.listen_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Voice toggle button
        self.voice_button = tk.Button(
            control_frame, text="🔊 Voice ON", command=self.toggle_voice, 
            font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
            relief=tk.RAISED, bd=2, cursor="hand2"
        )
        self.voice_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear button
        clear_button = tk.Button(
            control_frame, text="🗑️ Clear", command=self.clear_chat, 
            font=("Arial", 10, "bold"), bg="#FF6B6B", fg="white",
            relief=tk.RAISED, bd=2, cursor="hand2"
        )
        clear_button.pack(side=tk.RIGHT)
        
        # Status label
        self.status_label = tk.Label(
            main_frame, text="Status: Initializing...", 
            font=("Arial", 10), bg="#FF99E2", fg="#8B008B"
        )
        self.status_label.pack(pady=(5, 0))
        
        # Initialize chat
        self.root.after(100, self.clear_chat)  # Delay to ensure everything is loaded

    def run(self):
        """Start the assistant."""
        try:
            self.update_status("Ready - Type a message or click Listen!")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nShutting down dream zone Assistant...")
        except Exception as e:
            print(f"Runtime error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

def main():
    """Main function to start the assistant."""
    print("=" * 50)
    print("Starting dream zone Voice Assistant...")
    print("=" * 50)
    
    try:
        assistant = dreamzoneAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to start assistant: {e}")
        messagebox.showerror("Startup Error", f"Could not start the assistant: {e}")

if __name__ == "__main__":
    main()