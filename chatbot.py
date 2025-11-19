import json
import os
from typing import Dict, List, Optional
import re

# Hugging Face Transformers - Auto download on first use
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not installed. Install with: pip install transformers torch")

class CompanyChatbot:
    def __init__(self, data_file: str = "company_data.json"):
        """Initialize the chatbot with company data"""
        self.data_file = data_file
        self.company_data = self.load_company_data()
        self.generator = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading AI model (first time will download ~500MB)...")
                print("   This may take a few minutes on first run...")
                # Using distilgpt2 - small, fast, auto-downloads
                self.generator = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    tokenizer="distilgpt2",
                    device=-1,  # Use CPU (change to 0 for GPU if available)
                    max_length=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=50256  # GPT-2 pad token
                )
                print("✓ AI model loaded successfully!")
            except Exception as e:
                print(f"⚠ Warning: Could not load AI model: {e}")
                print("Falling back to basic FAQ matching...")
                self.generator = None
        
    def load_company_data(self) -> Dict:
        """Load company data from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.data_file} not found!")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.data_file}")
            return {}
    
    def get_context(self) -> str:
        """Generate context string from company data"""
        context = f"""
Company Information:
- Company Name: {self.company_data.get('company_name', 'N/A')}
- Tagline: {self.company_data.get('company_tagline', 'N/A')}

Contact Details:
- Email: {self.company_data.get('contact', {}).get('email', 'N/A')}
- Phone: {self.company_data.get('contact', {}).get('phone', 'N/A')}
- Address: {self.company_data.get('contact', {}).get('address', 'N/A')}
- Website: {self.company_data.get('contact', {}).get('website', 'N/A')}

About:
{self.company_data.get('about', {}).get('description', 'N/A')}
- Founded: {self.company_data.get('about', {}).get('founded', 'N/A')}
- Team Size: {self.company_data.get('about', {}).get('team_size', 'N/A')}
- Clients: {self.company_data.get('about', {}).get('clients', 'N/A')}

Services:
"""
        for service in self.company_data.get('services', []):
            context += f"- {service.get('service_name', 'N/A')}: {service.get('description', 'N/A')}\n"
        
        context += "\nFrequently Asked Questions:\n"
        for faq in self.company_data.get('faq', []):
            context += f"Q: {faq.get('question', 'N/A')}\n"
            context += f"A: {faq.get('answer', 'N/A')}\n\n"
        
        return context
    
    def find_relevant_faq(self, question: str) -> Optional[str]:
        """Find relevant FAQ answer based on keywords"""
        question_lower = question.lower()
        
        # Check for company name
        if any(word in question_lower for word in ['company name', 'name', 'kaun', 'kya naam']):
            return self.company_data.get('company_name', 'Softerio Solutions')
        
        # Check for contact info
        if any(word in question_lower for word in ['contact', 'email', 'phone', 'address', 'number', 'raabta']):
            contact = self.company_data.get('contact', {})
            return f"Email: {contact.get('email')}, Phone: {contact.get('phone')}, Address: {contact.get('address')}, Website: {contact.get('website')}"
        
        # Check for services
        if any(word in question_lower for word in ['service', 'services', 'kya karte', 'offer', 'provide']):
            services = [s.get('service_name') for s in self.company_data.get('services', [])]
            return f"We provide: {', '.join(services)}"
        
        # Check FAQ list
        for faq in self.company_data.get('faq', []):
            faq_q = faq.get('question', '').lower()
            if any(word in faq_q for word in question_lower.split()[:3]):
                return faq.get('answer')
        
        return None
    
    def chat(self, user_input: str, language: str = "auto") -> str:
        """Main chat function"""
        user_input = user_input.strip()
        if not user_input:
            return "Please ask me something about Softerio Solutions!"
        
        # First, try to find direct answer from FAQ/data
        direct_answer = self.find_relevant_faq(user_input)
        
        # If Transformers model is available, use it for better responses
        if self.generator and TRANSFORMERS_AVAILABLE:
            try:
                context = self.get_context()
                
                # Language instruction
                lang_instruction = ""
                if language == "urdu" or language == "ur":
                    lang_instruction = "Respond in Urdu."
                elif language == "english" or language == "en":
                    lang_instruction = "Respond in English."
                
                # Create prompt with context
                prompt = f"""Softerio Solutions Chatbot

Company Info:
{context}

User: {user_input}
Bot:"""
                
                # Generate response
                result = self.generator(
                    prompt,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    truncation=True,
                    pad_token_id=50256
                )
                
                # Extract generated text
                generated_text = result[0]['generated_text']
                
                # Extract only the bot's response (after "Bot:")
                if "Bot:" in generated_text:
                    answer = generated_text.split("Bot:")[-1].strip()
                    # Clean up the response
                    answer = answer.split('\n')[0].strip()  # Take first line
                    # Remove incomplete sentences at the end
                    if answer and len(answer) > 10:
                        # If we have a direct answer and model answer is too short, prefer direct
                        if direct_answer and len(answer) < 30:
                            return direct_answer
                        return answer
                
                # If generation failed, use direct answer
                if direct_answer:
                    return direct_answer
                return "I understand your question. Let me help you with information about Softerio Solutions."
                
            except Exception as e:
                print(f"Error with AI model: {e}")
                # Fallback to direct answer
                if direct_answer:
                    return direct_answer
                return "I'm having trouble processing that. Let me help you with basic information."
        
        # Fallback: Use direct FAQ matching
        if direct_answer:
            return direct_answer
        
        # Generic response if nothing matches
        return f"I understand you're asking about '{user_input}'. Let me help you with information about Softerio Solutions. Could you be more specific? For example, you can ask about our company name, contact details, or services."

def main():
    """Main function to run the chatbot"""
    print("=" * 60)
    print("🤖 Softerio Solutions Chatbot")
    print("=" * 60)
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'lang urdu' or 'lang english' to change language")
    print("Type 'help' for more information\n")
    
    chatbot = CompanyChatbot()
    current_language = "auto"
    
    if not chatbot.generator:
        print("\n⚠ Note: AI model not loaded. Using basic FAQ matching.")
        print("For better responses, make sure transformers is installed:")
        print("pip install transformers torch\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for chatting with Softerio Solutions! 👋")
                break
            
            if user_input.lower().startswith('lang '):
                lang = user_input.lower().split(' ', 1)[1]
                current_language = lang
                print(f"Language set to: {lang}")
                continue
            
            if user_input.lower() == 'help':
                print("\nYou can ask me about:")
                print("- Company name")
                print("- Contact information")
                print("- Services we provide")
                print("- About the company")
                print("- Any other questions about Softerio Solutions")
                continue
            
            response = chatbot.chat(user_input, current_language)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nThank you for chatting! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()

