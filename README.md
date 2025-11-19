# Softerio Solutions Chatbot 🤖

A multilingual chatbot for Softerio Solutions that can answer questions about the company, services, and contact information. Works completely offline with no API keys required!

## Features ✨

- ✅ **No API Key Required** - Uses Hugging Face Transformers (auto-downloads)
- ✅ **Multilingual Support** - Supports Urdu, English, and auto-detection
- ✅ **Company Information** - Answers questions about Softerio Solutions
- ✅ **Beautiful Web Interface** - Modern, responsive UI
- ✅ **Easy to Customize** - All company data in JSON format

## Installation & Setup 🚀

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Model Auto-Download

The AI model (distilgpt2) will **automatically download** on first run (~500MB).
- First run may take 5-10 minutes to download
- Subsequent runs will be instant
- No manual download needed!

**Note:** If transformers is not installed, the chatbot will still work with basic FAQ matching.

### Step 3: Update Company Data

Edit `company_data.json` to add your company information:
- Company name, contact details
- Services offered
- FAQ questions and answers
- Any other relevant information

### Step 4: Run the Chatbot

**Option A: Web Interface (Recommended)**
```bash
python app.py
```
Then open your browser and go to: `http://localhost:5000`

**Option B: Command Line Interface**
```bash
python chatbot.py
```

## Usage 📖

### Web Interface
1. Start the server: `python app.py`
2. Open browser: `http://localhost:5000`
3. Select language (Auto/English/Urdu)
4. Start chatting!

### Command Line
1. Run: `python chatbot.py`
2. Type your questions
3. Use `lang urdu` or `lang english` to change language
4. Type `quit` to exit

## Example Questions 💬

- "What is the company name?"
- "What services do you provide?"
- "How can I contact you?"
- "Tell me about your AI services"
- "Company ka naam kya hai?" (Urdu)
- "Aap kya services dete hain?" (Urdu)

## Project Structure 📁

```
company models/
├── company_data.json      # Company information dataset
├── chatbot.py             # Main chatbot logic
├── app.py                 # Flask web application
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Customization 🎨

### Adding More Company Data

Edit `company_data.json` to add:
- More services
- Additional FAQ entries
- Company details
- Contact information

### Changing the Model

In `chatbot.py`, you can change the model:
```python
self.generator = pipeline(
    "text-generation",
    model="distilgpt2",  # Change to: "gpt2", "microsoft/DialoGPT-small", etc.
    ...
)
```

Available models: `distilgpt2` (smallest), `gpt2`, `microsoft/DialoGPT-small`, etc.

## Troubleshooting 🔧

### Model Download Issues
- First run downloads model automatically (be patient, ~500MB)
- Check internet connection for first-time download
- Model saves in: `~/.cache/huggingface/` (Windows: `C:\Users\YourName\.cache\huggingface\`)

### Port Already in Use
- Change port in `app.py`: `app.run(port=5001)`

### JSON File Not Found
- Make sure `company_data.json` is in the same directory
- Check file name spelling

## Technologies Used 🛠️

- **Python 3.7+**
- **Flask** - Web framework
- **Hugging Face Transformers** - Local AI model (auto-downloads)
- **HTML/CSS/JavaScript** - Frontend
- **JSON** - Data storage

## License 📝

Free to use and modify for your company!

## Support 💬

For issues or questions, check the company_data.json file and chatbot.py code.

---

**Made for Softerio Solutions** 🚀

