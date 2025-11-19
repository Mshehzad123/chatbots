from flask import Flask, render_template, request, jsonify
from chatbot import CompanyChatbot
import os

app = Flask(__name__)
chatbot = CompanyChatbot()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        language = data.get('language', 'auto')
        
        if not user_message:
            return jsonify({'response': 'Please enter a message!'})
        
        # Get response from chatbot
        response = chatbot.chat(user_message, language)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'response': f'Sorry, an error occurred: {str(e)}',
            'status': 'error'
        })

@app.route('/company-info', methods=['GET'])
def company_info():
    """Get company information"""
    try:
        return jsonify({
            'data': chatbot.company_data,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Softerio Solutions Chatbot Web App")
    print("=" * 60)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

