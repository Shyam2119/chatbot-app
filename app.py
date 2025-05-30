# app.py
from flask import Flask, request, jsonify, render_template, session
import uuid
import json
import os
import sys
import re
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.intent_model import IntentClassifier
from utils.db_handler import DatabaseHandler
from utils.preprocessor import TextPreprocessor
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.context_manager import ContextManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize components
intent_classifier = IntentClassifier()
db_handler = DatabaseHandler()
preprocessor = TextPreprocessor()
sentiment_analyzer = SentimentAnalyzer()
context_manager = ContextManager()

# Load responses and configuration
with open('data/intents.json', 'r') as file:
    intents_data = json.load(file)

with open('data/config.json', 'r') as file:
    config = json.load(file)

# Train model if not already trained
if not os.path.exists('models/intent_model.h5'):
    print("Training intent model...")
    intent_classifier.build_model()
    print("Model training complete")

@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['start_time'] = datetime.now().isoformat()
    
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    
    try:
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze(user_message)
        
        # Get context from previous conversations
        context = context_manager.get_context(session_id)
        
        # Predict intent with context
        intent = intent_classifier.predict_intent(user_message, context)
        
        # Handle special cases
        response_data = handle_special_cases(user_message, intent, sentiment, session_id)
        
        if not response_data:
            # Get standard response
            response = get_response_for_intent(intent, context, sentiment)
            response_data = {
                'response': response,
                'intent': intent,
                'sentiment': sentiment,
                'confidence': intent_classifier.get_confidence(),
                'suggestions': get_suggestions(intent),
                'quick_replies': get_quick_replies(intent)
            }
        
        # Update context
        context_manager.update_context(session_id, user_message, response_data['response'], intent)
        
        # Store conversation with enhanced data
        conversation_id = db_handler.store_conversation(
            session_id, user_message, response_data['response'], 
            intent, sentiment, response_data.get('confidence', 0.0)
        )
        
        response_data['conversation_id'] = conversation_id
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': "I'm experiencing some technical difficulties. Please try again.",
            'intent': 'error',
            'sentiment': 'neutral'
        }), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    conversation_id = data.get('conversation_id')
    feedback_value = data.get('feedback')
    feedback_text = data.get('feedback_text', '')
    
    if not conversation_id or feedback_value is None:
        return jsonify({'error': 'Missing conversation_id or feedback'}), 400
    
    db_handler.store_feedback(conversation_id, feedback_value, feedback_text)
    
    return jsonify({'success': True})

@app.route('/api/typing', methods=['POST'])
def typing_indicator():
    """Handle typing indicator"""
    return jsonify({'status': 'typing'})

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions_endpoint():
    """Get contextual suggestions"""
    session_id = session.get('session_id')
    context = context_manager.get_context(session_id) if session_id else {}
    
    suggestions = [
        "Check order status",
        "Return policy",
        "Track my package",
        "Contact support",
        "Product information"
    ]
    
    return jsonify({'suggestions': suggestions})

@app.route('/api/analytics', methods=['GET'])
def analytics():
    intent = request.args.get('intent')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    analytics_data = db_handler.get_analytics(intent, date_from, date_to)
    
    return jsonify(analytics_data)

@app.route('/api/export', methods=['GET'])
def export_conversations():
    """Export conversations for analysis"""
    format_type = request.args.get('format', 'json')
    session_id = request.args.get('session_id')
    
    conversations = db_handler.export_conversations(session_id, format_type)
    
    return jsonify({'data': conversations})

def handle_special_cases(message, intent, sentiment, session_id):
    """Handle special cases like escalation, multilingual support, etc."""
    
    # Check for escalation keywords
    escalation_keywords = ['human', 'agent', 'speak to someone', 'manager', 'supervisor']
    if any(keyword in message.lower() for keyword in escalation_keywords):
        return {
            'response': "I understand you'd like to speak with a human agent. Let me connect you with our support team. Please hold on while I transfer your conversation.",
            'intent': 'escalation',
            'sentiment': sentiment,
            'escalation': True,
            'quick_replies': ['Continue with bot', 'Wait for agent']
        }
    
    # Handle frustrated users
    if sentiment == 'negative' and any(word in message.lower() for word in ['frustrated', 'angry', 'terrible', 'awful']):
        return {
            'response': "I understand your frustration, and I sincerely apologize for any inconvenience. Let me do my best to help resolve this issue quickly. What specific problem can I assist you with?",
            'intent': 'empathy',
            'sentiment': sentiment,
            'empathy_response': True
        }
    
    # Multi-turn conversation handling
    context = context_manager.get_context(session_id)
    if context.get('waiting_for_order_number') and re.search(r'[A-Z0-9]{6,}', message):
        order_number = re.search(r'[A-Z0-9]{6,}', message).group()
        return {
            'response': f"Thank you! I found your order {order_number}. Your order is currently being processed and will be shipped within 2-3 business days. You'll receive tracking information via email once it's dispatched.",
            'intent': 'order_status_provided',
            'sentiment': sentiment,
            'order_number': order_number
        }
    
    return None

def get_response_for_intent(intent, context=None, sentiment=None):
    """Enhanced response generation with context and sentiment awareness"""
    
    # Find intent responses
    for intent_data in intents_data['intents']:
        if intent_data['tag'] == intent:
            import random
            
            # Choose response based on sentiment if available
            if sentiment == 'negative' and 'negative_responses' in intent_data:
                return random.choice(intent_data['negative_responses'])
            elif sentiment == 'positive' and 'positive_responses' in intent_data:
                return random.choice(intent_data['positive_responses'])
            else:
                return random.choice(intent_data['responses'])
    
    # Fallback responses based on sentiment
    if sentiment == 'negative':
        return "I understand this might be frustrating. Let me help you find the right information. Could you please provide more details about what you're looking for?"
    elif sentiment == 'positive':
        return "I'm glad to help! Could you please provide more details about what you need assistance with?"
    
    return "I'm not sure how to respond to that. Could you please rephrase or ask something else? Here are some things I can help with: order status, returns, product information, or general support."

def get_suggestions(intent):
    """Get contextual suggestions based on intent"""
    suggestions_map = {
        'greeting': ['Check order status', 'Return policy', 'Track package'],
        'order_status': ['Track package', 'Cancel order', 'Change address'],
        'return_policy': ['Start return', 'Exchange item', 'Refund status'],
        'goodbye': ['Rate this conversation', 'Contact support', 'Visit help center']
    }
    
    return suggestions_map.get(intent, ['How can I help?', 'Contact support'])

def get_quick_replies(intent):
    """Get quick reply options based on intent"""
    quick_replies_map = {
        'greeting': ['Check my order', 'Return an item', 'General question'],
        'order_status': ['Yes, that helps', 'I need more info', 'Track different order'],
        'return_policy': ['Start return process', 'Ask about exchange', 'More questions'],
        'thanks': ['You\'re welcome!', 'Anything else?', 'Rate this chat']
    }
    
    return quick_replies_map.get(intent, [])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))