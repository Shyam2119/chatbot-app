# utils/context_manager.py
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta

class ContextManager:
    def __init__(self, max_context_length=10):
        self.contexts = defaultdict(lambda: {
            'conversation_history': deque(maxlen=max_context_length),
            'user_info': {},
            'current_intent': None,
            'waiting_for_info': None,
            'last_activity': None,
            'session_start': datetime.now(),
            'preferences': {},
            'resolved_issues': [],
            'escalation_count': 0
        })
        self.max_context_length = max_context_length
    
    def update_context(self, session_id, user_message, bot_response, intent):
        """Update conversation context for a session"""
        context = self.contexts[session_id]
        
        # Add to conversation history
        context['conversation_history'].append({
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update current intent and activity time
        context['current_intent'] = intent
        context['last_activity'] = datetime.now()
        
        # Extract and store user information
        self._extract_user_info(user_message, context)
        
        # Handle multi-turn conversations
        self._handle_multi_turn(user_message, intent, context)
    
    def get_context(self, session_id):
        """Get current context for a session"""
        if session_id not in self.contexts:
            return {}
        
        context = self.contexts[session_id]
        
        # Clean up old contexts (older than 1 hour of inactivity)
        if (context['last_activity'] and 
            datetime.now() - context['last_activity'] > timedelta(hours=1)):
            del self.contexts[session_id]
            return {}
        
        return dict(context)
    
    def _extract_user_info(self, message, context):
        """Extract user information from messages"""
        import re
        
        # Extract order numbers
        order_match = re.search(r'\b[A-Z0-9]{6,}\b', message.upper())
        if order_match:
            context['user_info']['last_order_number'] = order_match.group()
        
        # Extract email addresses
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
        if email_match:
            context['user_info']['email'] = email_match.group()
        
        # Extract phone numbers
        phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message)
        if phone_match:
            context['user_info']['phone'] = phone_match.group()
        
        # Extract names (simple heuristic)
        if 'my name is' in message.lower():
            name_match = re.search(r'my name is ([A-Za-z\s]+)', message.lower())
            if name_match:
                context['user_info']['name'] = name_match.group(1).strip().title()
    
    def _handle_multi_turn(self, message, intent, context):
        """Handle multi-turn conversation logic"""
        
        # If asking for order status without providing order number
        if intent == 'order_status' and 'order_number' not in context['user_info']:
            context['waiting_for_info'] = 'order_number'
        
        # If asking about returns without order info
        elif intent == 'return_policy' and not any(key in context['user_info'] 
                                                  for key in ['order_number', 'email']):
            context['waiting_for_info'] = 'order_info'
        
        # Clear waiting state if info is provided
        elif context.get('waiting_for_info'):
            if (context['waiting_for_info'] == 'order_number' and 
                'last_order_number' in context['user_info']):
                context['waiting_for_info'] = None
            elif (context['waiting_for_info'] == 'order_info' and 
                  ('last_order_number' in context['user_info'] or 'email' in context['user_info'])):
                context['waiting_for_info'] = None
    
    def is_returning_user(self, session_id):
        """Check if user has previous context"""
        return (session_id in self.contexts and 
                len(self.contexts[session_id]['conversation_history']) > 1)
    
    def get_conversation_summary(self, session_id):
        """Get a summary of the conversation"""
        if session_id not in self.contexts:
            return None
        
        context = self.contexts[session_id]
        history = list(context['conversation_history'])
        
        if not history:
            return None
        
        # Count intents
        intent_counts = {}
        for entry in history:
            intent = entry['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            'message_count': len(history),
            'session_duration': (datetime.now() - context['session_start']).total_seconds() / 60,
            'most_common_intent': max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else None,
            'user_info': context['user_info'],
            'issues_discussed': list(intent_counts.keys()),
            'escalation_count': context['escalation_count']
        }
    
    def mark_escalation(self, session_id):
        """Mark that this session required escalation"""
        if session_id in self.contexts:
            self.contexts[session_id]['escalation_count'] += 1
    
    def add_resolved_issue(self, session_id, issue):
        """Add a resolved issue to the context"""
        if session_id in self.contexts:
            self.contexts[session_id]['resolved_issues'].append({
                'issue': issue,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_user_preferences(self, session_id):
        """Get user preferences from context"""
        if session_id in self.contexts:
            return self.contexts[session_id]['preferences']
        return {}
    
    def set_user_preference(self, session_id, key, value):
        """Set a user preference"""
        if session_id in self.contexts:
            self.contexts[session_id]['preferences'][key] = value
    
    def cleanup_old_contexts(self):
        """Clean up contexts older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        sessions_to_remove = []
        
        for session_id, context in self.contexts.items():
            if (context['last_activity'] and context['last_activity'] < cutoff_time):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.contexts[session_id]