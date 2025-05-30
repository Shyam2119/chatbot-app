# utils/sentiment_analyzer.py
from textblob import TextBlob
import re

class SentimentAnalyzer:
    def __init__(self):
        # Keywords for enhanced sentiment detection
        self.positive_keywords = [
            'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'perfect',
            'love', 'awesome', 'brilliant', 'outstanding', 'superb', 'pleased',
            'satisfied', 'happy', 'delighted', 'impressed', 'recommend'
        ]
        
        self.negative_keywords = [
            'terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting',
            'disappointed', 'frustrated', 'angry', 'annoyed', 'upset', 'mad',
            'furious', 'dissatisfied', 'unhappy', 'complaint', 'problem',
            'issue', 'broken', 'defective', 'useless', 'waste'
        ]
        
        self.urgent_keywords = [
            'urgent', 'emergency', 'asap', 'immediately', 'critical', 'serious',
            'important', 'help', 'stuck', 'broken', 'not working', 'error'
        ]
    
    def analyze(self, text):
        """
        Analyze sentiment of given text
        Returns: 'positive', 'negative', 'neutral', or 'urgent'
        """
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        # Check for urgent keywords first
        if any(keyword in text_lower for keyword in self.urgent_keywords):
            return 'urgent'
        
        # Use TextBlob for basic sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Enhanced sentiment detection using keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Adjust polarity based on keyword counts
        if positive_count > negative_count:
            polarity += 0.2 * positive_count
        elif negative_count > positive_count:
            polarity -= 0.2 * negative_count
        
        # Check for negation patterns
        negation_pattern = r'\b(not|no|never|nothing|nobody|nowhere|neither|hardly|scarcely|barely)\b'
        if re.search(negation_pattern, text_lower):
            polarity *= -0.5  # Flip sentiment if negation is present
        
        # Classify sentiment
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def get_emotion_score(self, text):
        """
        Get more detailed emotion analysis
        Returns: dict with emotion scores
        """
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment': self.analyze(text)
        }
    
    def is_frustrated(self, text):
        """
        Specifically detect frustration indicators
        """
        frustration_indicators = [
            'frustrated', 'annoyed', 'fed up', 'sick of', 'tired of',
            'ridiculous', 'unacceptable', 'this is crazy', 'what the hell',
            'seriously', 'come on', 'are you kidding'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in frustration_indicators)
    
    def needs_escalation(self, text, sentiment):
        """
        Determine if conversation needs escalation based on sentiment and content
        """
        escalation_triggers = [
            'speak to manager', 'human agent', 'real person', 'supervisor',
            'this is ridiculous', 'unacceptable', 'lawyer', 'legal action',
            'complaint', 'report', 'cancel everything', 'close account'
        ]
        
        text_lower = text.lower()
        
        # Check for explicit escalation requests
        if any(trigger in text_lower for trigger in escalation_triggers):
            return True
        
        # Check for very negative sentiment
        if sentiment == 'negative' and self.is_frustrated(text):
            return True
        
        return False