# utils/db_handler.py
import sqlite3
import datetime
import json
from typing import List, Dict, Optional

class DatabaseHandler:
    def __init__(self, db_file='database/conversations.db'):
        self.db_file = db_file
        self.create_tables()
    
    def create_tables(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Enhanced conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            intent TEXT NOT NULL,
            sentiment TEXT,
            confidence REAL,
            timestamp DATETIME NOT NULL,
            feedback INTEGER DEFAULT 0,
            feedback_text TEXT,
            response_time REAL,
            escalated BOOLEAN DEFAULT 0,
            resolved BOOLEAN DEFAULT 0
        )
        ''')
        
        # User sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            start_time DATETIME NOT NULL,
            end_time DATETIME,
            total_messages INTEGER DEFAULT 0,
            satisfaction_score REAL,
            issues_resolved INTEGER DEFAULT 0,
            escalated BOOLEAN DEFAULT 0
        )
        ''')
        
        # Analytics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            total_conversations INTEGER DEFAULT 0,
            unique_users INTEGER DEFAULT 0,
            avg_satisfaction REAL DEFAULT 0,
            most_common_intent TEXT,
            resolution_rate REAL DEFAULT 0,
            avg_response_time REAL DEFAULT 0
        )
        ''')
        
        # Intent performance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS intent_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent TEXT NOT NULL,
            date DATE NOT NULL,
            count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_satisfaction REAL DEFAULT 0,
            success_rate REAL DEFAULT 0
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, session_id, user_message, bot_response, intent, 
                          sentiment=None, confidence=0.0, response_time=None):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
        INSERT INTO conversations (session_id, user_message, bot_response, intent, 
                                 sentiment, confidence, timestamp, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, intent, sentiment, 
              confidence, timestamp, response_time))
        
        conversation_id = cursor.lastrowid
        
        # Update session info
        cursor.execute('''
        INSERT OR REPLACE INTO user_sessions (session_id, start_time, total_messages)
        VALUES (?, ?, COALESCE((SELECT total_messages FROM user_sessions WHERE session_id = ?), 0) + 1)
        ''', (session_id, timestamp, session_id))
        
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def store_feedback(self, conversation_id, feedback, feedback_text=''):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE conversations
        SET feedback = ?, feedback_text = ?
        WHERE id = ?
        ''', (feedback, feedback_text, conversation_id))
        
        conn.commit()
        conn.close()
    
    def get_analytics(self, intent=None, date_from=None, date_to=None):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Base query
        where_conditions = []
        params = []
        
        if intent:
            where_conditions.append("intent = ?")
            params.append(intent)
        
        if date_from:
            where_conditions.append("date(timestamp) >= ?")
            params.append(date_from)
            
        if date_to:
            where_conditions.append("date(timestamp) <= ?")
            params.append(date_to)
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Get conversation statistics
        cursor.execute(f'''
        SELECT 
            COUNT(*) as total_conversations,
            COUNT(DISTINCT session_id) as unique_sessions,
            AVG(CASE WHEN feedback > 0 THEN feedback END) as avg_satisfaction,
            AVG(confidence) as avg_confidence,
            COUNT(CASE WHEN feedback >= 4 THEN 1 END) * 100.0 / COUNT(*) as satisfaction_rate,
            COUNT(CASE WHEN escalated = 1 THEN 1 END) * 100.0 / COUNT(*) as escalation_rate
        FROM conversations
        {where_clause}
        ''', params)
        
        stats = cursor.fetchone()
        
        # Get intent distribution
        cursor.execute(f'''
        SELECT intent, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM conversations
        {where_clause}
        GROUP BY intent
        ORDER BY count DESC
        LIMIT 10
        ''', params)
        
        intent_distribution = cursor.fetchall()
        
        # Get daily conversation counts
        cursor.execute(f'''
        SELECT date(timestamp) as date, COUNT(*) as count
        FROM conversations
        {where_clause}
        GROUP BY date(timestamp)
        ORDER BY date DESC
        LIMIT 30
        ''', params)
        
        daily_counts = cursor.fetchall()
        
        # Get sentiment distribution
        cursor.execute(f'''
        SELECT sentiment, COUNT(*) as count
        FROM conversations
        WHERE sentiment IS NOT NULL {" AND " + " AND ".join(where_conditions) if where_conditions else ""}
        GROUP BY sentiment
        ''', params)
        
        sentiment_distribution = cursor.fetchall()
        
        conn.close()
        
        return {
            'statistics': {
                'total_conversations': stats[0] or 0,
                'unique_sessions': stats[1] or 0,
                'avg_satisfaction': round(stats[2] or 0, 2),
                'avg_confidence': round(stats[3] or 0, 2),
                'satisfaction_rate': round(stats[4] or 0, 2),
                'escalation_rate': round(stats[5] or 0, 2)
            },
            'intent_distribution': [
                {'intent': row[0], 'count': row[1], 'avg_confidence': round(row[2], 2)}
                for row in intent_distribution
            ],
            'daily_counts': [
                {'date': row[0], 'count': row[1]}
                for row in daily_counts
            ],
            'sentiment_distribution': [
                {'sentiment': row[0], 'count': row[1]}
                for row in sentiment_distribution
            ]
        }
    
    def get_all_conversations(self, limit=100):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, session_id, user_message, bot_response, intent, sentiment, 
               confidence, timestamp, feedback, feedback_text
        FROM conversations
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        conversations = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'session_id': row[1],
                'user_message': row[2],
                'bot_response': row[3],
                'intent': row[4],
                'sentiment': row[5],
                'confidence': row[6],
                'timestamp': row[7],
                'feedback': row[8],
                'feedback_text': row[9]
            }
            for row in conversations
        ]
    
    def get_conversations_by_intent(self, intent):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, session_id, user_message, bot_response, intent, sentiment, 
               confidence, timestamp, feedback
        FROM conversations
        WHERE intent = ?
        ORDER BY timestamp DESC
        ''', (intent,))
        
        conversations = cursor.fetchall()
        conn.close()
        
        return conversations
    
    def export_conversations(self, session_id=None, format_type='json'):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
            SELECT * FROM conversations WHERE session_id = ?
            ORDER BY timestamp
            ''', (session_id,))
        else:
            cursor.execute('''
            SELECT * FROM conversations
            ORDER BY timestamp DESC
            LIMIT 1000
            ''')
        
        conversations = cursor.fetchall()
        conn.close()
        
        if format_type == 'json':
            return [dict(zip([col[0] for col in cursor.description], row)) 
                   for row in conversations] if conversations else []
        
        return conversations
    
    def get_user_session_summary(self, session_id):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            COUNT(*) as message_count,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            AVG(CASE WHEN feedback > 0 THEN feedback END) as avg_satisfaction,
            GROUP_CONCAT(DISTINCT intent) as intents_used
        FROM conversations
        WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'message_count': result[0],
                'start_time': result[1],
                'end_time': result[2],
                'avg_satisfaction': result[3],
                'intents_used': result[4].split(',') if result[4] else []
            }
        
        return None