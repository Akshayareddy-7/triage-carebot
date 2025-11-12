import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Home, FileText, User, Send, LogOut, Heart } from 'lucide-react';
import ChatMessage from './ChatMessage';
import '../styles/ChatPage.css';

function ChatPage({ setIsAuthenticated }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  const userName = localStorage.getItem('userName') || 'User';
  const [sessionId] = useState(`session-${Date.now()}`);

  useEffect(() => {
    setMessages([
      {
        text: `Hello ${userName}! I'm Dr. AI Assistant. How can I help you with your health concerns today?`,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
  }, [userName]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    const messageText = input;
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        session_id: sessionId,
        message: messageText
      });

      const botMessage = {
        text: response.data.response,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        triage: response.data.triage  // ✅ ADDED THIS LINE
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userName');
    setIsAuthenticated(false);
    navigate('/');
  };

  const generateSummary = () => {
    setShowSummary(true);
  };

  return (
    <div className="chat-page">
      <nav className="chat-navbar">
        <div className="navbar-left">
          <Heart className="navbar-logo" />
          <button onClick={() => navigate('/')} className="nav-btn">
            <Home size={20} />
            <span>Home</span>
          </button>
        </div>

        <div className="navbar-center">
          <img 
            src="https://ui-avatars.com/api/?name=Dr+AI&size=40&background=0066FF&color=fff&rounded=true"
            alt="Doctor"
            className="doctor-avatar"
          />
          <div className="doctor-info">
            <h3>Dr.Cura</h3>
            <span className="online-status">● Online</span>
          </div>
        </div>

        <div className="navbar-right">
          <button onClick={generateSummary} className="nav-btn">
            <FileText size={20} />
            <span>Summary</span>
          </button>
          <div className="profile-dropdown">
            <button 
              onClick={() => setShowProfile(!showProfile)} 
              className="nav-btn profile-btn"
            >
              <User size={20} />
            </button>
            {showProfile && (
              <div className="dropdown-menu">
                <p className="user-name">{userName}</p>
                <button onClick={handleLogout} className="logout-btn">
                  <LogOut size={16} />
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </nav>

      <div className="chat-container">
        <div className="messages-area">
          {messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))}
          {loading && (
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <input
            type="text"
            placeholder="Type your health question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            className="chat-input"
          />
          <button 
            onClick={handleSendMessage} 
            className="send-btn"
            disabled={loading}
          >
            <Send size={20} />
          </button>
        </div>
      </div>

      {showSummary && (
        <div className="summary-modal" onClick={() => setShowSummary(false)}>
          <div className="summary-content" onClick={(e) => e.stopPropagation()}>
            <h3>Conversation Summary</h3>
            <div className="summary-text">
              {messages.filter(m => m.sender === 'user').length > 0 ? (
                messages.filter(m => m.sender === 'user').map((m, i) => (
                  <p key={i}>• {m.text}</p>
                ))
              ) : (
                <p>No messages to summarize yet.</p>
              )}
            </div>
            <button onClick={() => setShowSummary(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatPage;
