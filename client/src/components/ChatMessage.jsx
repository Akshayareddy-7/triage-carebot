import React from 'react';
import { AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';

function ChatMessage({ message }) {
  const getTriageIcon = (level) => {
    switch(level) {
      case 'Emergency':
        return <AlertCircle size={18} />;
      case 'Urgent':
        return <AlertTriangle size={18} />;
      case 'Routine':
      case 'Normal':
        return <CheckCircle size={18} />;
      default:
        return null;
    }
  };

  return (
    <div className={`message ${message.sender}`}>
      {message.sender === 'bot' && (
        <img 
          src="https://ui-avatars.com/api/?name=Dr+AI&size=35&background=0066FF&color=fff&rounded=true"
          alt="Doctor"
          className="message-avatar"
        />
      )}
      <div className="message-bubble">
        <p>{message.text}</p>
        
        {message.sender === 'bot' && message.triage && (
          <div 
            className="message-triage" 
            style={{ 
              backgroundColor: message.triage.color + '20',
              borderLeft: `3px solid ${message.triage.color}`
            }}
          >
            <div className="triage-header" style={{ color: message.triage.color }}>
              {getTriageIcon(message.triage.level)}
              <strong>{message.triage.level}</strong>
            </div>
            <p className="triage-reason">{message.triage.reason}</p>
          </div>
        )}
        
        <span className="message-time">{message.timestamp}</span>
      </div>
    </div>
  );
}

export default ChatMessage;
