import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Heart, Shield, Clock, Brain } from 'lucide-react';
import '../styles/HomePage.css';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="homepage">
      <header className="homepage-header">
        <div className="logo-container">
          <Heart className="logo-icon" />
          <h1 className="logo-text">Carebot</h1>
        </div>
        <button 
          className="header-login-btn"
          onClick={() => navigate('/login')}
        >
          Login
        </button>
      </header>

      <section className="hero-section">
        <div className="hero-content">
          <h2 className="hero-title">Your AI Healthcare Assistant, 24/7</h2>
          <p className="hero-subtitle">
            Get instant medical guidance from our intelligent chatbot
          </p>
          <button 
            className="cta-button"
            onClick={() => navigate('/login')}
          >
            Start Conversation
          </button>
        </div>
        
        <div className="hero-image">
          <img 
  src="/assets/hero-healthcare1.jpg" 
  alt="Healthcare illustration"
/>

        </div>
      </section>

      <section className="features-section">
        <div className="feature-card">
          <Shield className="feature-icon" />
          <h3>Emergency Support</h3>
          <p>Quick response for urgent medical queries</p>
        </div>
        
        <div className="feature-card">
          <Clock className="feature-icon" />
          <h3>24/7 Availability</h3>
          <p>Healthcare guidance anytime you need</p>
        </div>
        
        <div className="feature-card">
          <Brain className="feature-icon" />
          <h3>AI-Powered</h3>
          <p>Smart diagnosis and health recommendations</p>
        </div>
      </section>
    </div>
  );
}

export default HomePage;
