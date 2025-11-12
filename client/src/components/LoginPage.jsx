import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { Mail, Lock, Heart } from 'lucide-react';
import '../styles/Auth.css';

function LoginPage({ setIsAuthenticated }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const response = await axios.post('/api/login', {
        email,
        password
      });

      localStorage.setItem('token', response.data.token);
      localStorage.setItem('userName', response.data.name);
      setIsAuthenticated(true);
      navigate('/chat');
    } catch (err) {
      setError(err.response?.data?.message || 'Login failed. Please try again.');
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-left">
        <img 
          src="/assets/hero-healthcare.jpg"
          alt="Medical consultation"
          className="auth-illustration"
        />
      </div>

      <div className="auth-right">
        <div className="auth-box">
          <div className="auth-logo">
            <Heart className="auth-logo-icon" />
            <h2>MediAI</h2>
          </div>

          <h3 className="auth-title">Welcome Back</h3>
          <p className="auth-subtitle">Login to continue your healthcare journey</p>

          {error && <div className="error-message">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="input-group">
              <Mail className="input-icon" />
              <input
                type="email"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="input-group">
              <Lock className="input-icon" />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <button type="submit" className="auth-submit-btn">
              Login
            </button>
          </form>

          <p className="auth-switch">
            Don't have an account? <Link to="/register">Register here</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
