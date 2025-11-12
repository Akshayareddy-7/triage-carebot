import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { Mail, Lock, User, Heart } from 'lucide-react';
import '../styles/Auth.css';

function RegisterPage({ setIsAuthenticated }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const response = await axios.post('/api/register', {
        name,
        email,
        password
      });

      localStorage.setItem('token', response.data.token);
      localStorage.setItem('userName', name);
      setIsAuthenticated(true);
      navigate('/chat');
    } catch (err) {
      setError(err.response?.data?.message || 'Registration failed. Please try again.');
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-left">
        <img 
          src="https://cdni.iconscout.com/illustration/premium/thumb/doctor-consultation-4934196-4108759.png"
          alt="Doctor consultation"
          className="auth-illustration"
        />
      </div>

      <div className="auth-right">
        <div className="auth-box">
          <div className="auth-logo">
            <Heart className="auth-logo-icon" />
            <h2>MediAI</h2>
          </div>

          <h3 className="auth-title">Create Account</h3>
          <p className="auth-subtitle">Start your AI healthcare journey today</p>

          {error && <div className="error-message">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="input-group">
              <User className="input-icon" />
              <input
                type="text"
                placeholder="Full name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>

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
                placeholder="Password (min 6 characters)"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                minLength="6"
                required
              />
            </div>

            <button type="submit" className="auth-submit-btn">
              Create Account
            </button>
          </form>

          <p className="auth-switch">
            Already have an account? <Link to="/login">Login here</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default RegisterPage;
