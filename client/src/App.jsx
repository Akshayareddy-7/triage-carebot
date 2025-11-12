import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './components/HomePage';
import LoginPage from './components/LoginPage';
import RegisterPage from './components/RegisterPage';
import ChatPage from './components/ChatPage';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(
    localStorage.getItem('token') !== null
  );

  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route 
          path="/login" 
          element={<LoginPage setIsAuthenticated={setIsAuthenticated} />} 
        />
        <Route 
          path="/register" 
          element={<RegisterPage setIsAuthenticated={setIsAuthenticated} />} 
        />
        <Route 
          path="/chat" 
          element={
            isAuthenticated ? 
            <ChatPage setIsAuthenticated={setIsAuthenticated} /> : 
            <Navigate to="/login" />
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;
