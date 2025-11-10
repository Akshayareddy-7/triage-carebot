import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/chat.css";
import doctorImg from "../assets/doctor.jpg";

export default function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [toast, setToast] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const inputRef = useRef(null);
  const token = localStorage.getItem("token");

  if (!token) navigate("/");

  // Utility to append messages to chat
  const appendMessage = (msg, role = "bot") => {
    setMessages((prev) => [...prev, { msg, role }]);
    setTimeout(() => {
      const chat = document.getElementById("chat");
      if (chat) chat.scrollTop = chat.scrollHeight;
    }, 50);
  };

  // Handle user sending a message
  const sendMessage = async () => {
    if (!input.trim()) return;
    appendMessage(input, "user");
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: "sess-" + Math.random().toString(36).slice(2, 9),
          message: input,
        }),
      });

      const data = await res.json();

      // Simulate typing animation delay
      setTimeout(() => {
        setIsTyping(false);

        // Doctor's reply
        appendMessage(data.reply || "No response received.", "bot");

        // Add triage badge (Routine/Urgent/Emergency)
        if (data.triage) {
          let badgeClass =
            data.triage.level.toLowerCase() === "emergency"
              ? "emergency"
              : data.triage.level.toLowerCase() === "urgent"
              ? "urgent"
              : "routine";

          appendMessage(
            `<div class="triage-badge ${badgeClass}">
              <strong>${data.triage.level}</strong> — ${data.triage.reason}
            </div>`,
            "bot"
          );
        }
      }, 1500);
    } catch (err) {
      setIsTyping(false);
      appendMessage("❌ Failed to send message.", "bot");
    }
  };

  // Run simulation
  const runSimulator = async () => {
    appendMessage("<em>Starting patient simulation...</em>", "meta");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/simulate_patient_chat", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      data.message_history.forEach((m) =>
        appendMessage(m.message, m.role === "patient" ? "user" : "bot")
      );
      appendMessage(`<strong>Doctor:</strong> ${data.doctor_reply}`, "bot");

      // Also append triage for simulated conversation
      if (data.triage) {
        let badgeClass =
          data.triage.level.toLowerCase() === "emergency"
            ? "emergency"
            : data.triage.level.toLowerCase() === "urgent"
            ? "urgent"
            : "routine";
        appendMessage(
          `<div class="triage-badge ${badgeClass}">
            <strong>${data.triage.level}</strong> — ${data.triage.reason}
          </div>`,
          "bot"
        );
      }
    } catch {
      appendMessage("❌ Simulation failed. Try again.", "bot");
    }
  };

  // Generate summary
  const generateSummary = async () => {
  appendMessage("<div class='meta'>🧠 Generating case summary...</div>", "meta");
  try {
    const res = await fetch("http://127.0.0.1:8000/api/generate_summary", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await res.json();

    appendMessage(
      `<div class='summary-box'>
         <h3>🧾 <strong>Case Summary:</strong></h3>
         <div>${data.summary}</div>
       </div>`,
      "bot"
    );
    showToast("✅ Summary generated successfully!");
  } catch {
    appendMessage("❌ Failed to generate summary.", "bot");
  }
};

  // Generate diagnosis
  const generateDiagnosis = async () => {
  appendMessage("<div class='meta'>Analyzing conversation for possible diagnoses...</div>", "meta");
  try {
    const res = await fetch("http://127.0.0.1:8000/api/generate_diagnosis", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await res.json();

    // Format top 5 diagnoses to bold the disease names
    const formattedDiagnosis = (data.diagnosis || "")
      .replace(/\d+\.\s*([A-Za-z0-9\s\(\)\-]+)/g, (match, p1) => `<strong>${match}</strong>`);

    appendMessage(
      `<div class='diagnosis-box'>
         <h3>🩻 <strong>Diagnoses:</strong></h3>
         <div>${formattedDiagnosis}</div>
       </div>`,
      "bot"
    );
    showToast("✅ Diagnosis generated successfully!");
  } catch {
    appendMessage("❌ Failed to generate diagnosis.", "bot");
  }
};


  // Toast notification
  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 4000);
  };

  return (
    <div className="chat-page">
      {/* ===== Navbar ===== */}
      <nav className="navbar">
        <div className="nav-left">
          <h1>🩺 Care Companion</h1>
          <button onClick={() => navigate("/home")}>🏠 Home</button>
          <button onClick={runSimulator}>Run Simulator</button>
          <button onClick={generateSummary}>Generate Summary</button>
          <button onClick={generateDiagnosis}>Generate Diagnosis</button>
        </div>

        <div className="profile-section">
          <img src={doctorImg} alt="Profile" className="profile-avatar" />
          <span className="profile-name">Dr. Cura</span>
          <button
            className="logout-btn"
            onClick={() => {
              localStorage.removeItem("token");
              navigate("/");
            }}
          >
            Logout
          </button>
        </div>
      </nav>

      {/* ===== Chat Container ===== */}
      <div className="chat-container">
        <div className="avatar-section">
          <img src={doctorImg} alt="Doctor" />
          <h2>Dr. Cura</h2>
          <p>Your trusted AI healthcare assistant.</p>
        </div>

        <div className="chat-section">
          <div id="chat">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`message ${m.role}`}
                dangerouslySetInnerHTML={{ __html: m.msg }}
              ></div>
            ))}

            {/* Doctor Typing Animation */}
            {isTyping && (
              <div className="message bot typing">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div id="input-area">
            <input
              id="input"
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Describe your symptoms..."
            />
            <button id="send" onClick={sendMessage}>
              ➤
            </button>
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && <div className="toast show">{toast}</div>}
    </div>
  );
}
