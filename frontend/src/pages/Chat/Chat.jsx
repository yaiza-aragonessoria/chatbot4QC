import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import api from "../../api/chatbot4QC"
import "./Chat.css"
const Chat = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [warning, setWarning] = useState("");
  // const token = localStorage.getItem("access");


  // const config = {
  //   headers: { Authorization: `Bearer ${token}` },
  // };

  const fetchMessages = async () => {
      let backendData = await api.get(
        "/messages/"
      );
      setMessages(backendData.data);
    };

  useEffect(() => {
    fetchMessages();
  }, []);

  const handleSendMessage = async (e) => {
    e.preventDefault();

    setWarning("");
    api.post(
        "/messages/",
        {
          content: input,
          role: "user",
          previous_message: messages[messages.length - 1].id
        }
      )
      .then((result) => {
        setInput("");
        setMessages([...messages, result.data]);
        fetchMessages();
      })
      .catch((error) => {
        // set warning
        setWarning(error.message);
      });
  };

  const handleRefresh = async (e) => {
    e.preventDefault();
    console.log("Handling refresh...");

    const deleteMessage = async (idMessage) => {
      await api.delete(`/messages/${idMessage}/`);
    };

    // Delete all messages except the first one
    await Promise.all(messages.slice(1).map(message => deleteMessage(message.id)));

    // Fetch the remaining messages
    fetchMessages();
  };

  return (
  <div className="chatbot-container">
    <div className="chatbot-messages">
      {messages.map((message, id) => (
        <div
          key={id}
          className={`message ${message.role === 'user' ? 'user-message' : 'ai-message'}`}
        >
          {message.content}
          {message.draw ? (
            <p><img src={message.draw} alt="Draw" />
            </p>) : null
          }
        </div>
      ))}
    </div>
    <div className="chatbot-menu">
      <div className="refresh-button">
        <button className='refresh-button' onClick={handleRefresh}><img src="/refresh.png" alt="Refresh Icon" /></button>
      </div>
      <form className="chatbot-input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  </div>
);
};

export default Chat;
