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
  // const token = localStorage.getItem("access");
  const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzAwNDE3MDk3LCJpYXQiOjE2OTk5ODUwOTcsImp0aSI6IjdiNDg5MDkzY2RmYjQwNGY5MDFiMTlmZWM4MmZlZmM5IiwidXNlcl9pZCI6MX0.m3YO28E3fqZQ9OzK2og35CfL5oNB19Oum12nOcYqMQI";

  const config = {
    headers: { Authorization: `Bearer ${token}` },
  };

  useEffect(() => {
    const fetchMessages = async () => {
      // console.log("fetchdata");
      let backendData = await api.get(
        "/messages/",
        {
    headers: { Authorization: `Bearer ${token}` },}
      );
      console.log(backendData);

      setMessages(backendData.data);
    };

    fetchMessages();
  }, []);

  const handleSubmit = async (e) => {
    console.log('handleSubmit')
    e.preventDefault();
    // if (!input.trim()) return;
    // const userMessage = { text: input, user: true };
    // setMessages((prevMessages) => [...prevMessages, userMessage]);
    // const aiMessage = { text: '...', user: false };
    // setMessages((prevMessages) => [...prevMessages, aiMessage]);
    // const response = await chatWithGPT3(input);
    // const newAiMessage = { text: response, user: false };
    // setMessages((prevMessages) => [...prevMessages.slice(0, -1), newAiMessage]);
    // setInput('');
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
          </div>
        ))}
      </div>
      <form className="chatbot-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default Chat;
