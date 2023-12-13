import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from 'uuid';
import api from "../../api/chatbot4QC"
import "./Chat.css"
const Chat = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [warning, setWarning] = useState("");
  const [userEmail, setUserEmail] = useState(`${uuidv4()}@email.com`);
  // const token = localStorage.getItem("access");


  // const config = {
  //   headers: { Authorization: `Bearer ${token}` },
  // };

  const createUser = async (userEmail)=>{
    try {
      const response = await api.post("/users/create/",
          {
            email: userEmail,
            password: 'iamapassword'
          })
      if (response.status !== 200) {
        console.error('Failed to create user:', response.statusText);
      } else {
        console.log(`User created with ${userEmail}`)
      }

      console.log(response.data.message); // Log the server response
    } catch (error) {
      console.error('Error during user creation:', error);
    }
  };

  const deleteUser = async (userEmail) => {
    console.log(`Deleting user with ${userEmail}`)
    try {
      const response = await api.post('/users/delete/',
              {email: userEmail,
                password: 'iamapassword'
              })

      console.log({response})

      if (response.status !== 200) {
        console.error('Failed to delete user:', response.statusText);
      } else {
        console.log(`User with ${userEmail} deleted`)
      }

      console.log(response.data.message); // Log the server response
    } catch (error) {
      console.error('Error during user deletion:', error);
    }
  };

  const createGreetingMessage = () => {
    setWarning("");
    api.post(
        "/messages/",
        {
          content: 'Hola, how can I help you?',
          role: "ai",
          user_email: userEmail
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

  }

  const fetchMessages = async () => {
      let backendData = await api.get("/messages/user/",
                                                  { params: {user_email: userEmail}}
                                                         );
      setMessages(backendData.data);
    };

  const deleteMessage = async (idMessage) => {
    console.log("Deleting...")
      await api.delete(`/messages/${idMessage}/`);
    };

  useEffect(() => {
    setMessages([])
    createUser(userEmail);
    createGreetingMessage();
    fetchMessages();

    // Specify a cleanup function for when the component is unmounted
    return () => {
      // User leaves the chatbot, delete the user
      deleteUser(userEmail);
    };
  }, []);

  useEffect(() => {
    window.addEventListener("beforeunload", alertUser);
    return () => {
      window.removeEventListener("beforeunload", alertUser);
    };
  });

  useEffect(() => {
    // Add an event listener for beforeunload
    const handleBeforeUnload = () => {
      // User is closing the tab, delete the user
      deleteUser(userEmail);
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    // Specify cleanup for the event listener
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [userEmail]); // Re-run the effect when userEmail changes
  const alertUser = (e) => {
    e.preventDefault();
    e.returnValue = "";
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();

    console.log({
          content: input,
          role: "user",
          previous_message: messages[messages.length - 1].id,
          user_email: userEmail
        })

    setWarning("");
    api.post(
        "/messages/",
        {
          content: input,
          role: "user",
          previous_message: messages[messages.length - 1].id,
          user_email: userEmail
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
