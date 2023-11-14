import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

const messages = [{id: 1, role: 'chatbot', content: 'Hi, how can I help you?'}, {id: 1, role: 'chatbot', content: 'You can try with "what is a cnot gate?"'} ]

const Chat = () => {
  // const navigate = useNavigate();
  // const dispatch = useDispatch();


  return (
    <>
      {(
        <div>
            {messages?.map((message, id) => {
                return (
                    <div>
                      {message.role}: {message.content}
                    </div>
                );})}
        </div>
      )}
    </>
  );
};

export default Chat;
