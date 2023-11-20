# content of test_file_manager.py

import datetime
import os
import sys

import pytest
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
import interface

# chatbot = interface.init()


# def test_chatbot_classify_user_input():
#     assert chatbot.classify_user_input("apply pauli x")[0] == 2
