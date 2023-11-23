# content of test_file_manager.py

import datetime
import os
import sys
import torch
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import interface
import chatbot.logic_engine as le


def init():
    # Create an instance of the Chatbot class, passing the data folder path
    data_folder_path = './data/gate_questions/'
    chatbot = interface.Chatbot(data_folder_path, le, save=False)
    # Initialize the chatbot.xml with the specified checkpoint file
    # checkpoint_folder_path = './model/'
    # file_manager = interface.FileManager(checkpoint_folder_path)
    # file_manager.get_latest_file()
    # checkpoint_path = checkpoint_folder_path + file_manager.file_name
    # chatbot.initialize(checkpoint_path, map_location='cpu', retraining_bound=20)

    return chatbot


data_folder_path = './data/gate_questions/'
checkpoint_folder_path = './model/'
map_location = "cpu"

chatbot = init()
file_manager = interface.FileManager(checkpoint_folder_path)
file_manager.get_latest_file()
retraining_bound = 20


@pytest.fixture
def chatbot_test_data_fixture():
    chatbot_instance = interface.Chatbot(data_folder_path, le, save=False)
    checkpoint_path = checkpoint_folder_path + file_manager.file_name
    chatbot_instance.initialize(checkpoint_path, map_location='cpu', retraining_bound=20)
    chatbot_instance.file_manager = interface.FileManager("./tests/tests_interface/data_for_tests/")
    return chatbot_instance


chatbot_test_data = interface.Chatbot(data_folder_path="./tests/tests_interface/data_for_tests/", le=le, save=False)
# checkpoint_path = checkpoint_folder_path + file_manager.file_name
# chatbot_instance.initialize(checkpoint_path, map_location='cpu', retraining_bound=20)
chatbot_test_data.file_manager = interface.FileManager("./tests/tests_interface/data_for_tests/")
chatbot_test_data.file_manager.get_latest_file()

# Define a fixture to create a temporary folder for testing
@pytest.fixture
def temp_folder(tmpdir):
    return str(tmpdir.mkdir("test_folder"))

@pytest.fixture
def remove_last_line(request, chatbot_test_data_fixture):

    # This function will be called after the test has finished
    def finalizer():
        # Code to remove the last line from the file
        chatbot_test_data_fixture.file_manager.get_latest_file()
        file_path = chatbot_test_data_fixture.file_manager.folder_path + chatbot_test_data_fixture.file_manager.file_name
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Remove the last line
        if lines:
            with open(file_path, 'w') as file:
                file.writelines(lines[:-1])
                print(f"The last line has been removed from {file_path}")

    # Register the finalizer
    request.addfinalizer(finalizer)

def test_initialize_initializes_bert_model_and_file_manager():
    # Mocking the load_checkpoint and get_latest_file methods
    with patch.object(chatbot.bert_model, "load_checkpoint") as mock_load_checkpoint, \
            patch.object(chatbot.file_manager, "get_latest_file") as mock_get_latest_file:
        chatbot.initialize(checkpoint_folder_path, map_location, retraining_bound)

        # Assertions
        mock_load_checkpoint.assert_called_once_with(train_from_scratch=False, path_to_model=checkpoint_folder_path)
        mock_get_latest_file.assert_called_once()


def test_initialize_does_not_retrain_bert_model_when_file_not_present():
    # Mocking the load_checkpoint and get_latest_file methods
    with patch.object(chatbot.bert_model, "load_checkpoint") as mock_load_checkpoint, \
            patch.object(chatbot.file_manager, "get_latest_file", return_value=None) as mock_get_latest_file, \
            patch.object(chatbot.bert_model, 'train') as mock_train:
        chatbot.initialize(checkpoint_folder_path, map_location, retraining_bound)

        # Assertions
        mock_load_checkpoint.assert_called_once_with(train_from_scratch=False, path_to_model=checkpoint_folder_path)
        mock_get_latest_file.assert_called_once()
        mock_train.assert_not_called()


def test_initialize_retraining_bound_exceeded(temp_folder, capsys):
    with open("./tests/tests_interface/data_for_tests/21questions.txt", 'r+') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Extract the old last line
        last_line = lines[-1].strip()

        # Truncate the file to remove the last line
        file.truncate(0)

        # Write back all lines except the last one
        file.seek(0)
        file.writelines(lines[:-1])

        # Write the new last line
        file.write(last_line + '\n')

    directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/model/"
    file_manager_model = interface.FileManager(directory)
    file_manager_model.get_latest_file()
    checkpoint_path = checkpoint_folder_path + file_manager.file_name

    chatbot_test_data.initialize(checkpoint_path, map_location='cpu', retraining_bound=20)
    captured = capsys.readouterr()

    assert "*** Chatbot initialized ***\nImproving chatbot...\n" in captured.out


def test_initialize_retraining_bound_not_exceeded(temp_folder, capsys):
    with open("./tests/tests_interface/data_for_tests/11questions.txt", 'r+') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Extract the old last line
        last_line = lines[-1].strip()

        # Truncate the file to remove the last line
        file.truncate(0)

        # Write back all lines except the last one
        file.seek(0)
        file.writelines(lines[:-1])

        # Write the new last line
        file.write(last_line + '\n')

    directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/model/"
    file_manager_model = interface.FileManager(directory)
    file_manager_model.get_latest_file()

    chatbot_test_data.initialize(directory + file_manager_model.file_name, map_location, retraining_bound=20)

    captured = capsys.readouterr()
    assert "*** Chatbot initialized ***\n" in captured.out
    assert "Improving chatbot..." not in captured.out


def read_test_cases(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        test_cases = [line.strip().split('\t') for line in lines]
    print(test_cases)
    return test_cases


# @pytest.mark.parametrize("expected_category", "input_question",
#                          read_test_cases('./tests/tests_interface/data_for_tests/11questions.txt'))

@pytest.mark.parametrize("input_question, expected_category", [
    ("apply pauli x", 2),
    ("draw circuit result", 1),
    ("define pauli x gate", 0),
])
def test_classify_user_input(chatbot_test_data_fixture, input_question, expected_category):
    # checkpoint_path = checkpoint_folder_path + file_manager.file_name
    # chatbot.initialize(checkpoint_path, map_location='cpu', retraining_bound=20)
    assert chatbot_test_data_fixture.classify_user_input(input_question)[0] == int(expected_category)


def test_process_user_question_both_found(chatbot_test_data_fixture):
    user_question = "What is the effect of applying the Hadamard gate on state |0>?"
    category = 2  # Assume the category for computation

    gate_name, initial_state_name, understood_question = chatbot_test_data_fixture.process_user_question(user_question, category)

    assert gate_name == "Hadamard"
    assert initial_state_name == "|0>"
    assert f"I understand that you want me to compute the resulting state after applying the gate {gate_name} on {initial_state_name}. Is this correct? (yes/no): " in understood_question


def test_process_user_question_only_gate_found(chatbot_test_data_fixture):
    user_question = "Compute the state after applying the pauli X gate"
    category = 2

    gate_name, initial_state_name, understood_question = chatbot_test_data_fixture.process_user_question(user_question, category)

    assert gate_name == "Pauli x"
    assert initial_state_name is None
    assert f"I understand that you want me to compute the resulting state after applying the gate {gate_name}. Is " \
           f"this correct?" in understood_question


def test_ask_user_for_confirmation_yes(capfd, chatbot_test_data_fixture):
    understood_question = "Do you want to proceed? (yes/no): "

    with patch("builtins.input", side_effect=["yes"]):
        user_response = chatbot_test_data_fixture.ask_user_for_confirmation(understood_question)

    captured = capfd.readouterr()
    assert user_response == "yes"
    assert "Ok, let's do it!" in captured.out


def test_ask_user_for_confirmation_no(capfd, chatbot_test_data_fixture):
    understood_question = "Do you want to proceed? (yes/no): "

    with patch("builtins.input", side_effect=["no"]):
        user_response = chatbot_test_data_fixture.ask_user_for_confirmation(understood_question)

    captured = capfd.readouterr()
    assert user_response == "no"
    assert "Ups... Let me try again." in captured.out


def test_ask_user_for_confirmation_invalid_input_then_yes(capfd, chatbot_test_data_fixture):
    understood_question = "Do you want to proceed? (yes/no): "

    with patch("builtins.input", side_effect=["invalid", "yes"]):
        user_response = chatbot_test_data_fixture.ask_user_for_confirmation(understood_question)

    captured = capfd.readouterr()
    assert user_response == "yes"
    assert "Please enter 'yes' or 'no'" in captured.out
    assert "Ok, let's do it!" in captured.out


def test_handle_user_response_yes(capfd, chatbot_test_data_fixture, remove_last_line):
    user_response = "yes"
    category = 0
    user_question = "What is a the Hadamard gate?"
    i = 3  # Some initial value

    chatbot_test_data_fixture.file_manager.get_latest_file()

    # Check if the question is appended to the data file
    with open(chatbot_test_data_fixture.file_manager.folder_path + chatbot_test_data_fixture.file_manager.file_name, 'r') as file:
        num_lines = sum(1 for line in file)

    exit_loop, i = chatbot_test_data_fixture.handle_user_response(user_response, category, user_question, i)

    assert exit_loop is True
    assert i == 3  # No change in 'i'

    # Check if the question is appended to the data file
    with open(chatbot_test_data_fixture.file_manager.folder_path + chatbot_test_data_fixture.file_manager.file_name, 'r') as file:
        lines = file.readlines()
        assert len(lines) == num_lines + 1
        assert lines[-1] == f'{category}\t{user_question}\n'

    # # Remove the line
    # with open(chatbot_test_data_fixture.file_manager.folder_path + chatbot_test_data_fixture.file_manager.file_name, 'w') as file:
    #     lines = file.readlines()
    #     lines = lines[:-1]
    #     file.writelines(lines)


def test_handle_user_response_no(capfd, chatbot_test_data_fixture):
    user_response = "no"
    category = 0
    user_question = "Draw a circuit."
    i = 2  # Some initial value

    chatbot_test_data_fixture.file_manager.get_latest_file()

    # Check if the question is appended to the data file
    with open(chatbot_test_data_fixture.file_manager.folder_path + chatbot_test_data_fixture.file_manager.file_name, 'r') as file:
        num_lines = sum(1 for line in file)

    exit_loop, i = chatbot_test_data_fixture.handle_user_response(user_response, category, user_question, i)

    assert exit_loop is False
    assert i == 3  # 'i' should be incremented by 1

    # Check that nothing is appended to the data file
    with open(chatbot_test_data_fixture.file_manager.folder_path + chatbot_test_data_fixture.file_manager.file_name, 'r') as file:
        lines = file.readlines()
        assert len(lines) == num_lines

def test_handle_user_response_invalid_input(capfd, chatbot_test_data_fixture):
    user_response = "invalid"
    category = 2
    user_question = "Compute the result."
    i = 1  # Some initial value

    exit_loop, i = chatbot_test_data_fixture.handle_user_response(user_response, category, user_question, i)

    assert exit_loop is False
    assert i == 1  # 'i' should remain the same

    captured = capfd.readouterr()
    assert "Please enter 'yes' or 'no'" in captured.out


def test_handle_unclassified_question_invalid_logits(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    chatbot_test_data_fixture.logits = MagicMock()
    chatbot_test_data_fixture.top_indices = MagicMock()
    i = 0  # Some initial value

    # Set the logits value to be greater than 0
    chatbot_test_data_fixture.logits[0][chatbot_test_data_fixture.top_indices[0][i]].item.return_value = 1.0

    chatbot_test_data_fixture.handle_unclassified_question(chatbot_test_data_fixture.logits, chatbot_test_data_fixture.top_indices, i)

    captured = capfd.readouterr()
    assert "Unfortunately I cannot understand your question." not in captured.out

def test_handle_unclassified_question_valid_logits(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    chatbot_test_data_fixture.logits = MagicMock()
    chatbot_test_data_fixture.top_indices = MagicMock()
    i = 0  # Some initial value

    # Set the logits value to be less than or equal to 0
    chatbot_test_data_fixture.logits[0][chatbot_test_data_fixture.top_indices[0][i]].item.return_value = 0.0

    chatbot_test_data_fixture.handle_unclassified_question(chatbot_test_data_fixture.logits, chatbot_test_data_fixture.top_indices, i)

    captured = capfd.readouterr()
    assert "Unfortunately I cannot understand your question." in captured.out


def test_handle_phase_or_rotation_question_phase_shift_valid_phase_shift(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the phase shift?': {
            'answer': {'answer': ['0.5']}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('phase', 'user_question')

    assert phase_shift == 0.5
    assert angle is None
    assert axis is None
    assert gate_name == 'phase'
    assert isinstance(gate, le.PhaseGate)


def test_handle_phase_or_rotation_question_phase_shift_no_phase_shift(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the phase shift?': {
            'answer': {'answer': []}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('phase', 'user_question')

    assert phase_shift == 0
    assert angle is None
    assert axis is None
    assert gate_name == 'phase'
    assert isinstance(gate, le.PhaseGate)

def test_handle_phase_or_rotation_question_rotation_valid_angle_axis(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    # chatbot_instance.gates = {'rotation': {'RX': MagicMock(), 'RY': MagicMock(), 'RZ': MagicMock()}}
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the angle of the rotation?': {
            'answer': {'answer': ['1.0']}
        },
        'What is the axis of the rotation?': {
            'answer': {'answer': ['y']}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('rotation', 'user_question')

    assert phase_shift is None
    assert angle == 1.0
    assert axis == 'y'
    assert gate_name == 'RY'
    assert isinstance(gate, le.RY)

def test_handle_phase_or_rotation_question_no_angle_no_axis(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the angle of the rotation?': {
            'answer': {'answer': []}
        },
        'What is the axis of the rotation?': {
            'answer': {'answer': []}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('rotation', 'user_question')

    assert phase_shift is None
    assert angle == 0
    assert axis == 'x'
    assert gate_name == 'RX'  # Default to 'RX'
    assert isinstance(gate, le.RX)


def test_handle_phase_or_rotation_question_no_angle_valid_axis(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the angle of the rotation?': {
            'answer': {'answer': []}
        },
        'What is the axis of the rotation?': {
            'answer': {'answer': ['y']}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('rotation', 'user_question')

    assert phase_shift is None
    assert angle == 0
    assert axis == 'y'
    assert gate_name == 'RY'
    assert isinstance(gate, le.RY)


def test_handle_phase_or_rotation_question_valid_angle_no_axis(capfd, chatbot_test_data_fixture):
    # Mocking the necessary attributes
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the angle of the rotation?': {
            'answer': {'answer': ['0.5']}
        },
        'What is the axis of the rotation?': {
            'answer': {'answer': []}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('rotation', 'user_question')

    assert phase_shift is None
    assert angle == 0.5
    assert axis == 'x'
    assert gate_name == 'RX'  # Default to 'RX'
    assert isinstance(gate, le.RX)

@pytest.mark.parametrize("axis, expected_gate_name", [("x", "RX"), ("y", "RY"), ("z", "RZ")])
def test_handle_phase_or_rotation_question_all_rotations(capfd, chatbot_test_data_fixture, axis, expected_gate_name):
    # Mocking the necessary attributes
    bert_qa_mock = MagicMock(spec=interface.BertQA)
    bert_qa_mock.ask_questions.return_value = {
        'What is the angle of the rotation?': {
            'answer': {'answer': []}
        },
        'What is the axis of the rotation?': {
            'answer': {'answer': [axis]}
        }
    }

    with patch('interface.BertQA', return_value=bert_qa_mock):
        phase_shift, angle, actual_axis, gate_name, gate = chatbot_test_data_fixture.handle_phase_or_rotation_question('rotation', 'user_question')

    assert phase_shift is None
    assert angle == 0
    assert actual_axis == axis
    assert gate_name == expected_gate_name
    assert isinstance(gate, getattr(le, expected_gate_name))


@pytest.fixture
def mock_gate():
    return MagicMock()

@pytest.fixture
def mock_parameters():
    return MagicMock()


def test_apply_gate_method_calls_answer_handler_apply_gate_method_with_correct_arguments(chatbot_test_data_fixture, mock_gate, mock_parameters):
    category = 0  # Replace with your category value
    chatbot_test_data_fixture.apply_gate_method(category, mock_gate, mock_parameters)

    # Mocking AnswerHandler's apply_gate_method to return the expected result
    with patch.object(interface.AnswerHandler, 'apply_gate_method') as mock_apply_gate_method:
        chatbot_test_data_fixture.apply_gate_method(category, mock_gate, mock_parameters)

        # Assertions
        mock_apply_gate_method.assert_called_once_with()

#
def test_apply_gate_method_returns_correct_result(chatbot_test_data_fixture, mock_gate):
    category = 0  # Replace with your category value
    expected_result = MagicMock()  # Replace with the expected result
    parameters = ["|1>"]

    # Mocking AnswerHandler's apply_gate_method to return the expected result
    with patch.object(interface.AnswerHandler, 'apply_gate_method', return_value=expected_result) as mock_apply_gate_method:
        result = chatbot_test_data_fixture.apply_gate_method(category, mock_gate, parameters)

    # Assertions
    assert result == expected_result


def test_start_exits_chatbot_when_user_enters_exit(chatbot_test_data_fixture):
    with patch("builtins.input", side_effect=["exit"]), \
            patch("builtins.print") as mock_print:
        chatbot_test_data_fixture.start()

        # Assertions
        mock_print.assert_called_with("Exiting the chatbot.")


def test_start_calls_expected_methods_with_user_input(chatbot_test_data_fixture):
    user_question = "define the gate pauli x"

    with patch("builtins.input", side_effect=[user_question, "exit"]), \
         patch.object(chatbot_test_data_fixture, "classify_user_input", return_value=(0, torch.tensor([[4.9444, -2.5437, -2.7457]]), torch.tensor([[0, 1, 2]]))) as mock_classify, \
         patch.object(chatbot_test_data_fixture, "process_user_question", return_value=("Pauli x", "", "I understand that you want to know the definition of the gate Pauli X. Is this correct?")) as mock_process, \
         patch.object(chatbot_test_data_fixture, "ask_user_for_confirmation", return_value="yes") as mock_ask_confirmation, \
         patch.object(chatbot_test_data_fixture, "handle_user_response", return_value=(True, 0)) as mock_handle_response, \
         patch.object(chatbot_test_data_fixture, "handle_unclassified_question"), \
         patch.object(chatbot_test_data_fixture, "apply_gate_method"):

        # with pytest.raises(SystemExit):  # To break out of the infinite loop
        chatbot_test_data_fixture.start()

        # Assertions
        mock_classify.assert_called_once_with(user_question)
        mock_process.assert_called_once_with(user_question, 0)
        mock_ask_confirmation.assert_called_once_with("I understand that you want to know the definition of the gate Pauli X. Is this correct?")
        mock_handle_response.assert_called_once_with("yes", 0, user_question, 0)
