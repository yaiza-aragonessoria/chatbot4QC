# content of test_file_manager.py

import json
import os
import sys
from unittest.mock import patch, MagicMock, call

import pytest
from simpletransformers.question_answering import QuestionAnsweringModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import LLM_QA


@pytest.fixture
def bert_qa_scratch_false():
    # You can customize the parameters for testing
    return LLM_QA.BertQA(epochs=1, train_files=["./tests/data_for_tests/test_phase_shifts_data_train.json",
                                                "./tests/data_for_tests/test_rotation_data_train.json"],
                         test_files=["./tests/data_for_tests/test_phase_shifts_data_test.json",
                                     "./tests/data_for_tests/test_rotation_data_test.json"],
                         train_from_scratch=False)


@pytest.fixture
def bert_qa_scratch_true():
    # You can customize the parameters for testing
    return LLM_QA.BertQA(epochs=1, train_files=["./tests/data_for_tests/test_phase_shifts_data_train.json",
                                                "./tests/data_for_tests/test_rotation_data_train.json"],
                         test_files=["./tests/data_for_tests/test_phase_shifts_data_test.json",
                                     "./tests/data_for_tests/test_rotation_data_test.json"],
                         train_from_scratch=True)


@pytest.fixture
def reset_bert_attributes(request, bert_qa_scratch_false):
    # This function will be called after the test has finished
    def finalizer():
        bert_qa_scratch_false.train_files = ["./tests/data_for_tests/test_phase_shifts_data_train.json",
                                             "./tests/data_for_tests/test_rotation_data_train.json"]
        bert_qa_scratch_false.test_files = ["./tests/data_for_tests/test_phase_shifts_data_test.json",
                                            "./tests/data_for_tests/test_rotation_data_test.json"]
        bert_qa_scratch_false.train_from_scratch = False

    # Register the finalizer
    request.addfinalizer(finalizer)


# def test_bertqa_init_defaults():
#     with patch.object(LLM_QA.BertQA, 'train_model') as mock_train_model:
#         bert_qa_default = LLM_QA.BertQA()
#
#         assert mock_train_model.called
#
#     assert bert_qa_default.train_files == ["./data/QA/phase_shifts_data_train.json",
#                                            "./data/QA/rotation_data_train.json"]
#     assert bert_qa_default.test_files == ["./data/QA/phase_shifts_data_test.json", "./data/QA/rotation_data_test.json"]
#     assert bert_qa_default.model_type == "bert"
#     assert bert_qa_default.model_name == "bert-base-uncased"
#     assert bert_qa_default.output_dir == "LLM_QA"
#     assert bert_qa_default.train_from_scratch == True
#
#
# def test_bertqa_init_train_false(bert_qa_scratch_false):
#     assert bert_qa_scratch_false.train_files == ["./tests/data_for_tests/test_phase_shifts_data_train.json",
#                                                  "./tests/data_for_tests/test_rotation_data_train.json"]
#     assert bert_qa_scratch_false.test_files == ["./tests/data_for_tests/test_phase_shifts_data_test.json",
#                                                 "./tests/data_for_tests/test_rotation_data_test.json"]
#     assert bert_qa_scratch_false.model_type == "bert"
#     assert bert_qa_scratch_false.model_name == "bert-base-uncased"
#     assert bert_qa_scratch_false.output_dir == "LLM_QA"
#     assert bert_qa_scratch_false.train_from_scratch == False
#
#
# def test_bertqa_load_data_success(bert_qa_scratch_false):
#     train_files = ["./tests/data_for_tests/test_phase_shifts_data_train.json",
#                    "./tests/data_for_tests/test_rotation_data_train.json"]
#     result = bert_qa_scratch_false.load_data(train_files)
#
#     context_count = sum(1 for item in result if "context" in item)
#
#     assert len(result) == context_count
#     assert result[0]["qas"][0]["question"] == "What is the phase shift?"
#     assert result[1]["qas"][0]["answers"][0]['text'] == "pi/2"
#
#
# def test_bertqa_load_data_file_not_found(bert_qa_scratch_false, capfd, reset_bert_attributes):
#     file_paths = ["nonexistent_file.json"]
#
#     bert_qa_scratch_false.test_files = file_paths
#
#     bert_qa_scratch_false.load_data(file_paths)
#
#     captured = capfd.readouterr()
#     assert "File not found: nonexistent_file.json" in captured.out
#
#
# def test_bertqa_load_data_json_decode_error(bert_qa_scratch_false, capfd):
#     with patch("builtins.open", create=True) as mock_open:
#         # Mock the return value of json.load to simulate a JSON decode error
#         mock_open.side_effect = [json.JSONDecodeError("Test JSON decode error", "", 0)]
#
#         file_paths = ["fake_file.json"]
#
#         bert_qa_scratch_false.load_data(file_paths)
#
#         captured = capfd.readouterr()
#         assert "Error decoding JSON in file: fake_file.json\n" in captured.out
#
#
# def test_bertqa_train_model(bert_qa_scratch_true):
#     with patch.object(LLM_QA.BertQA, 'load_data') as mock_load_data, \
#             patch.object(QuestionAnsweringModel, 'train_model') as mock_train_model:
#         # bert_qa = LLM_QA.BertQA()
#
#         # Mock the return values of load_data for both train and test files
#         mock_load_data.side_effect = [
#             [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}],
#             [{"question": "Q3", "answer": "A3"}, {"question": "Q4", "answer": "A4"}]
#         ]
#
#         bert_qa_scratch_true.train_model()
#
#         # Assert that load_data was called with the correct file paths
#         mock_load_data.assert_has_calls([
#             call(bert_qa_scratch_true.train_files),
#             call(bert_qa_scratch_true.test_files)
#         ])
#
#         # Assert that train_model was called with the correct data
#         mock_train_model.assert_called_once_with(
#             [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}],
#             eval_data=[{"question": "Q3", "answer": "A3"}, {"question": "Q4", "answer": "A4"}]
#         )
#
#
# def test_bertqa_load_saved_model(bert_qa_scratch_false):
#     expected_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     expected_dir_path = os.path.dirname(expected_file_path)
#     with patch('LLM_QA.os.path.dirname') as mock_dirname, \
#             patch('LLM_QA.os.path.abspath') as mock_abspath, \
#             patch('LLM_QA.QuestionAnsweringModel') as mock_qa_model:
#         # Mock os.path.dirname and os.path.abspath to simulate the correct model_path
#         mock_dirname.return_value = '/dir/to'
#         mock_abspath.return_value = '/dir/to/fake_file.py'
#
#         bert_qa_scratch_false.load_saved_model()
#
#         # Assert that os.path.dirname and os.path.abspath were called with the correct arguments
#         mock_dirname.assert_called_once_with('/dir/to/fake_file.py')
#         mock_abspath.assert_called_once_with(expected_file_path + "/LLM_QA.py")
#
#         # Assert that QuestionAnsweringModel was called with the correct arguments
#         mock_qa_model.assert_called_once_with("bert", "/dir/to" + "/LLM_QA/bert/best_model", use_cuda=False)
#
#
# def test_bertqa_ask_questions(bert_qa_scratch_false):
#     with patch.object(LLM_QA.BertQA, 'load_saved_model'), \
#             patch.object(QuestionAnsweringModel, 'predict') as mock_predict:
#         context = "This is a context."
#         questions = ["What is the answer?", "Another question."]
#
#         # Mock the return values for the model.predict method
#         mock_predict.return_value = (["Answer1", "Answer2"], [0.8, 0.9])
#
#         result = bert_qa_scratch_false.ask_questions(context, questions)
#
#         # Assert that load_saved_model was called
#         LLM_QA.BertQA.load_saved_model.assert_called_once_with()
#
#         # Assert that predict was called with the correct input
#         mock_predict.assert_called_once_with(
#             [{"context": context, "qas": [{"question": q, "id": str(i)} for i, q in enumerate(questions)]}],
#             n_best_size=1)
#
#         # Assert the expected results for each question
#         assert result == {
#             "What is the answer?": {"answer": "Answer1", "probability": 0.8},
#             "Another question.": {"answer": "Answer2", "probability": 0.9}
#         }
#

def transform_data_from_files_to_tuples(file_paths):
    result = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

            for item in data:
                print("item ", item)
                context = item["context"]
                qas = item["qas"]

                for qa in qas:
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]  # Assuming there is always one answer

                    result.append((context, question, answer))

    return result


# @pytest.mark.parametrize("context, input_question, expected_answer",
#                          transform_data_from_files_to_tuples(
#                              ["./tests/data_for_tests/test_phase_shifts_data_test.json"]))
# def test_bertqa_ask_questions_with_data(bert_qa_scratch_false, context, input_question, expected_answer):
#     bert_answer = bert_qa_scratch_false.ask_questions(context, [input_question])
#     assert bert_answer[input_question]['answer']['answer'][0] == expected_answer