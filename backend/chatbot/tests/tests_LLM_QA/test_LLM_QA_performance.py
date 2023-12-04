import json
import os
import sys
import random
from unittest.mock import patch, MagicMock, call

import pytest
from simpletransformers.question_answering import QuestionAnsweringModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import LLM_QA


@pytest.fixture
def bert_qa_scratch_false():
    # You can customize the parameters for testing
    return LLM_QA.BertQA(train_from_scratch=False)


def transform_data_from_files_to_tuples(file_paths, max_elements=None):
    result = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

            for item in data:
                context = item["context"]
                qas = item["qas"]

                for qa in qas:
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]  # Assuming there is always one answer

                    result.append((context, question, answer))

    if max_elements:
        # Shuffle the combined data
        random.shuffle(result)

        # Extract the first 'max_elements' elements
        result = result[:max_elements]

    return result


@pytest.mark.parametrize("context, input_question, expected_answer",
                         transform_data_from_files_to_tuples(
                             ["./tests/data_for_tests/test_rotation_data_test.json", "./tests/data_for_tests/test_phase_shifts_data_test.json"], max_elements=2000))
def test_bertqa_ask_questions_with_data(bert_qa_scratch_false, context, input_question, expected_answer):
    bert_answer = bert_qa_scratch_false.ask_questions(context, [input_question])
    assert bert_answer[input_question]['answer']['answer'][0] == expected_answer