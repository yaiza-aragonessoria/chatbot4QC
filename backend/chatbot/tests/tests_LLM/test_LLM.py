# content of test_file_manager.py

import datetime
import os
import sys
from unittest.mock import patch

import numpy
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import LLM


@pytest.fixture
def bert_classifier():
    # You can customize the parameters for testing
    return LLM.BertTextClassifier(val_ratio=0.2, batch_size=32, epochs=1, num_labels=3)


def test_initialize_model(bert_classifier):
    assert bert_classifier.model is not None
    assert bert_classifier.model.config.num_labels == bert_classifier.num_labels
    assert bert_classifier.optimizer is not None
    assert bert_classifier.val_ratio == 0.2
    assert bert_classifier.batch_size == 32
    assert bert_classifier.epochs == 1
    assert bert_classifier.seed_val == 42
    assert isinstance(bert_classifier.tokenizer, BertTokenizer)


def test_create_model(bert_classifier):
    bert_classifier.model = None
    bert_classifier.create_model()
    assert bert_classifier.model is not None


def test_create_optimizer(bert_classifier):
    bert_classifier.optimizer = None
    bert_classifier.create_optimizer()
    assert bert_classifier.optimizer is not None


def test_set_seed(bert_classifier):
    bert_classifier.set_seed(seed=123)
    assert torch.initial_seed() == 123


def test_set_device(bert_classifier):
    bert_classifier.device = None
    bert_classifier.set_device()
    assert bert_classifier.device is not None


def test_load_data(bert_classifier):
    test_data_path = "./tests/data_for_tests/for_test_generated_questions.txt"
    bert_classifier.load_data(test_data_path)

    # Check if the dataframe is not empty
    assert not bert_classifier.df.empty

    # Check if 'text' and 'label' columns are present in the dataframe
    assert 'text' in bert_classifier.df.columns
    assert 'label' in bert_classifier.df.columns

    # Check if 'text' and 'labels' attributes are set
    assert bert_classifier.text is not None
    assert bert_classifier.labels is not None

    # Check if the length of 'text' and 'labels' match the number of rows in the dataframe
    assert len(bert_classifier.text) == len(bert_classifier.df)
    assert len(bert_classifier.labels) == len(bert_classifier.df)

    # Check if the labels are correctly loaded
    assert all(isinstance(label, numpy.int64) for label in bert_classifier.labels)
    assert all(isinstance(text, str) for text in bert_classifier.text)


def test_tokenize_data(bert_classifier):
    data = ["This is a test.", "Another example.", "Yet another example."]
    result = bert_classifier.tokenize_data(data)

    # Check if the result is a dictionary
    assert isinstance(result, dict)

    # Check if 'input_ids' and 'attention_mask' keys are present in the result
    assert 'input_ids' in result
    assert 'attention_mask' in result

    # Check if the values associated with 'input_ids' and 'attention_mask' are torch tensors
    assert isinstance(result['input_ids'], torch.Tensor)
    assert isinstance(result['attention_mask'], torch.Tensor)

    # Check if the shape of 'input_ids' and 'attention_mask' tensors is as expected
    assert result['input_ids'].shape == torch.Size([1, len(data), 32])
    assert result['attention_mask'].shape == torch.Size([1, len(data), 32])


def test_tokenize_data_empty_input(bert_classifier):
    data = []
    with pytest.raises(ValueError):
        bert_classifier.tokenize_data(data)


def test_tokenize_data_large_text(bert_classifier):
    # Create a large text that exceeds the specified max_length
    large_text = " ".join(["word"] * 100)
    data = [large_text, 'more words']

    result = bert_classifier.tokenize_data(data)

    # Check if the 'input_ids' and 'attention_mask' tensors are correctly truncated
    assert result['input_ids'].shape == torch.Size([1, len(data), 32])
    assert result['attention_mask'].shape == torch.Size([1, len(data), 32])


def test_tokenize_data_variable_length(bert_classifier):
    # Create input data with variable text lengths
    data = ["Short text.", "This is a longer text that exceeds the max_length."]

    result = bert_classifier.tokenize_data(data)

    # Check if 'input_ids' and 'attention_mask' tensors have correct shapes for variable lengths
    assert result['input_ids'].shape == torch.Size([1, len(data), 32])  # Length of the longest text
    assert result['attention_mask'].shape == torch.Size([1, len(data), 32])


def test_preprocess_data(bert_classifier):
    # Mock data for testing
    bert_classifier.text = ["This is a positive sentence.", "This is a negative sentence.",
                            "This is another negative sentence."]
    bert_classifier.labels = [1, 0, 3]

    bert_classifier.preprocess_data()

    # Check if 'token_id', 'attention_masks', and 'labels_tensor' attributes are set
    assert bert_classifier.token_id is not None
    assert bert_classifier.attention_masks is not None
    assert bert_classifier.labels_tensor is not None

    # Check if 'token_id', 'attention_masks' and 'labels_tensor' are torch tensors
    assert isinstance(bert_classifier.token_id, torch.Tensor)
    assert isinstance(bert_classifier.attention_masks, torch.Tensor)
    assert isinstance(bert_classifier.labels_tensor, torch.Tensor)

    # Check if the shapes of 'token_id', 'attention_masks', and 'labels_tensor' are as expected
    assert bert_classifier.token_id.shape == torch.Size([len(bert_classifier.labels), 32])
    assert bert_classifier.attention_masks.shape == torch.Size([len(bert_classifier.labels), 32])
    assert bert_classifier.labels_tensor.shape == torch.Size([len(bert_classifier.labels)])


def test_preprocess_data_empty_text(bert_classifier):
    # Set empty text for testing
    bert_classifier.text = []
    bert_classifier.labels = []

    with pytest.raises(RuntimeError):
        bert_classifier.preprocess_data()


def test_split_data(bert_classifier):
    # Mock data for testing
    bert_classifier.token_id = torch.randint(100, (100, 32))
    bert_classifier.attention_masks = torch.randint(2, (100, 32))
    bert_classifier.labels_tensor = torch.randint(2, (100,))

    bert_classifier.split_data()

    # Check if 'train_dataloader' and 'val_dataloader' attributes are set
    assert bert_classifier.train_dataloader is not None
    assert bert_classifier.val_dataloader is not None

    # Check if 'train_dataloader' and 'val_dataloader' are instances of DataLoader
    assert isinstance(bert_classifier.train_dataloader, DataLoader)
    assert isinstance(bert_classifier.val_dataloader, DataLoader)

    # Check if the lengths of 'train_dataloader' and 'val_dataloader' are as expected
    assert len(bert_classifier.train_dataloader.dataset) == int(
        (1 - bert_classifier.val_ratio) * len(bert_classifier.labels_tensor))
    assert len(bert_classifier.val_dataloader.dataset) == int(
        bert_classifier.val_ratio * len(bert_classifier.labels_tensor))


def test_split_data_empty_tensors(bert_classifier):
    # Set empty tensors for testing
    bert_classifier.token_id = torch.empty(0)
    bert_classifier.attention_masks = torch.empty(0)
    bert_classifier.labels_tensor = torch.empty(0)

    with pytest.raises(ValueError,
                       match="With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters."):
        bert_classifier.split_data()


def test_create_scheduler(bert_classifier):
    # Mock data for testing
    bert_classifier.train_dataloader = [1, 2, 3]
    bert_classifier.epochs = 3
    bert_classifier.optimizer = AdamW(bert_classifier.model.parameters(), lr=2e-5, eps=1e-8)

    bert_classifier.create_scheduler()

    # Check if 'scheduler' attribute is set
    assert bert_classifier.scheduler is not None
    assert isinstance(bert_classifier.scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_load_checkpoint_train_from_scratch(bert_classifier, tmp_path):
    with patch.object(bert_classifier.model, 'load_state_dict') as mock_load_state_dict, \
            patch('LLM.BertTextClassifier.create_model') as mock_create_model, \
            patch('LLM.BertTextClassifier.create_optimizer') as mock_create_optimizer, \
            patch('LLM.BertTextClassifier.load_data') as mock_load_data, \
            patch('LLM.BertTextClassifier.preprocess_data') as mock_preprocess_data, \
            patch('LLM.BertTextClassifier.split_data') as mock_split_data, \
            patch('LLM.BertTextClassifier.create_scheduler') as mock_create_scheduler:
        bert_classifier.load_checkpoint(train_from_scratch=False, path_to_model=None)

        # Check if load_state_dict is not called
        mock_load_state_dict.assert_not_called()

    # Check if 'starting_epoch' is set to 0
    assert bert_classifier.starting_epoch == 0


def test_load_checkpoint_load_existing_model(bert_classifier, tmp_path):
    # Create a dummy model checkpoint file
    test_model_checkpoint = "./tests/data_for_tests/quantum_LLM.pth"
    test_data_path = "./tests/data_for_tests/for_test_generated_questions.txt"

    with patch.object(bert_classifier.model, 'load_state_dict') as mock_load_state_dict, \
            patch('LLM.BertTextClassifier.create_model') as mock_create_model, \
            patch('LLM.BertTextClassifier.create_optimizer') as mock_create_optimizer, \
            patch('LLM.BertTextClassifier.load_data') as mock_load_data, \
            patch('LLM.BertTextClassifier.preprocess_data') as mock_preprocess_data, \
            patch('LLM.BertTextClassifier.split_data') as mock_split_data, \
            patch('LLM.BertTextClassifier.create_scheduler') as mock_create_scheduler:
        bert_classifier.load_checkpoint(train_from_scratch=False, path_to_model=str(test_model_checkpoint))

        # Check if load_state_dict is called
        mock_load_state_dict.assert_called_once_with(bert_classifier.pretrained_model['model_state_dict'])

    # Check if 'starting_epoch' is set to 0
    assert bert_classifier.starting_epoch == 0

    # Check if the model and optimizer state are loaded from the checkpoint
    assert isinstance(bert_classifier.pretrained_model, dict)
    assert 'model_state_dict' in bert_classifier.pretrained_model
    assert 'optimizer_state_dict' in bert_classifier.pretrained_model


def test_load_checkpoint_no_model_file(bert_classifier, capsys):
    bert_classifier.load_checkpoint(train_from_scratch=False, path_to_model=None)

    # Check if 'starting_epoch' is set to 0
    assert bert_classifier.starting_epoch == 0

    # Check if a message is printed to indicate fine-tuning from scratch
    captured = capsys.readouterr()
    assert "No path to model provided" in captured.out


def create_dummy_dataloader():
    # Create a dummy dataloader for testing
    token_ids = torch.randint(100, (50, 32))
    attention_masks = torch.randint(2, (50, 32))
    labels = torch.randint(2, (50,))

    dataset = TensorDataset(token_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=32)


def test_evaluate(bert_classifier):
    # Mock data for testing
    dataloader = create_dummy_dataloader()

    avg_loss, avg_accuracy = bert_classifier.evaluate(dataloader)

    # Check if the returned values are floats
    assert isinstance(avg_loss, float)
    assert isinstance(avg_accuracy, float)


def test_evaluate_empty_dataloader(bert_classifier):
    # Set an empty dataloader for testing
    empty_dataloader = DataLoader([])

    with pytest.raises(ZeroDivisionError):
        bert_classifier.evaluate(empty_dataloader)


def test_flat_accuracy(bert_classifier):
    # Mock data for testing
    preds = np.array([[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]])
    labels = np.array([0, 1, 0])

    accuracy = bert_classifier.flat_accuracy(preds, labels)

    # Check if the calculated accuracy is as expected
    assert accuracy == 2 / 3


def test_flat_accuracy_empty_input(bert_classifier):
    # Test with empty input arrays
    preds = np.array([])
    labels = np.array([])

    with pytest.raises(ValueError, match="axis 1 is out of bounds for array of dimension 1"):
        bert_classifier.flat_accuracy(preds, labels)

def tokenize_data(bert_classifier, data):
    # Tokenize and preprocess text data
    tokenized_data = bert_classifier.tokenizer(
        data,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
    )
    return {
        'input_ids': torch.tensor([tokenized_data['input_ids']]),
        'attention_mask': torch.tensor([tokenized_data['attention_mask']]),
    }

@pytest.fixture
def work_data_to_test_train(bert_classifier):
    test_data_path = "./tests/data_for_tests/for_test_generated_questions.txt"

    val_ratio = bert_classifier.val_ratio
    batch_size = bert_classifier.batch_size
    tokenizer = bert_classifier.tokenizer
    epochs = bert_classifier.epochs

    # load_data
    data = []
    with open(test_data_path) as f:
        for line in f.readlines():
            split = line.split('\t')
            data.append({'label': int(split[0]),
                         'text': split[1].rstrip()})
    df = pd.concat([pd.DataFrame(data)])

    text = df.text.values
    labels = df.label.values

    # preprocess_data
    # Preprocess data and prepare tensors
    token_id = []
    attention_masks = []

    for sample in text:
        encoding_dict = tokenize_data(bert_classifier, sample)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])

    # convert data and labels to tensors of pytorch
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensor = torch.tensor(labels)

    labels_tensor = torch.tensor(labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels_tensor)),
        test_size=val_ratio,
        shuffle=True,
        stratify=labels_tensor
    )

    train_set = TensorDataset(token_id[train_idx],
                              attention_masks[train_idx],
                              labels_tensor[train_idx])

    val_set = TensorDataset(token_id[val_idx],
                            attention_masks[val_idx],
                            labels_tensor[val_idx])

    train_dataloader = DataLoader(
        train_set,
        sampler=RandomSampler(train_set),
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_set,
        sampler=SequentialSampler(val_set),
        batch_size=batch_size
    )

    # create_scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        bert_classifier.optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return {"text": text, "labels": labels, "df": df, "token_id": token_id,
            "attention_masks": attention_masks, "labels_tensor": labels_tensor, "val_ratio": val_ratio,
            "batch_size": batch_size, "epochs": epochs, "seed_val": 42, "num_labels": bert_classifier.num_labels,
            "tokenizer": BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True),
            "scheduler": scheduler, "device": 'cpu', "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader, "pretrained_model": None, "starting_epoch": 0}


def test_train_with_data(bert_classifier, tmp_path, work_data_to_test_train):
    # Mock data for testing
    test_data_path = "./tests/data_for_tests/for_test_generated_questions.txt"

    data = work_data_to_test_train
    bert_classifier.text = data.get('text')
    bert_classifier.labels = data.get('labels')
    bert_classifier.df = data.get('df')
    bert_classifier.token_id = data.get('token_id')
    bert_classifier.attention_masks = data.get('attention_masks')
    bert_classifier.labels_tensor = data.get('labels_tensor')
    bert_classifier.scheduler = data.get('scheduler')
    bert_classifier.device = data.get('device')
    bert_classifier.train_dataloader = data.get('train_dataloader')
    bert_classifier.val_dataloader = data.get('val_dataloader')
    bert_classifier.pretrained_model = data.get('pretrained_model')
    bert_classifier.starting_epoch = data.get('starting_epoch')

    with patch.object(LLM.BertTextClassifier, 'load_data') as mock_load_data, \
            patch.object(LLM.BertTextClassifier, 'preprocess_data') as mock_preprocess_data, \
            patch.object(LLM.BertTextClassifier, 'split_data') as mock_split_data, \
            patch.object(LLM.BertTextClassifier, 'create_scheduler') as mock_create_scheduler, \
            patch.object(LLM.BertTextClassifier, 'load_checkpoint') as mock_load_checkpoint, \
            patch.object(BertForSequenceClassification, 'train') as mock_model_train, \
            patch.object(LLM.BertTextClassifier, 'evaluate') as mock_evaluate, \
            patch('torch.save') as mock_torch_save:

        mock_evaluate.return_value = [0.002277433749248969, 1.0]

        bert_classifier.train(train_from_scratch=True, path_to_model=None, map_location='cpu', data_path=test_data_path,
                              save=True)

        # Check if the necessary methods are called
        mock_load_data.assert_called_once_with(test_data_path)
        mock_preprocess_data.assert_called_once()
        mock_split_data.assert_called_once()
        mock_create_scheduler.assert_called_once()
        mock_load_checkpoint.assert_called_once()
        mock_load_checkpoint.assert_called_once_with(True, None, 'cpu')
        mock_model_train.assert_called_once()
        assert mock_evaluate.call_count == 2
        mock_torch_save.assert_called_once()


def test_train_no_data(bert_classifier, capsys):
    # Test when no data is provided
    with patch.object(bert_classifier, 'load_data') as mock_load_data:
        bert_classifier.train(train_from_scratch=True, path_to_model=None, map_location='cpu', data_path=None,
                              save=True)

        # Check if 'load_data' method is not called
        mock_load_data.assert_not_called()

        # Check if a message is printed to indicate no data provided
        captured = capsys.readouterr()
        assert "Model couldn't train because no data was provided." in captured.out
