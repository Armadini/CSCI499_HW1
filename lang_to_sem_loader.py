import json
from black import out
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_string, build_output_tables, build_tokenizer_table

DATA_FILEPATH = "lang_to_sem_data.json"


def flatten_and_clean(data):
    # Flatten into one long list of shape: (instruction, (action, object))
    return [(preprocess_string(instruction), a_o)
            for sublist in data for (instruction, a_o) in sublist]


class LangToSemDataset(Dataset):

    def __init__(self, input_path: str, is_train: bool, vocab_size: int = 1000, debug=False):
        """Initializes a lang_to_sem_data dataset from JSON file."""
        self.vocab_size = vocab_size

        # Open the dataset file
        with open(input_path) as jsf:
            raw_data = json.load(jsf)

        # Read train data and build tables
        train_data = raw_data["train"]
        self.build_tables(train_data)
        # self.seq_len = 55

        # Use train or valid
        data_ = train_data if is_train else raw_data["valid_seen"]
        data = flatten_and_clean(data_)
        data = list(map(self.strings_to_number_tensors, data))

        if debug:
            data = data[:10]

        inputs, actions, objects = [i for i, a, o in data], [
            a for i, a, o in data], [o for i, a, o in data]
        self.inputs, self.actions, self.objects = torch.LongTensor(
            inputs), torch.LongTensor(actions), torch.LongTensor(objects)

        self.n_data = self.inputs.size(0)

    def build_tables(self, data):
        output_tables, tokenizer_tables = build_output_tables(
            data), build_tokenizer_table(data, vocab_size=self.vocab_size)
        self.actions_to_index, self.index_to_actions = output_tables[:2]
        self.targets_to_index, self.index_to_targets = output_tables[2:]
        self.vocab_to_index, self.index_to_vocab, self.seq_len = tokenizer_tables

        self.max_actions, self.max_objects = len(
            self.actions_to_index), len(self.targets_to_index)

    def strings_to_number_tensors(self, datapoint):
        instruction, (action, object) = datapoint

        instruction_nums = [self.vocab_to_index[word]
                            for word in instruction.split(" ") if word in self.vocab_to_index]
        if len(instruction_nums) > self.seq_len:
            instruction_nums = instruction_nums[:self.seq_len]
        instruction_nums = instruction_nums + \
            [self.vocab_to_index["<pad>"]] * \
            (self.seq_len - len(instruction_nums))

        action_num, object_num = self.actions_to_index[action], self.targets_to_index[object]
        action_vec = [i == action_num for i in range(self.max_actions)]
        object_vec = [i == object_num for i in range(self.max_objects)]

        return [instruction_nums, action_vec, object_vec]

    def __getitem__(self, index):
        # Get an index of the dataset
        return self.inputs[index], [self.actions[index], self.objects[index]]

    def __len__(self):
        # Number of instances
        return self.n_data


def get_loaders(input_path, batch_size=4, shuffle=False, debug=False):
    train_dataset = LangToSemDataset(input_path=input_path, is_train=True, debug=debug)
    valid_dataset = LangToSemDataset(input_path=input_path, is_train=False, debug=debug)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    metadata = {"max_actions": train_dataset.max_actions,
                "max_objects": train_dataset.max_objects}

    return train_dataloader, valid_dataloader, metadata
