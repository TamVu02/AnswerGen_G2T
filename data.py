import os
import json
import re
import string
import numpy as np
from tqdm import tqdm
import sys
import copy
import random
import time
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler


class VNHistoryDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(VNHistoryDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                                  num_workers=args.num_workers)


class VNHistoryDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f)

        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode
        self.metric = "BLEU"

        self.mask_token = self.tokenizer.additional_special_tokens[0]
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        self.add_bos_id = []

    def __len__(self):
        return len(self.data)

    def linearize_v2(self, entity, entity_change, relation_change):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        if len(entity[0]) == 0:
            return [], ''  # string_label, string_label_token
        string_label = []
        string_label_tokens = ''
        string_label += entity_change[entity[0]][0]
        string_label_tokens += ' {}'.format(entity[0])

        for rel in entity[2]:
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]]
                rel_label_token = copy.deepcopy(rel[0])
                words_label = rel_label + entity_change[rel[1]][0]
                words_label_tokens = ' {} {}'.format(rel_label_token, rel[1])
                string_label += words_label
                string_label_tokens += words_label_tokens

        return string_label, string_label_tokens

    def get_all_entities_per_sample(self, mark_entity_number, mark_entity, entry):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry['kbs'][entity_id]
            if len(entity[0]) == 0:
                continue
            for rel in entity[2]:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0])
                    text_entity.add(rel[1])

        text_entity_list = list(text_entity)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = mark_entity + text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)

        return ent_change, rel_change

    def truncate_pair_ar(self, a, add_bos_id):
        # add_bos_id + a + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
        input_ids = add_bos_id + a + [self.tokenizer.eos_token_id]
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length
        return input_ids, attn_mask

    def ar_prep_data(self, answers, questions, add_bos_id):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask = self.truncate_pair_ar(questions, add_bos_id)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask

    def __getitem__(self, idx):

        entry = self.data[idx]

        entities = []
        for _ in entry['kbs']:
            entities.append(_)

        strings_label = []
        strings_label_tokens = ''

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry['kbs'][ele_entity][0] for ele_entity in entities]
        mark_entity_number = entities
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity, entry)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        total_entity = mark_entity + text_entity

        # for adding description in training data
        if 'title' in entry:
            entity = self.knowledge[entry['title_kb_id']]

            string_label, string_label_tokens = self.linearize_v2(
                entity,
                entity_change,
                relation_change)

            strings_label += string_label
            strings_label_tokens += string_label_tokens

        for i, entity_id in enumerate(entities):
            entity = entry['kbs'][entity_id]

            string_label, string_label_tokens = self.linearize_v2(
                entity,
                entity_change,
                relation_change)

            strings_label += string_label
            strings_label_tokens += string_label_tokens

        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens = [], '', [], ''
        current_text = random.choice(entry['text'])

        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask = self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id)

        words_label_ids = self.tokenizer.encode(current_text, add_special_tokens=False,
                                                max_length=self.args.max_output_length, padding='max_length',
                                                truncation=True)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length
        assert len(decoder_label_ids) == len(decoder_attn_mask) == len(words_label_ids) == self.args.max_output_length

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        words_label_ar = torch.LongTensor(words_label_ids)

        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, words_label_ar
