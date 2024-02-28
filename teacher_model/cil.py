import argparse
import json
import pickle
import sys
from logging import getLogger

import torch
from flask import Flask, request, jsonify
from recbole.config import Config
from recbole.data import data_preparation, create_dataset, construct_transform
from recbole.utils import get_model, init_seed, init_logger, get_flops, set_color, get_trainer


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    users = data.get('users')
    item_length = data.get('item_lengths')
    k = data.get('k')
    item_lists = data.get('item_lists')
    candidate_item_lists = data.get('candidate_item_lists')
    target_category = data.get('target_category')
    max_len = max(item_length)
    for _ in item_lists:
        if len(_) < max_len:
            _ = ['[PAD]'] * (max_len - len(_)) + _
    batched_data = {
        'user_id': torch.tensor([dataset.field2token_id['user_id'][_] for _ in users], dtype=torch.int64).to(
            config["device"]),
        'item_length': torch.tensor(item_length, dtype=torch.int64).to(config["device"]),
        'item_id_list': torch.tensor([[dataset.field2token_id['item_id'][__] for __ in _] for _ in item_lists],
                                     dtype=torch.int64).to(config["device"]),
    }
    scores = trainer.model.full_sort_predict(batched_data)
    scores = torch.softmax(scores, dim=1)
    if candidate_item_lists is None and target_category is not None:
        candidate_item_lists = [
            list(set(category2item[_[1:]])-set(__))
            if _[0] == '+' else
            list(set(dataset.field2token_id['item_id'].keys())-set(category2item[_[1:]])-{'[PAD]'}-set(__))
            for _, __ in zip(target_category, item_lists)
        ]
    elif candidate_item_lists is None:
        candidate_item_lists = [
            list(set(dataset.field2token_id['item_id'].keys())-set(_)-{'[PAD]'})
            for _ in item_lists
        ]
    if candidate_item_lists is not None:
        candidate_id_lists = torch.tensor([
            [dataset.field2token_id['item_id'][__] for __ in _ if __ in dataset.field2token_id["item_id"]]
            for _ in candidate_item_lists],
            dtype=torch.int64).to(config["device"])

        for idx, _ in enumerate(scores):
            scores[idx][candidate_id_lists[idx]] += 1
        scores -= 1
        scores[..., dataset.field2token_id['item_id']['[PAD]']] = -torch.inf
    top_k_indices = torch.topk(scores, k).indices
    top_k = [[dataset.field2id_token['item_id'][__] for __ in _] for _ in top_k_indices.tolist()]

    return jsonify({'inference': top_k})


@app.route('/ranking', methods=['POST'])
def ranking():
    data = request.get_json()
    users = data.get('users')
    item_length = data.get('item_lengths')
    item_lists = data.get('item_lists')
    candidate_item_lists = data.get('candidate_item_lists')
    max_len = max(item_length)
    for _ in item_lists:
        if len(_) < max_len:
            _ = ['[PAD]'] * (max_len - len(_)) + _
    batched_data = {
        'user_id': torch.tensor([dataset.field2token_id['user_id'][_] for _ in users], dtype=torch.int64).to(
            config["device"]),
        'item_length': torch.tensor(item_length, dtype=torch.int64).to(config["device"]),
        'item_id_list': torch.tensor([[dataset.field2token_id['item_id'][__] for __ in _] for _ in item_lists],
                                     dtype=torch.int64).to(config["device"]),
    }
    scores = trainer.model.full_sort_predict(batched_data)
    scores = torch.softmax(scores, dim=1)

    assert candidate_item_lists is not None
    candidate_id_lists = torch.tensor([
        [dataset.field2token_id['item_id'][__] for __ in _ if __ in dataset.field2token_id["item_id"]]
        for _ in candidate_item_lists],
        dtype=torch.int64).to(config["device"])

    scores[..., dataset.field2token_id['item_id']['[PAD]']] = -torch.inf

    ranked_indices = torch.argsort(scores, descending=True)
    ranked_indices = torch.argsort(ranked_indices)
    ranking = [
        _[candidate_id_lists[idx]].tolist()
        for idx, _ in enumerate(ranked_indices)
    ]
    return jsonify({'ranking': ranking})


model_path = {
    'beauty': 'saved/beauty.pth',
    'sub_movie': 'saved/sub_movie_10.pth',
    'steam': 'saved/steam.pth',
}
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sub_movie')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()

    category2item = load_pickle(f'dataset/{args.dataset}/category1.pickle')

    config = Config(
        model="SASRec",
        dataset=args.dataset,
        config_file_list=['sasrec.yaml'],
        config_dict={
            'state': 'CRITICAL'
        }
    )

    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model evaluation
    setattr(trainer.eval_collector.register, 'rec.score', True)
    if True:
        # checkpoint_file = 'saved/SASRec-Jul-15-2023_03-52-17.pth'
        checkpoint_file = model_path[args.dataset]
        checkpoint = torch.load(checkpoint_file, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
        message_output = "Loading model structure and parameters from {}".format(
            checkpoint_file
        )
        trainer.logger.info(message_output)

    trainer.model.eval()

    eval_func = trainer._full_sort_batch_eval
    if trainer.item_tensor is None:
        trainer.item_tensor = test_data._dataset.get_item_feature().to(trainer.device)

    trainer.tot_item_num = test_data._dataset.item_num

    app.run(debug=False, host='0.0.0.0', port=args.port)
