from recbole.quick_start import run_recbole

# def load_json(file_path):
#     with open(file_path, "r") as f:
#         return json.load(f)
#
#
# def load_pickle(filename):
#     with open(filename, "rb") as f:
#         return pickle.load(f)
#
#
# config = Config(
#     model="SASRec",
#     dataset="sub_movie",
#     config_file_list=['sasrec.yaml'],
# )
#
# init_seed(config["seed"], config["reproducibility"])
# # logger initialization
# init_logger(config)
# logger = getLogger()
# logger.info(sys.argv)
# logger.info(config)
#
# # dataset filtering
# dataset = create_dataset(config)
# logger.info(dataset)
#
# # dataset splitting
# train_data, valid_data, test_data = data_preparation(config, dataset)
#
# # model loading and initialization
# init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
# model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
# logger.info(model)
#
# transform = construct_transform(config)
# flops = get_flops(model, dataset, config["device"], logger, transform)
# logger.info(set_color("FLOPs", "blue") + f": {flops}")
#
# # trainer loading and initialization
# trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
#
# # model training
# best_valid_score, best_valid_result = trainer.fit(
#     train_data, valid_data, saved=True, show_progress=config["show_progress"]
# )
#
#
# # model evaluation
# setattr(trainer.eval_collector.register, 'rec.score', True)
# if True:
#     checkpoint_file = 'saved/sub_movie_10.pth'
#     checkpoint = torch.load(checkpoint_file, map_location=trainer.device)
#     trainer.model.load_state_dict(checkpoint["state_dict"])
#     trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
#     message_output = "Loading model structure and parameters from {}".format(
#         checkpoint_file
#     )
#     trainer.logger.info(message_output)
#
# trainer.model.eval()
#
# eval_func = trainer._full_sort_batch_eval
# if trainer.item_tensor is None:
#     trainer.item_tensor = test_data._dataset.get_item_feature().to(trainer.device)
#
#
# trainer.tot_item_num = test_data._dataset.item_num
#
# iter_data = (
#     tqdm(
#         test_data,
#         total=len(test_data),
#         ncols=100,
#         desc=set_color(f"Evaluate   ", "pink"),
#     )
# )
#
# user_item = {}
# user_id = []
# item_id = []
#
# num_sample = 0
# for batch_idx, batched_data in enumerate(iter_data):
#     for idx, uid in enumerate(batched_data[0]['user_id']):
#         user_item[int(uid)] = int(batched_data[0]['item_id'][idx])
#     user_id.extend(batched_data[0]['user_id'].tolist())
#     item_id.extend(batched_data[0]['item_id'].tolist())
#
#     num_sample += len(batched_data)
#     interaction, scores, positive_u, positive_i = eval_func(batched_data)
#     if trainer.gpu_available:
#         iter_data.set_postfix_str(
#             set_color("GPU RAM: " + get_gpu_usage(trainer.device), "yellow")
#         )
#     trainer.eval_collector.eval_batch_collect(
#         scores, interaction, positive_u, positive_i
#     )
# trainer.eval_collector.model_collect(trainer.model)
# struct = trainer.eval_collector.get_data_struct()
#
# test_result = trainer.evaluator.evaluate(struct)
# logger.info(set_color("test result", "yellow") + f": {test_result}")
#
# score = struct.get('rec.score')
# # datamaps = load_json(f'dataset/{config.dataset}/datamaps.json')
# user_candidate = {
#     dataset.field2token_id['user_id'][user]:
#         [dataset.field2token_id['item_id'][i] for i in items[:config.candidate_num-1]]
#     for user, items in load_pickle(f'dataset/{config.dataset}/ranking_candidate.pickle').items()
# }
# user_candidate = {uid: [user_item[uid]]+iid for uid, iid in user_candidate.items()}
# score = [torch.index_select(s, 0, torch.tensor(user_candidate[user_id[idx]])).tolist() for idx, s in enumerate(score)]
# top_k = torch.cat(((torch.topk(torch.tensor(score), k=config.topk[0])[1] == 0).to(torch.int64), torch.tensor([[1]]*len(user_candidate))), dim=1)
# struct.set('rec.topk', top_k)
#
# test_result = trainer.evaluator.evaluate(struct)
# logger.info(set_color("test result on candidate", "yellow") + f": {test_result}")


run_recbole(model='SASRec', dataset='steam', config_file_list=['sasrec.yaml'])
