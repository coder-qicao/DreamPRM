class ResponseCollector:
    def __init__(self, out_path):
        self.responses = []
        self.out_path = out_path

    def add_response(self, processed_response, input, response_id, true_false):
        # processed response should be the output of generate_response.py
        # input shout be the input question of prompt_building.py
        # response_id can be num or dict
        self.responses.append({'id': response_id, 'response': processed_response, 'input': input, 'true_false': true_false})

    def add_MCTS_response(self, input, add, response_id, step_id, response_list, image_path, ground_truth, dataset):
        self.responses.append(
            {'id': response_id, 'step_id':step_id, 'response_list': response_list,
             'input': input, 'add': add, 'ground_truth': ground_truth, 'dataset': dataset, 'image_path': image_path})

    def get_responses(self):
        return self.responses

    def get_out_path(self):
        return self.out_path