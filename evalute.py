import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color
from trainer import Trainer

from utils import get_model, create_dataset


def evaluate(model_name, dataset, pretrained_file='', **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', 'props/overall.yaml']
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # trainer loading and initialization
    config['total_steps'] = config['epochs'] * len(train_data)
    logger.info('total steps: ' + str(config['total_steps']))
    trainer = Trainer(config, model)
    trainer.load_category_vector(config, dataset)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    return config['model'], config['dataset'], {
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='SASRec', help='model name')
    parser.add_argument('-d', type=str, default='Sports', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pretrained model file')
    args, unparsed = parser.parse_known_args()
    print(args)

    evaluate(args.m, args.d, args.p)