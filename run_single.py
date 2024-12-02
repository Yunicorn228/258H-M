import os
import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color
from trainer import Trainer

from utils import get_model, create_dataset


def run_single(model_name, dataset_name, **kwargs):
    # configurations initialization
    props = ['props/overall.yaml', f'props/{model_name}.yaml']
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
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

    # trainer loading and initialization
    config['total_steps'] = config['epochs'] * len(train_data)
    logger.info('total steps: ' + str(config['total_steps']))
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'model_path': trainer.saved_model_file
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='SASRec', help='model name')
    parser.add_argument('-d', type=str, default='Sports', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    run_single(args.m, args.d)
