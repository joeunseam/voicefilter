import os
import time
import logging
import argparse

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model. Used for both logging and saving checkpoints.")
    parser.add_argument('-ep', '--epochs', type=int, default=10000,
                        help="Number of training epochs") # 에포크 수정1/6
    parser.add_argument('-p', '--early_stop_patience', type=int, default=3,
                        help="Number of early stop patience") # 얼리스탑 수정
    parser.add_argument('-d', '--ealry_stop_delta', type=float, default=0.00001,
                        help="Delta value of early stop") # 얼리스탑 수정
    parser.add_argument('-em', '--early_stop_mode', type=bool, default=False,
                        help="early stop mode") # 얼리스탑 수정
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(args.base_dir, hp.log.chkpt_dir, args.model)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(args.base_dir, hp.log.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    n_epochs = args.epochs # 에포크 수정2/6

    early_stop_patience = args.early_stop_patience
    early_stop_delta = args.ealry_stop_delta
    early_stop_mode = args.early_stop_mode

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    if hp.data.train_dir == '' or hp.data.test_dir == '':
        logger.error("train_dir, test_dir cannot be empty.")
        raise Exception("Please specify directories of data in %s" % args.config)

    writer = MyWriter(hp, log_dir)

    trainloader = create_dataloader(hp, args, train=True)
    testloader = create_dataloader(hp, args, train=False)

    train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str, n_epochs, early_stop_mode, early_stop_patience, early_stop_delta) # 에포크 수정3/6
