from collections import OrderedDict
from tqdm import tqdm
import argparse
from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import cycle
from trainer import TrainerAE
import torch
import os

import signal
import sys
import gc

# 커스텀 예외 정의
class TimeoutException(Exception):
    pass

batch_info = {
    "items": [],
}
def timeout_handler(signum, frame):
    print(f"\n{'='*60}")
    print(f"Timeout occurred in dataset training: {batch_info}")
    print(f"{'='*60}")
    # sys.exit(1)
    raise TimeoutException("Dataset training timeout")  # sys.exit 대신

def main():
    # to ensure keeping train
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE('train')
    signal.signal(signal.SIGALRM, timeout_handler)

    # create network and training agent
    tr_agent = TrainerAE(cfg)


    # create dataloader
    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader_all = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            batch_info["items"] = data["id"]
            try:
                signal.alarm(100)  # 100초 타임아웃 시작
                outputs, losses = tr_agent.train_func(data)
                signal.alarm(0)  # 정상 완료, 타임아웃 취소

                pbar.set_description("EPOCH[{}][{}]".format(e, b))
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

                # validation step
                if clock.step % cfg.val_frequency == 0:
                    data = next(val_loader)
                    outputs, losses = tr_agent.val_func(data)

                clock.tick()

                tr_agent.update_learning_rate()
            except TimeoutException:
                print(f"Timeout occurred in dataset loading: {batch_info}")
                continue

        torch.cuda.empty_cache()
        gc.collect()
        if clock.epoch % 5 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        # if clock.epoch % 10 == 0:
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':

    torch.cuda.empty_cache()
    main()
