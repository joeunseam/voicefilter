import os
import math
import torch
import torch.nn as nn
import traceback

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from .early_stopping import EarlyStopping


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str, n_epochs): # 에포크 수정4
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceFilter(hp).cuda()
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        early_stop_mode = False #True
        ealry_stop_patience = 3
        early_stop_delta = 0.00001
        # validation_intervals = cheackpoint_intervals
        if early_stop_mode:
            early_stopping = EarlyStopping(
            patience=ealry_stop_patience,
            delta=early_stop_delta,
            checkpoint_file_path=pt_dir,
            )
        pre_epoch = 0

        criterion = nn.MSELoss()
        for epoch in range(1, n_epochs +1): # 에포크 수정5
            logger.info(f"----- Epoch {epoch} starts -----") # 에포크 수정6
            model.train()
            for dvec_mels, target_mag, mixed_mag in trainloader:
                target_mag = target_mag.cuda()
                mixed_mag = mixed_mag.cuda()

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda()
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask

                # output = torch.pow(torch.clamp(output, min=0.0), hp.audio.power)
                # target_mag = torch.pow(torch.clamp(target_mag, min=0.0), hp.audio.power)
                loss = criterion(output, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if epoch % hp.train.summary_interval == 0:
                    #writer.log_training(loss, step)
                    logger.info("Wrote summary at step %d" % step)

            # write loss to tensorboard
            if e == 1 or epoch % hp.train.checkpoint_interval == 0:
                writer.log_training(loss, epoch)
                logger.info("Wrote summary at epoch %d" % epoch)

                validation_loss = validate(audio, model, embedder, testloader, writer, step)
                num = 0

                if early_stop_mode:
                    message, early_stop, num = early_stopping.check_and_save(validation_loss, model, epoch)

                    logger.info(
                      f"[Epoch {epoch:>3}] "
                      f"T_loss: {loss:7.5f}, "
                      f"V_loss: {validation_loss:7.5f}, "
                      f"{message} | "
                    )

                    if early_stop:
                      break
                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if (not early_stop_mode) or num == 2:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % epoch)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)

                    # 이전 체크포인트 삭제 추가 (직전 파일은 제외)
                    prev_checkpoint_path = os.path.join(pt_dir, 'chkpt_%d.pt' % pre_epoch)
                    if os.path.exists(prev_checkpoint_path):
                      os.remove(prev_checkpoint_path)
                    pre_epoch = epoch
                    # --------------------------------------

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
