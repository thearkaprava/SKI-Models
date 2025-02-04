import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, MetricMeterMCA, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper

from datasets.build_AS_p1 import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from trainers import P2_SKIVLM_model as skivlm 

from torchlight import DictAction
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

import pickle


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--resume_pose', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')

    #HYPERFORMER ARGS
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    # model
    parser.add_argument('--hyp_model', default='trainers.Hyperformer.Hyperformer_Model', help='the model will be used')
    parser.add_argument('--graph', default='graph.ntu_rgb_d.Graph', help='the graph will be used')
    parser.add_argument('--weights', 
        default = [],
        help='the weights of Hyperformer')

    parser.add_argument(
        '--model_args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

def main(args, config):
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    class_names = [class_name for i, class_name in train_data.classes]

    # Custom trainer for different variants of SKI-ViFi-CLIP
    model = skivlm.returnCLIP(config,
                                logger=logger,
                                class_names=class_names,
                                args=args) # pass args to indicate that we want to load the hyperformer model

    model = model.cuda()  # changing to cuda here

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                       smoothing=config.AUG.LABEL_SMOOTH,
                                       mixup_alpha=config.AUG.MIXUP,
                                       cutmix_alpha=config.AUG.CUTMIX,
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    dist_loss_func = nn.MSELoss() ## Distillation Loss Function

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=True) #find_unused_parameters=False
    
    writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT, "logs"))  # Initialize TensorBoard writer

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME or config.MODEL.RESUME_POSE:
        # start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        start_epoch, max_accuracy = load_checkpoint(config, model, logger)
        if start_epoch > 1:
            logger.info("resetting epochs no and max. accuracy to 0 after loading pre-trained weights")
            start_epoch = 0
            max_accuracy = 0
    if config.TEST.ONLY_TEST:
        acc1, mca = validate(val_loader, model, config, writer, config.TRAIN.EPOCHS)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1[0]:.1f}%. mCA: {mca[0]:.1f}")
        return
    

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, dist_loss_func, optimizer, lr_scheduler, train_loader, config, mixup_fn, writer)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            modality_names = ['Image-Text', 'Text-Pose', 'Image-Text-Pose']
            # modality_names = ['Text-Pose']
            acc1_list, mca_list = validate(val_loader, model, config, writer, epoch)
            # acc1, mca = validate(val_loader, model, hyp_model, config)

            logger.info(f"Accuracy of the network on the {len(val_data)} test videos on {modality_names[0]}: {acc1_list[0]:.1f}%. mCA: {mca_list[0]:.1f}")
            is_best = acc1_list[0] > max_accuracy
            max_accuracy = max(max_accuracy, acc1_list[0])

            # logger.info(f"Accuracy of the network on the {len(val_data)} test videos on {modality_names[0]}: {acc1_list[0]:.1f}%. mCA: {mca_list[0]:.1f}")
            # is_best = acc1_list[0] > max_accuracy
            # max_accuracy = max(max_accuracy, acc1_list[0])

            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            if dist.get_rank() == 0 and (
                    epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1) or is_best):
                epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT,
                             is_best)
    
    writer.close()  # Close the TensorBoard writer

    # Now doing the multi-view inference crop for videos
    # 4 CLIPs are obtained from each video, and for each CLIP, we get 3 crops (augmentations)
    multi_view_inference = config.TEST.MULTI_VIEW_INFERENCE
    if multi_view_inference:
        config.defrost()
        config.TEST.NUM_CLIP = 4
        config.TEST.NUM_CROP = 3
        config.freeze()
        train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

        acc1, mca = validate(val_loader, model, config, writer, config.TRAIN.EPOCHS)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%. mCA: {mca:.1f}")

def train_one_epoch(epoch, model, criterion, dist_loss_func, optimizer, lr_scheduler, train_loader, config, mixup_fn, writer):
    model.train()
    # hyp_model.train()

    # print("Debug: Setting Hyperformer Module to eval mode!")
    # model.module.hyperformer_model.eval()

    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    dist_loss_meter = AverageMeter()
    image_text_loss_meter = AverageMeter()
    pose_text_loss_meter = AverageMeter()
    # total_ce_loss_meter = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        hyperformer_data = batch_data["hformer_data"].cuda(non_blocking=True) # B x 216
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        '''Get model outputs'''

        image_text_logits, text_pose_logits = model(images, hyperformer_data)
        image_text_loss = criterion(image_text_logits, label_id)
        pose_text_loss = criterion(text_pose_logits, label_id)
        total_ce_loss = image_text_loss + pose_text_loss
        dist_loss = dist_loss_func(image_text_logits, text_pose_logits)

        if config.TRAIN.LOSS_SCALE:
            alpha = config.TRAIN.LOSS_SCALE
            total_loss = total_ce_loss + alpha*dist_loss 
            # total_loss = dist_loss + alpha*image_text_loss # + alpha*text_pose_loss
            # total_loss = dist_loss + alpha*image_text_loss # + alpha*text_pose_loss
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        else:
            ## Total Loss = Loss_{distil} + Loss_{img-text} + Loss_{pose-text}
            total_loss = dist_loss + image_text_loss # + text_pose_loss 
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        '''Backprop & lr schedule update'''
        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()

        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        '''Allow the gradient to accumulate over TRAIN.ACCUMULATION_STEPS batches and then do backprop & update lr_scheduler'''
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
                # lr_scheduler.step()
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
            # lr_scheduler.step()

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        dist_loss_meter.update(dist_loss.item(), len(label_id))
        image_text_loss_meter.update(image_text_loss.item(), len(label_id))
        pose_text_loss_meter.update(pose_text_loss.item(), len(label_id))
        # total_ce_loss_meter.update(total_ce_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'dist_loss {dist_loss_meter.val:.4f} ({dist_loss_meter.avg:.4f})\t'
                f'image_text_loss {image_text_loss_meter.val:.4f} ({image_text_loss_meter.avg:.4f})\t'
                f'pose_text_loss {pose_text_loss_meter.val:.4f} ({pose_text_loss_meter.avg:.4f})\t'
                # f'total_ce_loss {total_ce_loss_meter.val:.4f} ({total_ce_loss_meter.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            
            writer.add_scalar('Train/Total_Loss', tot_loss_meter.avg, epoch * num_steps + idx)
            writer.add_scalar('Train/Distillation_Loss', dist_loss_meter.avg, epoch * num_steps + idx)
            writer.add_scalar('Train/Image_Text_Loss', image_text_loss_meter.avg, epoch * num_steps + idx)
            writer.add_scalar('Train/Pose_Text_Loss', pose_text_loss_meter.avg, epoch * num_steps + idx)
            # writer.add_scalar('Train/Total_CE_Loss', total_ce_loss_meter.avg, epoch * num_steps + idx)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    


@torch.no_grad()
def validate(val_loader, model, config, writer, epoch=0):
    model.eval()
    modality_names = ['Image-Text', 'Text-Pose', 'Image-Text-Pose']
    # modality_names = ['Image-Text']
    acc1_meters = [AverageMeter() for _ in range(len(modality_names))]
    acc5_meters = [AverageMeter() for _ in range(len(modality_names))]
    mca_meters = [MetricMeterMCA(config.DATA.NUM_CLASSES) for _ in range(len(modality_names))]
    test_image_text_logits = None
    test_pose_text_logits = None
    ground_truth = []

    with torch.no_grad():
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            if "hformer_data" not in batch_data:    
                hyperformer_data = None
            else:
                hyperformer_data = batch_data["hformer_data"].cuda(non_blocking=True) # B x 216
            
            # hyperformer_data = batch_data["hformer_data"].cuda(non_blocking=True)
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = [torch.zeros((b, config.DATA.NUM_CLASSES)).cuda() for _ in range(3)]

            for i in range(n):
                image = _image[:, i, :, :, :, :]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                image_text_logits, text_pose_logits = model(image_input, hyperformer_data)
                # image_features, text_features, pose_embd = model(image_input, hyperformer_data)
                # text_features, pose_embd = model(image_input, hyperformer_data)

                # image_text_logits = image_features @ text_features.t()
                # text_pose_logits = pose_embd @ text_features.t()

                img_text_pose_logits = image_text_logits + text_pose_logits

                logits_list = [image_text_logits, text_pose_logits, img_text_pose_logits]
                # logits_list = [image_text_logits]

                # if test_pose_text_logits is None and test_image_text_logits is None:
                #     test_pose_text_logits = text_pose_logits
                #     test_image_text_logits = image_text_logits
                # else:
                #     test_pose_text_logits = torch.cat([test_pose_text_logits, text_pose_logits], dim=0)
                #     test_image_text_logits = torch.cat([test_image_text_logits, image_text_logits], dim=0)


                for k, logits in enumerate(logits_list):
                    similarity = logits.view(b, -1).softmax(dim=-1)
                    tot_similarity[k] += similarity
            
            # modality_names = ['Image-Text', 'Text-Pose', 'Image-Text-Pose']
            for k in range(len(modality_names)):
                values_1, indices_1 = tot_similarity[k].topk(1, dim=-1)
                values_5, indices_5 = tot_similarity[k].topk(5, dim=-1)
                acc1, acc5 = 0, 0

                conf_matrix = torch.zeros((config.DATA.NUM_CLASSES, config.DATA.NUM_CLASSES)).cuda()
                for i in range(b):
                    conf_matrix[label_id[i].item(), indices_1[i].item()] += 1
                    if indices_1[i] == label_id[i]:
                        acc1 += 1
                    if label_id[i] in indices_5[i]:
                        acc5 += 1

                acc1_meters[k].update(float(acc1) / b * 100, b)
                acc5_meters[k].update(float(acc5) / b * 100, b)
                mca_meters[k].update(conf_matrix)

                if idx % config.PRINT_FREQ == 0:
                # if idx % 1 == 0:
                    logger.info(
                        f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Acc@1 for {modality_names[k]}: {acc1_meters[k].avg:.3f}\t'
                        f'mCA for {modality_names[k]}: {mca_meters[k].mca:.3f}\t'
                    )

            label_id = label_id.reshape(-1)
            ground_truth.extend(label_id.tolist())

            del logits_list, _image, label_id, image, image_input

    # torch.save(test_image_text_logits, f'{config.OUTPUT}/test_image_text_logits.pt')
    # torch.save(test_pose_text_logits, f'{config.OUTPUT}/test_pose_text_logits.pt')


    # image_text_results = (test_image_text_logits, ground_truth)
    # with open(f'{config.OUTPUT}/test_image_text_logits.pickle', 'wb') as f:
    #     pickle.dump(image_text_results, f)

    # pose_text_results = (test_pose_text_logits, ground_truth)
    # with open(f'{config.OUTPUT}/test_pose_text_logits.pickle', 'wb') as f:
    #     pickle.dump(pose_text_results, f)

    for k in range(len(modality_names)):
        acc1_meters[k].sync()
        acc5_meters[k].sync()
        mca_meters[k].sync()
        torch.save(mca_meters[k].confusion_matrix, f'{config.OUTPUT}/{modality_names[k]}_confusion_matrix.pth')
        logger.info(f' * Acc@1 for {modality_names[k]}: {acc1_meters[k].avg:.3f} Acc@5 for {modality_names[k]}: {acc5_meters[k].avg:.3f} mCA for {modality_names[k]}: {mca_meters[k].mca:.3f}')

        if writer:
            writer.add_scalar(f'Validation/Accuracy@1_{modality_names[k]}', acc1_meters[k].avg, epoch)
            writer.add_scalar(f'Validation/mCA_{modality_names[k]}', mca_meters[k].mca, epoch)

    return [acc1_meter.avg for acc1_meter in acc1_meters], [mca_meter.mca for mca_meter in mca_meters]

if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:10001', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(args, config)
