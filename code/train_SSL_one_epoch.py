# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim
import os
import time
import Config_SSL as config
import warnings
import numpy as np
from utils_train import *


warnings.filterwarnings("ignore")

def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################


def train_one_epoch(loader, model_evi, all_text, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger, scaler, global_step):

    device = torch.device(config.device)
    logging_mode = 'Train' if model_evi.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    epoch_num = epoch + 1
    if logging_mode == 'Train':
        for i, (sampled_batch, names) in enumerate(loader, 1):

            try:
                loss_name = criterion._get_name()
            except AttributeError:
                loss_name = criterion.__name__

            # Take variable and put them to GPU
            images, masks = sampled_batch['image'], sampled_batch['label']
            images, masks = images.to(device), masks.to(device)   # images[b, 3, 224, 224], masks[b, 224, 224]

            # 获取文本数据--------------------------------------------------------------------
            text_str = []
            for txt_name in names:
                txt_str = all_text[txt_name]
                # txt_str = txt_str.split('\n')
                text_str.append(txt_str)

            with torch.cuda.amp.autocast(enabled=True):
                prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = model_evi(images, text_str)  # [b, 1, 224, 224]
                prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL = prob_V.half(), prob_L.half(), prob_VL.half(), evi_V.half(), evi_L.half(), evi_VL.half()

            #################################################################################################################
            target = masks.reshape(-1)
            target = F.one_hot(target, 2)
            evi_V_and_L = dict()
            evi_V_and_L[0] = evi_V
            evi_V_and_L[1] = evi_L
            loss_evi = get_loss(evi_V_and_L, evi_VL, target, epoch_num, num_classes=2, annealing_step=50, device=device)
            #################################################################################################################

            prob_final = prob_VL
            # loss1 = criterion(prob_V, masks.half())  # Loss_value = ([b, 1, 224, 224], [b, 1, 224, 224])
            # loss2 = criterion(prob_L, masks.half())
            loss_seg = criterion(prob_VL, masks.half())

            loss = ((loss_seg + loss_evi) / 2.0) + loss_sim * 0.2

            loss_ = loss.item()
            loss_all = loss

            if model_evi.training:
                optimizer.zero_grad()
                scaler.scale(loss_all).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                #--------------------------
                loss_all.detach_()
                prob_V.detach_()
                prob_L.detach_()
                prob_VL.detach_()
                prob_final.detach_()
                evi_V.detach_()
                evi_L.detach_()
                evi_VL.detach_()


            train_dice = criterion._show_dice(prob_final, masks.half())
            train_iou = iou_on_batch(masks, prob_final)

            batch_time = time.time() - end
            # if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            #     vis_path = config.visualize_path+str(epoch)+'/'
            #     if not os.path.isdir(vis_path):
            #         os.makedirs(vis_path)
            #     save_on_batch(images,masks,prob_final.float(),names,vis_path)
            dices.append(train_dice)

            time_sum += len(images) * batch_time
            loss_sum += len(images) * loss_
            iou_sum += len(images) * train_iou
            # acc_sum += len(images) * train_acc
            dice_sum += len(images) * train_dice

            if i == len(loader):
                average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
                average_time = time_sum / (config.batch_size*(i-1) + len(images))
                train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
                # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
                train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
            else:
                average_loss = loss_sum / (i * config.batch_size)
                average_time = time_sum / (i * config.batch_size)
                train_iou_average = iou_sum / (i * config.batch_size)
                # train_acc_average = acc_sum / (i * config.batch_size)
                train_dice_avg = dice_sum / (i * config.batch_size)

            end = time.time()
            if i % config.print_frequency == 0:
                print_summary(epoch + 1, i, len(loader), loss_, loss_name, batch_time,
                              average_loss, average_time, train_iou, train_iou_average,
                              train_dice, train_dice_avg, 0, 0,  logging_mode,
                              lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

            if config.tensorboard:
                step = epoch * len(loader) + i
                writer.add_scalar(logging_mode + '_' + loss_name, loss_, step)  # loss.item()改为loss

                # plot metrics in tensorboard
                writer.add_scalar(logging_mode + '_iou', train_iou, step)
                # writer.add_scalar(logging_mode + '_acc', train_acc, step)
                writer.add_scalar(logging_mode + '_dice', train_dice, step)


        if lr_scheduler is not None:
            lr_scheduler.step()

    if logging_mode == 'Val':
        for i, (sampled_batch, names) in enumerate(loader, 1):

            try:
                loss_name = criterion._get_name()
            except AttributeError:
                loss_name = criterion.__name__

            # Take variable and put them to GPU
            images, masks = sampled_batch['image'], sampled_batch['label']

            # 将数据分为: 有标签-无标签
            images, masks = images.to(device), masks.to(device)  # images[b, 3, 224, 224], masks[b, 224, 224], text[b, 10, 768]

            text_str = []
            for txt_name in names:
                txt_str = all_text[txt_name]
                # txt_str = txt_str.split('\n')
                text_str.append(txt_str)

            with torch.cuda.amp.autocast(enabled=True):
                prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, _ = model_evi(images, text_str)  # [b, 1, 224, 224]
                prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL = prob_V.half(), prob_L.half(), prob_VL.half(), evi_V.half(), evi_L.half(), evi_VL.half()

            prob_final = prob_VL
            loss_out = criterion(prob_final, masks.half())  # Loss_value = ([b, 1, 224, 224], [b, 1, 224, 224])
            loss_ = loss_out.item()

            train_dice = criterion._show_dice(prob_final, masks.half())
            train_iou = iou_on_batch(masks, prob_final)

            batch_time = time.time() - end
            if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
                vis_path = config.visualize_path+str(epoch)+'/'
                if not os.path.isdir(vis_path):
                    os.makedirs(vis_path)
                save_on_batch(images,masks,prob_final.float(),names,vis_path)
            dices.append(train_dice)

            time_sum += len(images) * batch_time
            loss_sum += len(images) * loss_
            iou_sum += len(images) * train_iou
            # acc_sum += len(images) * train_acc
            dice_sum += len(images) * train_dice

            if i == len(loader):
                average_loss = loss_sum / (config.batch_size_val*(i-1) + len(images))
                average_time = time_sum / (config.batch_size_val*(i-1) + len(images))
                train_iou_average = iou_sum / (config.batch_size_val*(i-1) + len(images))
                # train_acc_average = acc_sum / (config.batch_size_val*(i-1) + len(images))
                train_dice_avg = dice_sum / (config.batch_size_val*(i-1) + len(images))
            else:
                average_loss = loss_sum / (i * config.batch_size_val)
                average_time = time_sum / (i * config.batch_size_val)
                train_iou_average = iou_sum / (i * config.batch_size_val)
                # train_acc_average = acc_sum / (i * config.batch_size_val)
                train_dice_avg = dice_sum / (i * config.batch_size_val)

            end = time.time()
            if i % config.print_frequency == 0:
                print_summary(epoch + 1, i, len(loader), loss_, loss_name, batch_time,
                              average_loss, average_time, train_iou, train_iou_average,
                              train_dice, train_dice_avg, 0, 0,  logging_mode,
                              lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

            if config.tensorboard:
                step = epoch * len(loader) + i
                writer.add_scalar(logging_mode + '_' + loss_name, loss_, step)  # loss.item()改为loss

                # plot metrics in tensorboard
                writer.add_scalar(logging_mode + '_iou', train_iou, step)
                # writer.add_scalar(logging_mode + '_acc', train_acc, step)
                writer.add_scalar(logging_mode + '_dice', train_dice, step)


        if lr_scheduler is not None:
            lr_scheduler.step()

    return average_loss, train_dice_avg





