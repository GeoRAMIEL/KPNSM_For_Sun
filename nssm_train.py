import torch
from model import create_model
from loss import create_loss, create_temporal_loss
from nssm_dataset import NSSMDataset
from knsm_utils import save_image, save_kernel, apply_kernel, expand_kernels, save_dilated_kernels, dilation_kernel_actual_size, reconstruct_full_kernel, visualize_and_save_difference, find_good_pixel_with_penumbra
from config_check import check_config
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.onnx
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from temporal import depth_tolerance, normal_tolerance
from evaluate import evaluate

def nssm_train(args):
    print(f"config file: {args.config}")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    run_name = config["name"]
    
    # device fallback: cuda -> mps -> cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"output/{run_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=False)
    logger = SummaryWriter(log_dir)

    check_config(run_name, config)

    with open(os.path.join(log_dir, "config.yaml"), 'w') as file:
        yaml.safe_dump(config, file)
    
    filter_config = config["filter"]
    filter_size = filter_config["filter_size"]
    dilation = filter_config["dilation"]
    assert filter_size % 2 == 1
    kernel_actual_size = dilation_kernel_actual_size(filter_size, dilation)
    kernel_half_actual_size = kernel_actual_size // 2
    print(f"kernel: size={filter_size}, dilation={dilation}, actual_size={kernel_actual_size}")

    Loss = create_loss(config["loss"])
    temporal_weight = config.get("temporal_weight", 0.0)
    use_temporal = temporal_weight > 0
    if use_temporal:
        Tloss = create_temporal_loss(config["temporal_loss"])

    model = create_model(config)
    if "ckpt" in config:
        model.load_state_dict(torch.load(config["ckpt"]))
        print("loading ckpt from", config["ckpt"])
    using_kpnsm_ckpt = config.get("using_kpnsm_ckpt", False)
    model = model.to(device)
    print("number of parameters:", sum(p.numel() for p in model.parameters()))
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.data.equal(torch.zeros_like(param.data)):
                print(f"Parameter {name} has not been initialized (all zeros).")
            else:
                print(f"Parameter {name} has been initialized.")
                print(f"Max value: {param.data.max().item()}, Min value: {param.data.min().item()}")
    """
    dataset_config = config["dataset"]
    val_use_temporal = dataset_config.get("val_use_temporal", False)
    # get train and val data dir
    train_data_dir = dataset_config["train"]
    train_data_feature_dir = dataset_config.get("train_feature_dir", None)
    train_data_gt_dir = dataset_config.get("train_gt_dir", None)
    val_data_dir = dataset_config["val"]
    val_data_feature_dir = dataset_config.get("val_feature_dir", None)
    val_data_gt_dir = dataset_config.get("val_gt_dir", None)
    val_data_range = dataset_config.get("val_data_range", None)
    dataset = NSSMDataset(train_data_feature_dir, train_data_gt_dir, gt_index_multiplier=dataset_config.get("gt_index_multiplier", 1), use_temporal=use_temporal, **dataset_config["configs"])
    dataloader = DataLoader(dataset, batch_size=config["training"]["train_batch_size"], shuffle=True, num_workers=dataset_config["train_workers"])
    validation = NSSMDataset(val_data_feature_dir, val_data_gt_dir, gt_index_multiplier=dataset_config.get("gt_index_multiplier", 1), use_temporal=val_use_temporal, data_range=val_data_range, **dataset_config["configs"])
    # dataset_config["configs"]["filter_gt"] = pre_k
    valDataloader = DataLoader(validation, batch_size=config["val"]["val_batch_size"], shuffle=False, num_workers=dataset_config["val_workers"])

    max_step = config["training"].get("max_step", 1000000000)
    train_accum = config.get("train_batch_accum", 1)
    optimizer = Adam(model.parameters(), config["training"]["learning_rate"] / train_accum)
    scheduler = MultiStepLR(optimizer, milestones=[20000, 40000, 80000], gamma=0.1)
    gradient_clip = config["training"].get("gradient_clip", 0.0) # 0.0 for no clipping
    running_loss = 0.0
    running_loss_base = 0.0
    running_loss_temporal = 0.0
    running_start = time.time()
    running_cnt = 0
    step = 0
    if "ckpt" in config:
        if using_kpnsm_ckpt:
            print("Starting fresh training from KPNSM checkpoint")
        else:
            step = int(os.path.split(config["ckpt"])[-1][5:-4])
            print("resuming from step", step)
    model.train()
    ep = 0
    while step < max_step:
        print("Epoch", ep)
        ep += 1
        for input_dict in dataloader:
            if use_temporal:
            #    model.eval()
            #    with torch.no_grad():
            #        ce = input_dict["ce_t"]
            #        cv = input_dict["cv_t"]
            #        depthInShadowMap = input_dict["depthInShadowMap_t"]
            #        distE = input_dict["distE_t"]
            #        # normalE = input_dict["normE"]
            #        # normalV = input_dict["normV"]
            #        depthDiff = (distE - depthInShadowMap)
            #        depthDiv = (depthInShadowMap / (distE + 1e-10))
            #        # gt_prev = input_dict["gt_t"]
            #        mask = input_dict["mask_t"].to(device)
            #        penumbra_width = input_dict["penumbra_width_t"]
            #        if config["include_de_ds"]:
            #            if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
            #                x = [ce, cv, distE, penumbra_width, depthDiff, depthDiv, depthInShadowMap]
            #            else:
            #                x = [ce, cv, distE, depthDiff, depthDiv, depthInShadowMap]
            #        else:
            #            if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
            #                x = [ce, cv, penumbra_width, depthDiff, depthDiv]
            #            else:
            #                x = [ce, cv, depthDiff, depthDiv]
            #        if config["include_sm"]:
            #            if "msm" in input_dict:
            #                msm = input_dict["msm_t"]
            #                x.append(msm)
            #            else:
            #                shadowMap = input_dict["shadowMap_t"]
            #                x.append(shadowMap)
            #        else:
            #            shadowMap = input_dict["shadowMap_t"]
            #        x = torch.cat(x, dim=1).to(device)
            #        if filter_size > 1:
            #            kernel = model(x)
            #            kernel = expand_kernels(kernel, filter_size, dilation)
            #            if "msm" in input_dict:
            #                shadowMap_filtered_prev = apply_kernel(msm.to(device), kernel, filter_size, dilation)
            #            else:
            #                shadowMap_filtered_prev = apply_kernel(shadowMap.to(device), kernel, filter_size, dilation)
            #        else:
            #            shadowMap_filtered_prev = model(x)            
            #        if config["temporal_loss"]["proj"] == True:   
            #            B, _, H, W = ce.shape
            #            view_proj_prev = input_dict["view_proj_t"].to(device) 
            #            normalW_prev = input_dict["normW_t"].to(device)
            #            depth_prev = input_dict["depth_t"].to(device)
            #            pos = input_dict["posW"].to(device) # (B, 3, H, W)
            #            pos = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1) # (B, 4, H, W)
#
#
            #            last_ss_pos = torch.einsum('bij,bjhw->bihw', view_proj_prev, pos) # (B, 4, H, W)
            #            last_ss_pos = last_ss_pos[:, :2] / last_ss_pos[:, 3:4] # (B, 2, H, W)
            #            last_ss_pos = last_ss_pos / 2.0 + 0.5
            #            last_i, last_j = last_ss_pos[:, 1] * H, last_ss_pos[:, 0] * W # (B, H, W)
            #            last_i = torch.floor(last_i).to(torch.int32)
            #            last_j = torch.floor(last_j).to(torch.int32)
#
            #            valid = ((last_i >= 0) & (last_i < H) & (last_j >= 0) & (last_j < W)).unsqueeze(1) & mask # (B, 1, H, W)
            #            last_i = torch.clamp(last_i, 0, H-1)
            #            last_j = torch.clamp(last_j, 0, W-1)
            #            b = torch.arange(B).view(B, 1, 1).expand(B, H, W).to(device)
            #            shadowMap_filtered_projected = torch.permute(shadowMap_filtered_prev[b, :, last_i, last_j], (0, 3, 1, 2))
            #            # gt_projected = torch.permute(gt_prev[b, :, last_i, last_j], (0, 3, 1, 2)) # (B, 1, H, W)
            #            depth_projected = torch.permute(depth_prev[b, :, last_i, last_j], (0, 3, 1, 2)) # (B, 1, H, W)
            #            normal_projected = torch.permute(normalW_prev[b, :, last_i, last_j], (0, 3, 1, 2)) # (B, 3, H, W)
            #        
            #            normal = input_dict["normW"].to(device)
            #            depth = input_dict["depth"].to(device)
            #            combined_mask = mask & valid & (torch.abs(depth - depth_projected) < depth_tolerance) & (torch.norm(normal - normal_projected, dim=1, p=2, keepdim=True) < normal_tolerance)
            #        else:
            #            shadowMap_filtered_projected = shadowMap_filtered_prev
            #            combined_mask = mask
#
            #        if Tloss.use_vgg():
            #            vgg_prev = Loss.vgg_loss.get_features(shadowMap_filtered_projected)
            #        
#
            #    model.train()
                print("Warning: temporal loss not adapted yet")

            ce = input_dict["ce"]
            cv = input_dict["cv"]
            shadowStdDev = input_dict["stdDev"]
            depthDiff = input_dict["distRtoB"] # depthDiff = dist(receiver, blocker)
            distVtoR = input_dict["distVtoR"]
            gt = input_dict["gt"].to(device)
            mask = input_dict["mask"].to(device)
            shadowDvg = input_dict["shadowDvg"]
            shadowMap = input_dict["shadowMap"]
            
            B, _, H, W = ce.shape

            if step % train_accum == 0: # start of new round
                optimizer.zero_grad()
            
            # NSSM input features
            x = [ce, cv, distVtoR, shadowDvg, depthDiff, shadowStdDev[0], shadowStdDev[1], shadowMap]
            x = torch.cat(x, dim=1).to(device)

            if filter_size > 1:
                kernel = model(x)
                kernel = expand_kernels(kernel, filter_size, dilation)
                #if "msm" in input_dict:
                #    shadowMap_filtered = apply_kernel(msm.to(device), kernel, filter_size, dilation)
                #else:
                #    shadowMap_filtered = apply_kernel(shadowMap.to(device), kernel, filter_size, dilation)
                shadowMap_filtered = apply_kernel(shadowMap.to(device), kernel, filter_size, dilation)
            else:
                shadowMap_filtered = model(x)

            if use_temporal and Tloss.use_vgg():
                loss, vgg_cur = Loss.apply(shadowMap_filtered, gt, mask=mask, return_vgg=True)
            else:
                loss = Loss.apply(shadowMap_filtered, gt, mask=mask)
            total_loss = loss
            running_loss += loss.item()
            running_loss_base += loss.item()

            #if use_temporal:
            #    if Tloss.use_vgg():
            #        tloss = Tloss.apply(shadowMap_filtered, shadowMap_filtered_projected, vgg_cur, vgg_prev, combined_mask)
            #    else:
            #        tloss = Tloss.apply(shadowMap_filtered, shadowMap_filtered_projected, combined_mask)
            #    total_loss += temporal_weight * tloss
            #    running_loss += tloss.item()
            #    running_loss_temporal += tloss.item()
            
            running_cnt += 1
            step += 1

            total_loss.backward()
            if step % train_accum == 0: # next step is start of new round
                if gradient_clip != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
            scheduler.step()

            if running_cnt % 20 == 0:
                running_loss_avg = running_loss / running_cnt
                running_loss_base_avg = running_loss_base / running_cnt
                running_loss_temporal_avg = running_loss_temporal / running_cnt
                running_time_avg = (time.time() - running_start) / running_cnt
                print(f"step {step}, running_loss = {running_loss_avg} running_loss_base = {running_loss_base_avg}, running_loss_temporal = {running_loss_temporal_avg}, speed={running_time_avg} per cnt")
                running_cnt = 0
                running_loss = 0.0
                running_loss_base = 0.0
                running_loss_temporal = 0.0
                running_start = time.time()
                logger.add_scalar("Loss/running_train", running_loss_avg, step)
                logger.add_scalar("Loss/running_train_base", running_loss_base_avg, step)
                logger.add_scalar("Loss/running_train_temporal", running_loss_temporal_avg, step)
                logger.add_scalar("Speed/train", running_time_avg, step)

            if step == 1: # save a dummy model
                torch.save(model.state_dict(), os.path.join(log_dir, f"model{step}.pth"))

            if step % config["val"]["val_frequency"] == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, f"model{step}.pth"))

                val_log_dir_h = os.path.join(log_dir, f"{step}half")
                evaluate(step, model, config, val_log_dir_h, logger, valDataloader, device, half=True, prefix="half")
            
            if step >= max_step:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSSM Training configuration.")
    parser.add_argument('--config', type=str, default="config/nssm_default.yaml", help='Path to the YAML configuration file.')
    args = parser.parse_args()

    nssm_train(args)
