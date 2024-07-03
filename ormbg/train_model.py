import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from pathlib import Path

from models.ormbg import ORMBG

from data_loader_cache import (
    get_im_gt_name_dict,
    create_dataloaders,
    GOSGridDropout,
    GOSRandomHFlip,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    net,
    optimizer,
    train_dataloaders,
    train_datasets,
    valid_dataloaders,
    valid_datasets,
    hypar,
):

    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    ite_num = hypar["start_ite"]  # count the toal iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets[0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0

    for epoch in range(epoch_num):

        for i, data in enumerate(gos_dataloader):

            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels = data["image"], data["label"]

            if hypar["model_digit"] == "full":
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(
                    inputs.cuda(), requires_grad=False
                ), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
                    labels, requires_grad=False
                )

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            ds, _ = net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(
                ">>>"
                + model_path.split("/")[-1]
                + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f"
                % (
                    epoch + 1,
                    epoch_num,
                    (i + 1) * batch_size_train,
                    train_num,
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,
                    time.time() - start_last,
                    time.time() - start_last - end_inf_loss_back,
                )
            )
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(
                    net, valid_dataloaders, valid_datasets, hypar, epoch
                )
                net.train()  # resume train

                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if tmp_f1[fi] > last_f1[fi]:
                        tmp_out = 1
                print("tmp_out:", tmp_out)
                if tmp_out:
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = "_".join(tmp_f1_str)
                    meanM = "_".join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = (
                        "/gpu_itr_"
                        + str(ite_num)
                        + "_traLoss_"
                        + str(np.round(running_loss / ite_num4val, 4))
                        + "_traTarLoss_"
                        + str(np.round(running_tar_loss / ite_num4val, 4))
                        + "_valLoss_"
                        + str(np.round(val_loss / (i_val + 1), 4))
                        + "_valTarLoss_"
                        + str(np.round(tar_loss / (i_val + 1), 4))
                        + "_maxF1_"
                        + maxf1
                        + "_mae_"
                        + meanM
                        + "_time_"
                        + str(
                            np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)
                        )
                        + ".pth"
                    )
                    torch.save(net.state_dict(), model_path + model_name)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if notgood_cnt >= hypar["early_stop"]:
                    print(
                        "No improvements in the last "
                        + str(notgood_cnt)
                        + " validation periods, so training stopped !"
                    )
                    exit()

    print("Training Reaches The Maximum Epoch Number")


def main(train_datasets, valid_datasets, hypar):

    print("--- create training dataloader ---")

    train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    ## build dataloader for training datasets
    train_dataloaders, train_datasets = create_dataloaders(
        train_nm_im_gt_list,
        cache_size=hypar["cache_size"],
        cache_boost=hypar["cache_boost_train"],
        my_transforms=[GOSGridDropout(), GOSRandomHFlip()],
        batch_size=hypar["batch_size_train"],
        shuffle=True,
    )

    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")

    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_nm_im_gt_list,
        cache_size=hypar["cache_size"],
        cache_boost=hypar["cache_boost_valid"],
        my_transforms=[],
        batch_size=hypar["batch_size_valid"],
        shuffle=False,
    )

    net = hypar["model"]

    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if hypar["restore_model"] != "":
        print("restore model from:")
        print(hypar["model_path"] + "/" + hypar["restore_model"])
        if torch.cuda.is_available():
            net.load_state_dict(
                torch.load(hypar["model_path"] + "/" + hypar["restore_model"])
            )
        else:
            net.load_state_dict(
                torch.load(
                    hypar["model_path"] + "/" + hypar["restore_model"],
                    map_location="cpu",
                )
            )

    optimizer = optim.Adam(
        net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    train(
        net,
        optimizer,
        train_dataloaders,
        train_datasets,
        valid_dataloaders,
        valid_datasets,
        hypar,
    )


if __name__ == "__main__":

    output_model_folder = "saved_models"
    Path(output_model_folder).mkdir(parents=True, exist_ok=True)

    train_datasets, valid_datasets = [], []
    dataset_1, dataset_1 = {}, {}

    dataset_training = {
        "name": "ormbg-training",
        "im_dir": str(Path("dataset", "training", "im")),
        "gt_dir": str(Path("dataset", "training", "gt")),
        "im_ext": ".png",
        "gt_ext": ".png",
        "cache_dir": str(Path("cache", "teacher", "training")),
    }

    dataset_validation = {
        "name": "ormbg-training",
        "im_dir": str(Path("dataset", "validation", "im")),
        "gt_dir": str(Path("dataset", "validation", "gt")),
        "im_ext": ".png",
        "gt_ext": ".png",
        "cache_dir": str(Path("cache", "teacher", "validation")),
    }

    train_datasets = [dataset_training]
    valid_datasets = [dataset_validation]

    ### --------------- STEP 2: Configuring the hyperparamters for Training, validation and inferencing ---------------
    hypar = {}

    hypar["model"] = ORMBG()
    hypar["seed"] = 0

    ## model weights path
    hypar["model_path"] = "saved_models"

    ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
    hypar["restore_model"] = ""

    ## start iteration for the training, can be changed to match the restored training process
    hypar["start_ite"] = 0

    ## indicates "half" or "full" accuracy of float number
    hypar["model_digit"] = "full"

    ## To handle large size input images, which take a lot of time for loading in training,
    #  we introduce the cache mechanism for pre-convering and resizing the jpg and png images into .pt file
    hypar["cache_size"] = [
        1024,
        1024,
    ]

    ## cached input spatial resolution, can be configured into different size
    ## "True" or "False", indicates wheather to load all the training datasets into RAM, True will greatly speed the training process while requires more RAM
    hypar["cache_boost_train"] = False

    ## "True" or "False", indicates wheather to load all the validation datasets into RAM, True will greatly speed the training process while requires more RAM
    hypar["cache_boost_valid"] = False

    ## stop the training when no improvement in the past 20 validation periods, smaller numbers can be used here e.g., 5 or 10.
    hypar["early_stop"] = 20

    ## valid and save model weights every 2000 iterations
    hypar["model_save_fre"] = 2000

    ## batch size for training
    hypar["batch_size_train"] = 8

    ## batch size for validation and inferencing
    hypar["batch_size_valid"] = 1

    ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_ite"] = 10000000

    ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    hypar["max_epoch_num"] = 1000000

    main(train_datasets, valid_datasets, hypar=hypar)
