import os
import gc
import psutil
from collections import Counter

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from scipy.special import softmax
from efficientnet_pytorch import EfficientNet

advprop = False  # For models using advprop pretrained weights different normalization


def check_memory(info_str, logger):
    p = psutil.Process(os.getpid())
    mem_usage = p.memory_info().rss / 1024 / 1024
    # print(f"{info_str}: {mem_usage} MB")
    logger.debug("%s: %s MB", info_str, mem_usage)


def load_checkpoint(checkpoint_path=None, device=None, model_name='efficientnet-b5'):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global advprop

    if checkpoint_path is None:
        if model_name == 'resnet18':
            checkpoint_path = "classifier/checkpoints/model_arch_test_new_50_epoch.pt"  # test acc voited = 0.9217

        elif model_name == 'resnet152':
            checkpoint_path = "classifier/checkpoints/model_resnet152_gray_0_5_num_1.pt"

        elif model_name == 'efficientnet-b5':
            advprop = True
            checkpoint_path = 'classifier/checkpoints/model_advprop_efficientnet-b5_num_1.pt'

        elif model_name == 'efficientnet-b6':
            checkpoint_path = 'classifier/checkpoints/model_efficientnet-b6_num_1.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model_loaded = None

    if model_name == 'resnet18':
        model_loaded = models.resnet18(pretrained=False)  # resnet18  # resnet152 # wide_resnet101_2
        num_ftrs = model_loaded.fc.in_features
        model_loaded.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet152':
        model_loaded = models.resnet152(pretrained=False)  # resnet18  # resnet152 # wide_resnet101_2
        num_ftrs = model_loaded.fc.in_features
        model_loaded.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet-b5':
        model_loaded = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)

    elif model_name == 'efficientnet-b6':
        model_loaded = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)

    model_loaded = model_loaded.to(device)

    model_loaded.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    return model_loaded, class_names


def classifier_predict(model, input_img, logger, device=None, is_debug=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_debug:
        check_memory('classifier_predict_1', logger)

    model.eval()

    if is_debug:
        check_memory('classifier_predict_2', logger)

    if advprop:  # for models using advprop pretrained weights
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_evaluate = transforms.Compose([
        transforms.ToTensor(),
        normalize  # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tensor_img = transform_evaluate(input_img)

    inputs = tensor_img.to(device)

    inputs = torch.unsqueeze(inputs, 0)  # one image as batch

    outputs = model(inputs)

    _, preds = torch.max(outputs, 1) # top_1 predicted class

    if is_debug:
        check_memory('classifier_predict_3', logger)

    return preds, outputs.squeeze().detach().numpy()


def classifier_predict_voting(model, input_img, logger, num_samples=18, batch_size=6, device=None,
                              is_debug=False):
    """
    num_samples:  How many transformation with one image and voting prediction classes
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_debug:
        check_memory('classifier_predict_voting_1', logger)

    model.eval()

    if advprop:  # for models using advprop pretrained weights
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_for_voting = transforms.Compose([
        transforms.ToTensor(),
        normalize  # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preds = np.array([])
    outputs = None
    for i_batch in range(num_samples // batch_size):
        # Via batches 'num_samples' of original images transformed by 'transform_for_voting'
        batch_imgs = torch.stack([transform_for_voting(input_img) for _ in range(batch_size)])

        inputs = batch_imgs.to(device)

        outputs_batch = model(inputs)

        if is_debug:
            check_memory(f'classifier_predict_voting_batch_{i_batch}', logger)

        _, preds_batch = torch.max(outputs_batch, 1)  # top_1 predicted class

        if device == torch.device("cpu"):
            preds_batch, outputs_batch = preds_batch.detach().numpy(), outputs_batch.detach().numpy()
        else:
            preds_batch, outputs_batch = preds_batch.cpu().detach().numpy(), outputs_batch.cpu().detach().numpy()

        preds = np.append(preds, preds_batch)

        if outputs is None:
            outputs = outputs_batch
        else:
            outputs = np.vstack((outputs, outputs_batch))

    if is_debug:
        check_memory(f'classifier_predict_voting_2', logger)

    # Voting by counting top class
    vote_preds = Counter(preds)
    win_class = max(vote_preds, key=vote_preds.get)

    wieght_outputs = outputs.sum(axis=0) / num_samples

    return win_class, wieght_outputs


def arch_style_predict_by_image(img, model, class_names, logger,
                                samples_for_voting=None, batch_size_voting=None,
                                is_debug=False):
    preds = None
    outputs = None

    if samples_for_voting is None:
        preds, outputs = classifier_predict(model=model, input_img=img, is_debug=is_debug)
    else:
        preds, outputs = classifier_predict_voting(model=model, input_img=img,
                                                   logger=logger,
                                                   num_samples=samples_for_voting,
                                                   batch_size=batch_size_voting,
                                                   is_debug=is_debug)

    probabilities = softmax(outputs)  # From output CrossEntropy obtain probabilities

    sorted_ind_by_proba = probabilities.argsort()
    top_3_ind = sorted_ind_by_proba[-3:][::-1]
    # other_ind = sorted_ind_by_proba[:-3][::-1]  # Indexes of remaining probabilities

    top_3_styles_probability = {class_names[i]: round(probabilities[i], 3) for i in top_3_ind}
    top_3_styles_probability.update({'Остальные': abs(round(1.0 - sum(top_3_styles_probability.values()), 3))})

    return top_3_styles_probability
