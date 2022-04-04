import os
import psutil
import logging
import io
from collections import Counter

import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.special import softmax
from PIL import Image

# from efficientnet_pytorch import EfficientNet

CLASS_REMAIN = 'Остальные'

advprop = False  # For models using advprop pretrained weights different normalization

logger = logging.getLogger("classifier_prediction")


def check_memory(info_str):
    p = psutil.Process(os.getpid())
    mem_usage = p.memory_info().rss / 1024 / 1024
    # print(f"{info_str}: {mem_usage} MB")
    logger.debug("%s: %s MB", info_str, mem_usage)


def load_checkpoint(checkpoint_path=None, device=None, model_name='resnet18'):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global advprop

    if checkpoint_path is None:
        if model_name == 'resnet18':
            checkpoint_path = "classifier/checkpoints_prod/resnet_18_b_32_img_500_Adam_sched_wd_-07_lr_005.pt"

        if model_name == 'resnet34':
            checkpoint_path = "classifier/checkpoints_prod/full_data_resnet_34_b_64_img_500_Adam_wd_-07_lr_0003-40-epoch.pt"

        if model_name == 'resnet50':
            checkpoint_path = "classifier/checkpoints_prod/full_data_resnet_50_b_64_img_500_Adam_wd_-07_lr_0003-30-epoch.pt"

        if model_name == 'resnet152':
            checkpoint_path = "classifier/checkpoints_prod/model_resnet152_gray_0_5_num_1.pt"

        # elif model_name == 'efficientnet-b5':
        #     advprop = True
        #     checkpoint_path = 'classifier/checkpoints/model_advprop_efficientnet-b5_num_1.pt'
        #
        # elif model_name == 'efficientnet-b6':
        #     checkpoint_path = 'classifier/checkpoints/model_efficientnet-b6_num_1.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model_loaded = None

    if model_name == 'resnet18':
        model_loaded = models.resnet18(pretrained=False)

    if model_name == 'resnet34':
        model_loaded = models.resnet34(pretrained=False)

    if model_name == 'resnet50':
        model_loaded = models.resnet50(pretrained=False)

    if model_name == 'resnet152':
        model_loaded = models.resnet152(pretrained=False)

    # elif model_name == 'efficientnet-b5':
    #     model_loaded = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    #
    # elif model_name == 'efficientnet-b6':
    #     model_loaded = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)

    num_ftrs = model_loaded.fc.in_features  # For resnet
    model_loaded.fc = nn.Linear(num_ftrs, num_classes) # For resnet

    model_loaded = model_loaded.to(device)
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.eval()

    return model_loaded, class_names


def classifier_predict(model, input_img, device=None, is_debug=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_debug:
        check_memory('classifier_predict_start')

    if advprop:  # for models using advprop pretrained weights
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_evaluate = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    with torch.no_grad():
        tensor_img = transform_evaluate(input_img)
        inputs = tensor_img  # .to(device)
        inputs = torch.unsqueeze(inputs, 0)  # make one batch with one image
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # top_1 predicted class

    if is_debug:
        check_memory('classifier_predict_finish')

    return preds, outputs.squeeze().numpy()


def classifier_predict_voting(model, input_img, device=None, is_debug=False):
    """
    Voting FIVE CROPS
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_debug:
        check_memory('classifier_predict_voting_start')

    if advprop:  # for models using advprop pretrained weights
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    NUM_CROPS = 5

    crop_width, crop_height = input_img.size
    crop_width, crop_height = crop_width * 3 // 4, crop_height * 3 // 4

    five_crops_transform = transforms.FiveCrop((crop_width, crop_height))

    transform_evaluate = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    five_crops = five_crops_transform(input_img)  # return 5 crops: top_left,top_right,bottom_left,bottom_right,center
    batch_imgs = torch.stack([transform_evaluate(crop) for crop in five_crops])

    inputs = batch_imgs.to(device)
    with torch.no_grad():
        outputs = model(inputs)

    _, preds = torch.max(outputs, 1)  # top_1 predicted class
    if device == torch.device("cpu"):
        preds, outputs = preds.numpy(), outputs.numpy()
    else:
        preds, outputs = preds.cpu().numpy(), outputs.cpu().numpy()
    # By voting predict class
    vote_preds = Counter(preds)
    win_class = max(vote_preds, key=vote_preds.get)
    wieght_outputs = outputs.sum(axis=0) / NUM_CROPS

    if is_debug:
        check_memory('classifier_predict_voting_finish')

    return win_class, wieght_outputs


def arch_style_predict_by_image(img, model, class_names, is_debug=False, is_five_crop_voting=False):

    if not is_five_crop_voting:
        preds, outputs = classifier_predict(model=model, input_img=img, is_debug=is_debug)
    else:
        preds, outputs = classifier_predict_voting(model=model, input_img=img, is_debug=is_debug)

    probabilities = softmax(outputs)  # From output Logits obtain probabilities

    sorted_ind_by_proba = probabilities.argsort()
    top_3_ind = sorted_ind_by_proba[-3:][::-1]
    # other_ind = sorted_ind_by_proba[:-3][::-1]  # Indexes of remaining probabilities

    top_3_styles_probability = {class_names[i]: f"{probabilities[i]:.02f}" for i in top_3_ind}
    top_3_styles_probability.update({
        CLASS_REMAIN: str(abs(round(1.0 - sum(map(float, top_3_styles_probability.values())), 2)))
    })

    return top_3_styles_probability


def predict_image_bytes(model, styles, img_bytes, is_five_crop_voting=False):
    if not img_bytes:
        return None

    img = Image.open(io.BytesIO(img_bytes))

    logger.info("Start predict image %sx%s class", img.size[0], img.size[1])
    prediction_top_3_styles_with_proba = arch_style_predict_by_image(img,
                                                                     model=model,
                                                                     class_names=styles,
                                                                     is_debug=True,
                                                                     is_five_crop_voting=is_five_crop_voting)
    logger.info("Finish predict image class")
    logger.info("Predictions: %s ", prediction_top_3_styles_with_proba)

    prediction_top_3_styles_with_proba = {class_name: f"{int(100 * float(proba))}" for class_name, proba
                                          in prediction_top_3_styles_with_proba.items()}

    return prediction_top_3_styles_with_proba
