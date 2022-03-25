import os
import psutil
import logging

import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.special import softmax
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

        if model_name == 'resnet50':
            checkpoint_path = "classifier/checkpoints_prod/resnet50_batch_16_imgsize_600_SGD.pt"

        elif model_name == 'resnet152':
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
        num_ftrs = model_loaded.fc.in_features
        model_loaded.fc = nn.Linear(num_ftrs, num_classes)

    if model_name == 'resnet50':
        model_loaded = models.resnet50(pretrained=False)
        num_ftrs = model_loaded.fc.in_features
        model_loaded.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet152':
        model_loaded = models.resnet152(pretrained=False)
        num_ftrs = model_loaded.fc.in_features
        model_loaded.fc = nn.Linear(num_ftrs, num_classes)

    # elif model_name == 'efficientnet-b5':
    #     model_loaded = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    #
    # elif model_name == 'efficientnet-b6':
    #     model_loaded = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)

    model_loaded = model_loaded.to(device)

    model_loaded.load_state_dict(checkpoint['model_state_dict'])

    return model_loaded, class_names


def classifier_predict(model, input_img, device=None, is_debug=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

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

    tensor_img = transform_evaluate(input_img)
    inputs = tensor_img.to(device)
    inputs = torch.unsqueeze(inputs, 0)  # make one batch with one image
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)  # top_1 predicted class

    if is_debug:
        check_memory('classifier_predict_finish')

    return preds, outputs.squeeze().detach().numpy()


def arch_style_predict_by_image(img, model, class_names, is_debug=False):
    preds, outputs = classifier_predict(model=model, input_img=img, is_debug=is_debug)
    probabilities = softmax(outputs)  # From output CrossEntropy obtain probabilities

    sorted_ind_by_proba = probabilities.argsort()
    top_3_ind = sorted_ind_by_proba[-3:][::-1]
    # other_ind = sorted_ind_by_proba[:-3][::-1]  # Indexes of remaining probabilities

    top_3_styles_probability = {class_names[i]: f"{probabilities[i]:.02f}" for i in top_3_ind}
    top_3_styles_probability.update({
        CLASS_REMAIN: str(abs(round(1.0 - sum(map(float, top_3_styles_probability.values())), 2)))
    })

    return top_3_styles_probability
