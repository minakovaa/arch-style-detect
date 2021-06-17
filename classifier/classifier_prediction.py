import torch
import torch.nn as nn

from torchvision import models, transforms
import numpy as np
from scipy.special import softmax


def load_checkpoint(checkpoint_path=None, device=None):
    if checkpoint_path is None:
        checkpoint_path = "classifier/checkpoints/model_arch_test_9375.pt"

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # num_classes = checkpoint['num_classes']
    num_classes = 4
    # class_names = checkpoint['class_names']
    class_names = ['барокко', 'классицизм', 'русское_барокко', 'узорочье']

    model_loaded = models.resnet18(pretrained=False)  # models.resnet152(pretrained=False)
    num_ftrs = model_loaded.fc.in_features
    model_loaded.fc = nn.Linear(num_ftrs, num_classes)

    model_loaded = model_loaded.to(device)

    model_loaded.load_state_dict(checkpoint['model_state_dict'])

    return model_loaded, class_names


def classifier_predict(model, input_img, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    transform_evaluate = transforms.Compose([
        transforms.Resize(400), transforms.CenterCrop(224),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tensor_img = transform_evaluate(input_img)
    inputs = tensor_img.to(device)
    inputs = torch.unsqueeze(inputs, 0)  # one image as batch
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1) # top_1 predicted class

    return preds, outputs.squeeze().detach().numpy()


def arch_style_predict_by_image(img):
    model_loaded, class_names = load_checkpoint()
    preds, outputs = classifier_predict(model=model_loaded, input_img=img)

    probabilities = softmax(outputs)  # From output CrossEntropy obtain probabilities

    sorted_ind_by_proba = probabilities.argsort()
    top_3_ind = sorted_ind_by_proba[-3:][::-1]
    other_ind = sorted_ind_by_proba[:-3][::-1]  # Indexes of remaining probabilities

    top_3_styles_probability = {class_names[i]: probabilities[i] for i in top_3_ind}
    top_3_styles_probability.update({'Остальные': np.sum(probabilities[other_ind])})

    return class_names[preds], top_3_styles_probability
