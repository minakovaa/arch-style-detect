import torch
import torch.nn as nn

from torchvision import models, transforms
from collections import Counter
from scipy.special import softmax


def load_checkpoint(checkpoint_path=None, device=None):
    if checkpoint_path is None:
        checkpoint_path = "classifier/checkpoints/model_arch_test_new_50_epoch.pt"

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tensor_img = transform_evaluate(input_img)
    inputs = tensor_img.to(device)
    inputs = torch.unsqueeze(inputs, 0)  # one image as batch
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1) # top_1 predicted class

    return preds, outputs.squeeze().detach().numpy()


def classifier_predict_voting(model, input_img, num_samples=5, device=None):
    """
    num_samples:  How many transformation with one image and voting prediction classes
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    transform_for_voting = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    imgs = []
    for i in range(num_samples):
        imgs.append(transform_for_voting(input_img))

    batch_imgs = torch.stack(imgs)

    inputs = batch_imgs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)  # top_1 predicted class

    if device == torch.device("cpu"):
        preds, outputs = preds.detach().numpy(), outputs.detach().numpy()
    else:
        preds, outputs = preds.cpu().detach().numpy(), outputs.cpu().detach().numpy()

    # By voting predict class
    vote_preds = Counter(preds)
    win_class = max(vote_preds, key=vote_preds.get)

    wieght_outputs = outputs.sum(axis=0) / num_samples

    return win_class, wieght_outputs


def arch_style_predict_by_image(img, model, class_names, samples_for_voting=None):
    preds = None
    outputs = None

    if samples_for_voting is None:
        preds, outputs = classifier_predict(model=model, input_img=img)
    else:
        preds, outputs = classifier_predict_voting(model=model, input_img=img, num_samples=samples_for_voting)

    probabilities = softmax(outputs)  # From output CrossEntropy obtain probabilities

    sorted_ind_by_proba = probabilities.argsort()
    top_3_ind = sorted_ind_by_proba[-3:][::-1]
    # other_ind = sorted_ind_by_proba[:-3][::-1]  # Indexes of remaining probabilities

    top_3_styles_probability = {class_names[i]: round(probabilities[i], 3) for i in top_3_ind}
    top_3_styles_probability.update({'Остальные': round(1.0 - sum(top_3_styles_probability.values()), 3)})

    return top_3_styles_probability
