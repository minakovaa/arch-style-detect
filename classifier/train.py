import os

import torch
from torch import nn
from torchvision import datasets, models, transforms
from sklearn import metrics
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


CHKPT_STATE_DICT = 'model_state_dict'
CHKPT_EPOCH = 'epoch'
CHKPT_OPTIM_STATE = 'optimizer_state_dict'
CHKPT_LOSS_HISTORY = 'loss_history'
CHKPT_ACC_HISTORY = 'acc_history'
CHKPT_CLASS_NAMES = 'class_names'
CHKPT_LR = 'lr'
CHKPT_WEIGHT_DECAY = 'weight_decay'

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def split_train_test_val():
    pass

def create_dataloader(
    data_dir, 
    name='test', 
    train_batch_size=1, 
    train_img_size=224, 
    data_transforms: transforms.Compose=None,
    num_workers=0,
    ):
    """
    @param name: This is the name continue 'data_dir' path. Usually it is 'train', 'val' or 'test'.
                 For example for 'train': $data_dir$/train

    @param batch_size: Defined only fot 'train' becouse images different sizes

    @param train_img_size: Defined only fot 'train'

    @param data_transforms: Should be torchvision.transforms.Compose(). 
                            By default only convert to torch tensor and normalize
    """
    if name not in ['train', 'val', 'test']:
        raise ValueError("'name' should be only 'train', 'val' or 'test'")

    NORMALIZE_TRANSFORM = transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    
    if name == 'train':
        data_transforms = transforms.Compose([
            # transforms.RandomPerspective(distortion_scale=0.1),
            transforms.RandomAffine(degrees=3,),
            transforms.ColorJitter(brightness=0.25,  #[max(0, 1 - brightness), 1 + brightness] 
                                   contrast=0.25, #[max(0, 1 - contrast), 1 + contrast] 
                                   saturation=0.5, #[max(0, 1 - saturation), 1 + saturation]
                                   hue=0.01 #(-hue, hue) 0 <= hue <= 0.5.
                                   ),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomResizedCrop(train_img_size), #, scale=(1., 1.), ratio=(1., 1.)
            # transforms.Resize(400), transforms.CenterCrop(train_img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORMALIZE_TRANSFORM
            ])
        image_dataset = datasets.ImageFolder(os.path.join(data_dir, name), data_transforms)
            
        dataloader = torch.utils.data.DataLoader(image_dataset, 
                                                 batch_size=train_batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers, 
                                                 # sampler = ImbalancedDatasetSampler(train_dataset), # from torchsampler import ImbalancedDatasetSampler
                                                 )
    
    else: # For 'val' or 'test' make dataloaders with batch = 1 becouse images different sizes
        if data_transforms is None:
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                NORMALIZE_TRANSFORM
                ])

        image_dataset = datasets.ImageFolder(os.path.join(data_dir, name), data_transforms)
        dataloader = torch.utils.data.DataLoader(image_dataset, 
                                                 batch_size=1, 
                                                 shuffle=False, 
                                                 num_workers=num_workers)
    return dataloader


def model_load():
    pass


def grid_search_model_params():
    pass


def train_model():
    """ Train model with fixed params and save checkpoints
        every `every_k_epoch_saved_ckpt` epochs
    """
    pass


def imshow(inp, title=None, is_inp_numpy=False, is_efficientnet_advprop=False):
    """Imshow for Tensor."""
    if not is_inp_numpy: # if torch tensor
        inp = inp.numpy().transpose((1, 2, 0))

    if is_efficientnet_advprop:
        inp = (inp + 1.0) / 2.0
    else:
        mean = np.array(NORMALIZE_MEAN)
        std = np.array(NORMALIZE_STD)
        inp = std * inp + mean
        
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def test_model(model, class_names, test_folder_path, show_mistakes=False, device='cpu', is_return_top_3_acc=True):
    """ Test model on test set and return metrics"""
    acc = 0.0

    pred_list = []
    top3_acc_list = [] # 1 - true in top3, 0 - not
    true_list = []
    
    dataloader = create_dataloader(data_dir=test_folder_path, name='test')

    model = model.to(device)
    model.eval()

    with torch.no_grad():
      # Iterate over data.
      for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if hasattr(outputs, 'logits'):
                outputs = outputs.logits # !!!!!!! only for inception_v3

            _, preds = torch.max(outputs, 1)

            acc += accuracy_score(y_true=labels.cpu(),
                                  y_pred=preds.cpu(),
                                  normalize=False)

            pred_list.extend(preds.cpu().tolist())
            true_list.extend(labels.cpu().tolist())

                
            if is_return_top_3_acc:
                for j in range(inputs.size()[0]):
                    pr = softmax(outputs[j].detach().cpu().numpy())

                    sorted_ind_by_proba = pr.argsort()
                    top_3_ind = sorted_ind_by_proba[-3:][::-1]
                    
                    true_class = labels.cpu()[j].item()

                    is_in_top3 = 1 if true_class in top_3_ind else 0
                    top3_acc_list.append(is_in_top3)

            if show_mistakes and top3_acc_list[-1] == 0:
                for j in range(inputs.size()[0]):
                      if preds[j] != labels.cpu()[j]:

                        pr = softmax(outputs[j].detach().cpu().numpy())
                        sorted_ind_by_proba = pr.argsort()
                        top_3_ind = sorted_ind_by_proba[-3:][::-1]
                        other_ind = sorted_ind_by_proba[:-3][::-1]  # Indexes of remaining probabilities

                        top_3_styles_probability = {class_names[i]: pr[i] for i in top_3_ind}
                        top_3_styles_probability.update({'Остальные': np.sum(pr[other_ind])})

                        print('predicted: {}'.format(class_names[preds[j]]) + 
                            f' real: {class_names[labels.cpu()[j]]}')
                        print(top_3_styles_probability)

                        imshow(inputs.cpu().data[j])
            
            # del outputs
            # del preds
            # del labels
            # del inputs
            
            # torch.cuda.empty_cache()

    acc = acc / len(dataloader.dataset) # dataset_sizes['test'] # Normilize accuracy
    
    top3_acc = sum(top3_acc_list) / len(top3_acc_list)

    print(f'Accuracy on test set = {acc}')
    print(f'Top3 acc on test set = {top3_acc}')

    true_class_names = [class_names[label] for label in true_list]
    pred_class_names = [class_names[label] for label in pred_list]

    # Printing the confusion matrix
    # The columns will show the instances predicted for each label,
    # and the rows will show the actual number of instances for each label.
    print(metrics.confusion_matrix(true_class_names, pred_class_names, labels=class_names))
    # Printing the precision and recall, among other metrics
    print(metrics.classification_report(true_class_names, pred_class_names, labels=class_names))

    return  pred_list, true_list, top3_acc_list


def save_to_checkpoint(model, path_to_chkpt, epoch, acc_history, loss_history, class_names, lr, weight_decay):
    torch.save({
        CHKPT_EPOCH: epoch,
        CHKPT_STATE_DICT: model.state_dict(),
        #CHKPT_OPTIM_STATE: optimizer.state_dict(),
        CHKPT_ACC_HISTORY: acc_history,
        CHKPT_LOSS_HISTORY: loss_history,
        CHKPT_CLASS_NAMES: class_names,
        CHKPT_LR: lr,
        CHKPT_WEIGHT_DECAY: weight_decay
    }, path_to_chkpt)


def load_from_checkpoint(path_to_chkpt, model_name=None, model_loaded=None, device='cpu'):
    """Need to define name 'model_name' or model 'model_loaded'"""
    checkpoint = torch.load(path_to_chkpt, map_location=device)
    # optimizer.load_state_dict(checkpoint[CHKPT_OPTIM_STATE])
    num_epoch = checkpoint[CHKPT_EPOCH]
    loss_history = checkpoint[CHKPT_LOSS_HISTORY]
    acc_history = checkpoint[CHKPT_ACC_HISTORY]
    class_names = checkpoint[CHKPT_CLASS_NAMES]
    lr = checkpoint[CHKPT_LR]
    weight_decay = checkpoint[CHKPT_WEIGHT_DECAY]

    num_classes = len(class_names)

    if model_loaded is None:
        if model_name == 'resnet18':
            model_loaded = models.resnet18(pretrained=False)
        elif model_name == 'resnet34':
            model_loaded = models.resnet34(pretrained=False)
        elif model_name == 'resnet50':
            model_loaded = models.resnet50(pretrained=False)
        elif model_name == 'resnet152':
            model_loaded = models.resnet152(pretrained=False)
       
        if model_name is not None:
            num_ftrs = model_loaded.fc.in_features
            model_loaded.fc = nn.Linear(num_ftrs, num_classes, bias=True)

    if model_loaded is not None:
        model_loaded = model_loaded.to(device)
        model_loaded.load_state_dict(checkpoint[CHKPT_STATE_DICT])

    return model_loaded, {
        CHKPT_EPOCH: num_epoch,
        #CHKPT_OPTIM_STATE: optimizer.state_dict(),
        CHKPT_ACC_HISTORY: acc_history,
        CHKPT_LOSS_HISTORY: loss_history,
        CHKPT_CLASS_NAMES: class_names,
        CHKPT_LR: lr,
        CHKPT_WEIGHT_DECAY: weight_decay
    }