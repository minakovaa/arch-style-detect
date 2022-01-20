import torch

CHKPT_STATE_DICT = 'model_state_dict'
CHKPT_EPOCH = 'epoch'
CHKPT_LOSS_HISTORY = 'loss_history'
CHKPT_ACC_HISTORY = 'acc_history'
CHKPT_CLASS_NAMES = 'class_names'
CHKPT_LR = 'lr'
CHKPT_WEIGHT_DECAY = 'weight_decay'


def split_train_test_val():
    pass


def create_dataloaders():
    pass


def model_load():
    pass


def grid_search_model_params():
    pass


def train_model():
    """ Train model with fixed params and save checkpoints
        every `every_k_epoch_saved_ckpt` epochs
    """
    pass


def test_model():
    """ Test model on test set and return metrics"""
    pass


def save_to_checkpoint(model, path_to_chkpt, epoch, acc_history, loss_history, class_names, lr, weight_decay):
    torch.save({
        CHKPT_EPOCH: epoch,
        CHKPT_STATE_DICT: model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        CHKPT_ACC_HISTORY: acc_history,
        CHKPT_LOSS_HISTORY: loss_history,
        CHKPT_CLASS_NAMES: class_names,
        CHKPT_LR: lr,
        CHKPT_WEIGHT_DECAY: weight_decay
    }, path_to_chkpt)


def load_from_checkpoint(path_to_chkpt, model_loaded=None, device='cpu'):
    checkpoint = torch.load(path_to_chkpt, map_location=device)

    if model_loaded is not None:
        model_loaded = model_loaded.to(device)
        model_loaded.load_state_dict(checkpoint[CHKPT_STATE_DICT])

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    num_epoch = checkpoint[CHKPT_EPOCH]
    loss_history = checkpoint[CHKPT_LOSS_HISTORY]
    acc_history = checkpoint[CHKPT_ACC_HISTORY]
    class_names = checkpoint[CHKPT_CLASS_NAMES]
    lr = checkpoint[CHKPT_LR]
    weight_decay = checkpoint[CHKPT_WEIGHT_DECAY]

    return model_loaded, {
        CHKPT_EPOCH: num_epoch,
        #'optimizer_state_dict': optimizer.state_dict(),
        CHKPT_ACC_HISTORY: acc_history,
        CHKPT_LOSS_HISTORY: loss_history,
        CHKPT_CLASS_NAMES: class_names,
        CHKPT_LR: lr,
        CHKPT_WEIGHT_DECAY: weight_decay
    }