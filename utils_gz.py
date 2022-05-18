import torch


def adjust_learning_rate(optimizer, scale):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
        print(f"DECAYING learning rate.\n The new LR is {param_group['lr']}\n")


def save_checkpoint(epoch, model, optimizer, filename, total_train_loss, total_valid_loss):
    """
    Save model checkpoint.
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'train_loss': total_train_loss,
             'valid_loss': total_valid_loss}

    torch.save(state, filename)
