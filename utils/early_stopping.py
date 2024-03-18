import os
import torch


class EarlyStopping:
  """Early stops the training if validation loss doesn't improve after a given patience."""
  def __init__(self, patience=10, delta=0.00001, checkpoint_file_path=None):
    self.patience = patience
    self.counter = 0
    self.delta = delta

    self.val_loss_min = None
    
    """
    self.file_path = os.path.join(
      self.checkpoint_file_path, f"chkpt_{run_time_str}.pt"
    )
    self.latest_file_path = os.path.join(
      self.checkpoint_file_path, f"chkpt_latest.pt"
    )
    """

  def check_and_save(self, new_validation_loss, model, epoch):
    early_stop = False
    num = 0

    if self.val_loss_min is None:
      self.val_loss_min = new_validation_loss
      message = f'Early stopping is stated!'
      num = 1
    elif new_validation_loss < self.val_loss_min - self.delta:
      message = f'V_loss decreased ({self.val_loss_min:7.5f} --> {new_validation_loss:7.5f}). Saving model...'
      self.save_checkpoint(new_validation_loss, model, epoch)
      self.val_loss_min = new_validation_loss
      self.counter = 0
      num = 2
    else:
      self.counter += 1
      message = f'Early stopping counter: {self.counter} out of {self.patience}'
      if self.counter >= self.patience:
        early_stop = True
        message += " *** TRAIN EARLY STOPPED! ***"
      num = 3

    return message, early_stop, num

  def save_checkpoint(self, val_loss, model,epoch):
    '''Saves model when validation loss decrease.'''
    #torch.save(model.state_dict(), self.file_path)
    #torch.save(model.state_dict(), self.latest_file_path)
    self.val_loss_min = val_loss
