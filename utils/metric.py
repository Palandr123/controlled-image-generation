class LossMetric:
    """
    Keep track of the loss over an epoch
    :param self.running_loss - total loss
    :param self.count - amount of processed items
    """
    def __init__(self) -> None:
        self.running_loss = 0
        self.count = 0

    def update(self, loss: float, batch_size: int) -> None:
        """
        update the loss
        :param loss: new obtained loss
        :param batch_size: batch size
        :return: None
        """
        self.running_loss += loss * batch_size
        self.count += batch_size

    def compute(self) -> float:
        """"
        :return:compute the average loss
        """
        return self.running_loss / self.count

    def reset(self) -> None:
        """
        reset the loss
        :return: None
        """
        self.running_loss = 0
        self.count = 0
