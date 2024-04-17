from collections import OrderedDict
from typing import Union, Tuple, List, Dict

import torch

from fedavg import FedAvgClient
from src.utils.tools import trainable_params


class FedNewClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super(FedNewClient, self).__init__(model, args, logger, device)

    def fit(self):
        self.model.train()
        self.dataset.train()
        global_params = trainable_params(self.model, detach=True)
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for w, w_t in zip(trainable_params(self.model), global_params):
                    w.grad.data += self.args.mu * (w.data - w_t.data)
                    # w.grad.data += self.args.mu * torch.sin(w.data - w_t.data)
                self.optimizer.step()

    def local_train(
        self,
        client_id: int,
        local_epoch: int,
        local_parameters: OrderedDict[str, torch.Tensor],
        cluster_parameters: OrderedDict[int, OrderedDict[str, torch.Tensor]],
        return_diff=True,
        verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            cluster_parameters (OrderedDict[int, OrderedDict[str, torch.Tensor]]):.

            local_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            `True`: to send the differences between FL model parameters that before and after training;
            `False`: to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(local_parameters)
        eval_results = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_results
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_results,
            )
