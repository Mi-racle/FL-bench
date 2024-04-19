from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List, Dict

import torch

from fedavg import FedAvgClient
from src.utils.metrics import Metrics
from src.utils.tools import trainable_params


class FedNewClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super(FedNewClient, self).__init__(model, args, logger, device)
        self.cluster_models = []
        # TOD
        for i in range(2):
            self.cluster_models.append(deepcopy(self.model))

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

    def evaluate_distribution(self):

        for model in self.cluster_models:
            model.eval()
            metrics = Metrics()
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = criterion(logits, y).item()
                pred = torch.argmax(logits, -1)
                metrics.update(Metrics(loss, pred, y))
        return metrics

    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        eval_metrics = {"before": self.evaluate(), "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()}}
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            eval_metrics["after"] = self.evaluate()
        if verbose:
            for split, color, flag, subset in [
                ["train", "yellow", self.args.eval_train, self.trainset],
                ["val", "green", self.args.eval_val, self.valset],
                ["test", "cyan", self.args.eval_test, self.testset],
            ]:
                if len(subset) > 0 and flag:
                    self.logger.log(
                        "client [{}] [{}]({})  loss: {:.4f} -> {:.4f}   accuracy: {:.2f}% -> {:.2f}%".format(
                            self.client_id,
                            color,
                            split,
                            eval_metrics["before"][split].loss,
                            eval_metrics["after"][split].loss,
                            eval_metrics["before"][split].accuracy,
                            eval_metrics["after"][split].accuracy,
                        )
                    )

        return eval_metrics

    def set_cluster_parameters(self, cluster_parameters: List[OrderedDict[str, torch.Tensor]]):
        """Load model parameters received from the server.

        Args:
            cluster_parameters (List[OrderedDict[str, torch.Tensor]]): Parameters of cluster models.
        """
        for i in range(len(cluster_parameters)):
            self.cluster_models[i].load_state_dict(cluster_parameters[i], strict=False)

    def local_train(
        self,
        client_id: int,
        local_epoch: int,
        local_parameters: OrderedDict[str, torch.Tensor],
        return_diff=True,
        verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

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
                local_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_results
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_results,
            )
