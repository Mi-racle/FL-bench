from argparse import Namespace, ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from typing import Type, Any

from src.client.fednew import FedNewClient
from src.server.fedavg import FedAvgServer, get_fedavg_argparser
from src.utils.tools import trainable_params


def get_fednew_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument('--lambda', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=1.0)
    return parser


class FedNewServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedNew",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fednew_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        random_init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )
        self.cluster_params_dict = OrderedDict[
            zip(self.trainable_params_name, random_init_params)
        ]
        self.trainer = FedNewClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
                self.client_metrics[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)


    def generate_cluster_params(self) -> Type[OrderedDict[Any]]:
        """
        This function is for outputting model parameters that asked by `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            OrderedDict[str, torch.Tensor]: The trainable model parameters.
        """
        return self.cluster_params_dict
