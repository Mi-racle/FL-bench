from argparse import Namespace, ArgumentParser
from copy import deepcopy

from src.client.fednew import FedNewClient
from src.server.fedavg import FedAvgServer, get_fedavg_argparser


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
        self.trainer = FedNewClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )