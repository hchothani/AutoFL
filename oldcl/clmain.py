import torch
import flwr
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

from clientCL import client_fn
from clserver import server_fn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(DEVICE)


def main():
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    # Run Simluation

    print("running simul")

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=5,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    main()
