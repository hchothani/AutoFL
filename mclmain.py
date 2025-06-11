import torch
import flwr
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp

from omegaconf import OmegaConf
import warnings

from testmclientCL import client_fn
from mclserver import server_fn

# Ignore Depcreation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load Config
cfg = OmegaConf.load('config/config.yaml')
print(OmegaConf.to_yaml(cfg))

def main():
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    backend_config = {
            "client_resources": {
                "num_cpus": cfg.client.num_cpus,
                "num_gpus": cfg.client.num_gpus,
                }
            }

    # Run Simluation

    print("Running Simulation")

    run_simulation(
        server_app = server,
        client_app = client,
        num_supernodes = cfg.server.num_clients,
        backend_config = backend_config,
    )

if __name__ == "__main__":
        main()
