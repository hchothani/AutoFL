import wandb
wandb.init(project="autofl-testing", name="test-run")
for i in range(10):
    wandb.log({"accuracy": i/10, "loss": 1-i/10})