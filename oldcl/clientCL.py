from utils.ParamFns import set_parameters, get_parameters
from utils.TrainTestFns import train, test
from models.SimpleCNN import Net
from workloads.CIFAR10CL import load_datasets 
from clutils.make_experiences import split_dataset
from clutils.clstrat import make_cl_strat 
from clutils.create_benchmarks import make_benchmark

import json

from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.utils import as_avalanche_dataset

import flwr
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLIENTS = 5

class FlowerClient(NumPyClient):
    def __init__(self, net, benchmark, trainloader_len, testloader_len):
        self.net = net
        self.benchmark = benchmark
        self.trainloader_len = trainloader_len
        self.testloader_len = testloader_len
        self.cl_strategy, self.evaluation = make_cl_strat(net) 

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
            # TRAINING LOOP
        print("for client config:")
        print(config)
        print('Starting experiment...')
        results_stream_dict = []
        for experience in self.benchmark.train_stream:
            print("Start of experience: ", experience.current_experience)

            # train returns a dictionary which contains all the metric values

            res = self.cl_strategy.train(experience)
            print('Training completed: ', experience.current_experience)
           ## train(self.net, self.trainloader, epochs=1)
            print("Computing Accuracy on Test for exp:", experience.current_experience)
            results_stream_dict.append(self.cl_strategy.eval(self.benchmark.test_stream))

        print('--------------------------------RES_TRAIN----------------------------------')
        print(res)
        print('-------------------------------RES_STREAM_EVAL-----------------------------')
        print(results_stream_dict)
#        print('-------------------------------GET_ALL_METRICS-----------------------------')
#        results_get_all = self.evaluation.get_all_metrics()
#        print(results_get_all)
        print('---------------------------------LAST_METRICS------------------------------')
        results_stream = self.evaluation.get_last_metrics()
        print(results_stream)
        confusion_matrix = results_stream["ConfusionMatrix_Stream/eval_phase/test_stream"]
        forgetting_measure = results_stream["StreamForgetting/eval_phase/test_stream"]
        stream_loss = results_stream["Loss_Stream/eval_phase/test_stream"]
        stream_acc = results_stream["Top1_Acc_Stream/eval_phase/test_stream"]
        stream_disc_usage = results_stream["DiskUsage_Stream/eval_phase/test_stream"]
        fit_dict_return = {
#                "confusion_matrix": confusion_matrix.tolist(),
                "forgetting_measure":  float(forgetting_measure),
                "stream_loss":  float(stream_loss),
                "stream_acc":  float(stream_acc),
                "stream_disc_usage":  float(stream_disc_usage)
            }
        print("----------------------------CLIENT_INFO--------------------------------")
        print(fit_dict_return)
        print('-----------------------------------------------------------------------')
        return get_parameters(self.net), self.trainloader_len, fit_dict_return

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        cl_strategy, evaluation = make_cl_strat(self.net)
        results = []
        print("------------------------Evaluating Client for Server on Updated Global Model on Test Set--------------------")
        results.append(cl_strategy.eval(self.benchmark.test_stream))
##        loss, accuracy = test(self.net, self.valloader)
        last_metrics = evaluation.get_last_metrics()
        loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        accuracy = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]
        print("Results of Eval for GCF----------------------------------------------------------------")
        print(results)
        exp_acc = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    exp_acc.append(float(acc))
                    


        print("Eval of Client: ")
        print("Loss: ", loss)
        print("Acc: ", accuracy)
        print("Per Exp Acc: ", exp_acc)

        return float(loss), self.testloader_len, {"accuracy": float(accuracy), "loss": float(loss), "ExpAccuracy": json.dumps(exp_acc)}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    train_data, test_data = load_datasets(partition_id=partition_id)
#    print(type(train_data))
    total_train_samples = len(train_data)
    total_eval_samples = len(test_data)
    trainloader_len = total_train_samples//BATCH_SIZE
    testloader_len = total_eval_samples//BATCH_SIZE

    # Splitting Data into Experiences

    n_experiences = cfg.cl.num_experiences
    train_experiences = split_dataset(train_data, n_experiences)
    test_experiences = split_dataset(test_data, n_experiences)  # optional
    ava_train = []
    ava_test = []
#    ava_test.append(as_avalanche_dataset(test_data))
    for exp in train_experiences:
        ava_exp = as_avalanche_dataset(exp)
#        print(type(ava_exp))
        ava_train.append(exp)

#    Provision for Splitting Test Stream 

    for exp in test_experiences:
        ava_exp = as_avalanche_dataset(exp)
        ava_test.append(exp)

    # Creating Bencmarks -> using Entire testdata for eval -> have to check difference

    benchmark = benchmark_from_datasets(train=ava_train, test=ava_test)
    bm = make_benchmark(as_avalanche_dataset(train_data), as_avalanche_dataset(test_data)) 

    # Print ClientID

    print("------------------------------------------------ClientID: ", partition_id, "----------------------------------------------")


    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, bm, trainloader_len, testloader_len).to_client()


