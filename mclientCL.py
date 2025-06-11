# Import From Custom Modules
from clutils.ParamFns import set_parameters, get_parameters
from models.SimpleCNN import Net
from workloads.testCIFAR10CL import load_datasets 
from clutils.make_experiences import split_dataset
from clutils.clstrat import make_cl_strat 
from clutils.create_benchmarks import make_benchmark

#Import basic Modules
import json
import  random
import os
import warnings
from omegaconf import OmegaConf

# Avalanche Imports
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.utils import as_avalanche_dataset

# Flower Imports
import flwr
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ConfigRecord

# Ignore Flower Warnings
warnings.filterwarnings("ignore")


#Setting up Configuration
cfg = OmegaConf.load('config/config.yaml')

# Setting Global Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = cfg.dataset.batch_size
NUM_CLIENTS = cfg.server.num_clients
NUM_EXP = cfg.cl.num_experiences

# Enable Green Print
def cprint(text, color="green"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(colors.get(color, colors["reset"]) + text + colors["reset"])


# Persistent State of Clients
partition_strategies = [make_cl_strat(Net().to(DEVICE)) for _ in range(NUM_CLIENTS)]

# Client Class
class FlowerClient(NumPyClient):
    def __init__(self, context: Context, net, benchmark, trainlen_per_exp, testlen_per_exp, partition_id):
        self.client_state = (context.state)
        if "local_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["local_eval_metrics"] = ConfigRecord()
        if "global_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["global_eval_metrics"] = ConfigRecord()
        if "availability" not in self.client_state.config_records:
            self.client_state.config_records["availability"] = ConfigRecord()
        # Special Provision for acc per exp as needed to calculate fm
        if "accuracy_per_exp" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["accuracy_per_exp"] = []
        if "accuracy_per_exp" not in self.client_state.config_records["global_eval_metrics"]:
            self.client_state.config_records["global_eval_metrics"]["accuracy_per_exp"] = []
        if "rounds_selected" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["rounds_selected"] = []
        if "rounds_selected" not in self.client_state.config_records["global_eval_metrics"]:
            self.client_state.config_records["global_eval_metrics"]["rounds_selected"] = []
        self.net = net
        self.benchmark = benchmark
        self.trainlen_per_exp = trainlen_per_exp
        self.testlen_per_exp = testlen_per_exp
        self.cl_strategy, self.evaluation = partition_strategies[partition_id]
        self.partition_id = partition_id

        # To add  later: Battery, Location, Speed, Mobility_Trace

        print(self.client_state.config_records)

    # Get Params from Global Model
    def get_parameters(self, config):
        return get_parameters(self.cl_strategy.model)

    # Fit on Local Data
    def fit(self, parameters, config):
        set_parameters(self.cl_strategy.model, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        cprint("FIT")
        print(f"Client {self.partition_id} Fit on round: {rnd}")

        # Train on Experience as per Round
        cprint("Starting Training")
        results = []
        for i, experience in enumerate(self.benchmark.train_stream, start=1):
            if i == rnd:
                print(f"EXP: {experience.current_experience}")
                trainres = self.cl_strategy.train(experience)
                cprint('Training completed: ')

        # Loal Eval after fit on client for metrics
        print(f"Local Evaluation of client {self.partition_id} on round {rnd}")
        results.append(self.cl_strategy.eval(self.benchmark.test_stream))

        # Calc Accuracy per Experience 
        curr_accpexp = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    curr_accpexp.append(float(acc))
                 

        # Get Local Eval Metrics from Avalanche
        last_metrics = self.evaluation.get_last_metrics()
        confusion_matrix = last_metrics["ConfusionMatrix_Stream/eval_phase/test_stream"].tolist()
        stream_loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        stream_acc = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]
        stream_disc_usage = last_metrics["DiskUsage_Stream/eval_phase/test_stream"]

        # Calculating Forgetting Measures
        local_eval_metrics = self.client_state.config_records["local_eval_metrics"]
        hist_accpexp = local_eval_metrics["accuracy_per_exp"]
        round_fit = local_eval_metrics["rounds_selected"]

        # Calculating Running Cumalative Forgetting Measure
        cm_fmpexp = []
        for i, e in enumerate(hist_accpexp):
            e = json.loads(e)
            fm = e[i] - curr_accpexp[i];
            cm_fmpexp.append(fm)
        if cm_fmpexp:
            cmfm = sum(cm_fmpexp)/len(cm_fmpexp)
        else:
            cmfm = 0

        # Checking Cumalative Forgetting Measure
        cprint("Check Cumalative FM", "blue")
        cprint("History of Accuracy per Experience for this client")
        print(json.dumps(hist_accpexp, indent=2))
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Cumalative Forgetting per Experience: {json.dumps(cm_fmpexp, indent=4)}")
        print(f"Cumalative Forgetting Measure: {cmfm}")
 
        # Calculate Running Stepwise Forgetting Measure
        sw_fmpexp = []
        if hist_accpexp:
            prev_accpexp = json.loads(hist_accpexp[-1])
        else:
            prev_accpexp = []
        for i, (prev_acc, curr_acc) in enumerate(zip(prev_accpexp, curr_accpexp)):
            sw_fmpexp.append(prev_acc - curr_acc)
        swfm = sum(sw_fmpexp)/NUM_EXP

        # Checking Stepwise Forgetting Measure
        cprint("Check StepWise FM", "blue")
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Prev Accuracy per Experience {json.dumps(prev_accpexp, indent=4)}")
        print(f"StepWise Forgetting per Experience: {json.dumps(sw_fmpexp, indent=4)}")
        print(f"StepWise Forgetting Measure: {swfm}")
            
        # Make Fit Metrics Dictionary
        fit_dict_return = {
                "confusion_matrix": json.dumps(confusion_matrix),
                "cumalative_forgetting_measure":  float(cmfm),
                "stepwise_forgetting_measure": float(swfm),
                "stream_loss":  float(stream_loss),
                "stream_acc":  float(stream_acc),
                "stream_disc_usage":  float(stream_disc_usage),
                "accuracy_per_experience": json.dumps(curr_accpexp),
                "stepwise_forgetting_per_exp": json.dumps(sw_fmpexp),
                "cumalative_forgetting_per_exp": json.dumps(cm_fmpexp),
                "pid": self.partition_id,
                "round": rnd,
            }
        cprint("----------------------------Results After Fit--------------------------------")
        print(json.dumps(fit_dict_return, indent=4))
        cprint('-----------------------------------------------------------------------')

        
        # Logging Client State
        cprint("Logging Client States")
        if rnd != 0:
            if "accuracy_per_exp" not in local_eval_metrics:
                local_eval_metrics["accuracy_per_exp"] = [json.dumps(curr_accpexp)]
            else:
                local_eval_metrics["accuracy_per_exp"].append(json.dumps(curr_accpexp))
            if "stream_accuracy" not in local_eval_metrics:
                local_eval_metrics["stream_accuracy"] = [stream_acc]
            else:
                local_eval_metrics["stream_accuracy"].append(stream_acc)
            if "stream_loss" not in local_eval_metrics:
                local_eval_metrics["stream_loss"] = [stream_loss]
            else:
                local_eval_metrics["stream_loss"].append(stream_loss)
            if "cumalative_forgetting_measure" not in local_eval_metrics:
                local_eval_metrics["cumalative_forgetting_measure"] = [cmfm] 
            else:
                local_eval_metrics["cumalative_forgetting_measure"].append(cmfm)
            if "stepwise_forgetting_measure" not in local_eval_metrics:
                local_eval_metrics["stepwise_forgetting_measure"] = [swfm]
            else:
                local_eval_metrics["stepwise_forgetting_measure"].append(swfm)
            local_eval_metrics["rounds_selected"].append(rnd)
            

        
        cprint("Finished Fit")
        # Client Failure Provision
        if random.random() < cfg.client.falloff:
            return None
        else:
            return get_parameters(self.cl_strategy.model), self.trainlen_per_exp[rnd-1], fit_dict_return

    # Evaluate After Updating Global Model
    def evaluate(self, parameters, config):
        # Setting Global Model param
        set_parameters(self.net, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        # Creating a new CL Strategy for Evaluation
        cl_strategy, evaluation = make_cl_strat(self.net)

        # Distributed Client Evaluation
        results = []
        print(f"------------------------Local Client {self.partition_id} Evaluation on Updated Global Model--------------------")
        results.append(cl_strategy.eval(self.benchmark.test_stream))
        last_metrics = evaluation.get_last_metrics()
        stream_loss = last_metrics["Loss_Stream/eval_phase/test_stream"]
        stream_acc = last_metrics["Top1_Acc_Stream/eval_phase/test_stream"]

        # Getting Accuracy per Experience for client
        curr_accpexp = []
        for res in results:
            for exp, acc in res.items():
                if exp.startswith("Top1_Acc_Exp/"):
                    curr_accpexp.append(float(acc))
                    
         # Calculating Forgetting Measures
        global_eval_metrics = self.client_state.config_records["global_eval_metrics"]
        hist_accpexp = global_eval_metrics["accuracy_per_exp"]

        # Calculating Running Cumalative Forgetting Measure
        cm_fmpexp = []
        for i, e in enumerate(hist_accpexp):
            e = json.loads(e)
            fm = e[i] - curr_accpexp[i];
            cm_fmpexp.append(fm)
        if cm_fmpexp:
            cmfm = sum(cm_fmpexp)/len(cm_fmpexp)
        else:
            cmfm = 0

        # Checking Cumalative Forgetting Measure
        cprint("Check Cumalative FM", "blue")
        cprint("History of Accuracy per Experience for this client")
        print(json.dumps(hist_accpexp, indent=2))
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Cumalative Forgetting per Experience: {json.dumps(cm_fmpexp, indent=4)}")
        print(f"Cumalative Forgetting Measure: {cmfm}")
 
        # Calculate Running Stepwise Forgetting Measure
        sw_fmpexp = []
        if hist_accpexp:
            prev_accpexp = json.loads(hist_accpexp[-1])
        else:
            prev_accpexp = []
        for i, (prev_acc, curr_acc) in enumerate(zip(prev_accpexp, curr_accpexp)):
            sw_fmpexp.append(prev_acc - curr_acc)
        swfm = sum(sw_fmpexp)/NUM_EXP

        # Checking Stepwise Forgetting Measure
        cprint("Check StepWise FM", "blue")
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Prev Accuracy per Experience {json.dumps(prev_accpexp, indent=4)}")
        print(f"StepWise Forgetting per Experience: {json.dumps(sw_fmpexp, indent=4)}")
        print(f"StepWise Forgetting Measure: {swfm}")

        eval_dict_return = {
                "stream_accuracy": float(stream_acc),
                "stream_loss": float(stream_loss),
                "accuracy_per_experience": json.dumps(curr_accpexp),
                "stepwise_forgetting_measure": float(swfm),
                "cumalative_forgetting_measure":  float(cmfm),
                "stepwise_forgetting_per_experience": json.dumps(sw_fmpexp),
                "cumalative_forgetting_per_experience": json.dumps(cm_fmpexp),
                "server_round": rnd,
                "pid": self.partition_id,
                }

        # Printing Global Evaluation Results:
        print(f"Global Distributed Evaluation of Client {self.partition_id}")
        print(json.dumps(eval_dict_return, indent=4))

        cprint("Logging Client States")
        if rnd != 0:
            if "accuracy_per_exp" not in global_eval_metrics:
                global_eval_metrics["accuracy_per_exp"] = [json.dumps(curr_accpexp)]
            else:
                global_eval_metrics["accuracy_per_exp"].append(json.dumps(curr_accpexp))
            if "stream_accuracy" not in global_eval_metrics:
                global_eval_metrics["stream_accuracy"] = [stream_acc]
            else:
                global_eval_metrics["stream_accuracy"].append(stream_acc)
            if "stream_loss" not in global_eval_metrics:
                global_eval_metrics["stream_loss"] = [stream_loss]
            else:
                global_eval_metrics["stream_loss"].append(stream_loss)
            if "cumalative_forgetting_measure" not in global_eval_metrics:
                global_eval_metrics["cumalative_forgetting_measure"] = [cmfm] 
            else:
                global_eval_metrics["cumalative_forgetting_measure"].append(cmfm)
            if "stepwise_forgetting_measure" not in global_eval_metrics:
                global_eval_metrics["stepwise_forgetting_measure"] = [swfm]
            else:
                global_eval_metrics["stepwise_forgetting_measure"].append(swfm)
            global_eval_metrics["rounds_selected"].append(rnd)

        return float(stream_loss), sum(self.testlen_per_exp), eval_dict_return


# Function that launches a Client
def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Grab Partition Data
    partition_id = context.node_config["partition-id"]
    train_data, test_data = load_datasets(partition_id=partition_id)

    total_train_samples = len(train_data)
    total_eval_samples = len(test_data)

    # Splitting Data into Experiences
    n_experiences = cfg.cl.num_experiences
    train_experiences = split_dataset(train_data, n_experiences)
    test_experiences = split_dataset(test_data, n_experiences)  # optional


    # Preparing list of train experiences
    trainlen_per_exp = []
    ava_train = []
    for i, exp in enumerate(train_experiences, start=1):
        trainlen_per_exp.append(len(exp))
        ava_exp = as_avalanche_dataset(exp)
        ava_train.append(exp)

    # Preparing list of test experiences
    testlen_per_exp = []
    ava_test = []
    for i, exp in enumerate(test_experiences):
        testlen_per_exp.append(len(exp))
        ava_exp = as_avalanche_dataset(exp)
        ava_test.append(exp)

    # Creating benchmarks 
    benchmark = benchmark_from_datasets(train=ava_train, test=ava_test)

    # Testing Direct BM Creation
    bm = make_benchmark(train_data, test_data)

    # Print ClientID
    print(f"---------------------------------LAUNCHING CLIENT: {partition_id}-----------------------------------------------")

    return FlowerClient(context, net, bm, trainlen_per_exp, testlen_per_exp, partition_id).to_client()


