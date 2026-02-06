from __future__ import annotations

import gc
import json
import random
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from flwr.client import NumPyClient
from flwr.common import ConfigRecord, Context

from clutils.ParamFns import get_parameters, set_parameters
from clutils.clstrat import make_cl_strat

from .benchmark_adapter import BenchmarkAdapter

Colors = Dict[str, str]


def cprint(text: str, color: str = "green") -> None:
    colors: Colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    color_code = colors.get(color.lower(), colors["green"])
    print(f"{color_code}{text}{colors['reset']}")


def clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def initialize_partition_strategies(
    model_factory: Callable[[], torch.nn.Module],
    num_clients: int,
) -> List[Tuple[Any, Any]]:
    return [make_cl_strat(model_factory()) for _ in range(num_clients)]


class FlowerClient(NumPyClient):
    def __init__(
        self,
        context: Context,
        net: torch.nn.Module,
        benchmark: Any,
        trainlen_per_exp: Sequence[int] | None,
        testlen_per_exp: Sequence[int] | None,
        partition_id: int,
        strategy_bundle: Tuple[Any, Any],
        latency_simulator: Any,
        cfg: Any,
        experience_count: int,
    ) -> None:
        self.client_state = context.state
        if not hasattr(self.client_state, "config_records"):
            self.client_state.config_records = ConfigRecord()
        if "local_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["local_eval_metrics"] = ConfigRecord()
        if "global_eval_metrics" not in self.client_state.config_records:
            self.client_state.config_records["global_eval_metrics"] = ConfigRecord()
        if "availability" not in self.client_state.config_records:
            self.client_state.config_records["availability"] = ConfigRecord()
        if "accuracy_per_exp" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["accuracy_per_exp"] = []
        if "accuracy_per_exp" not in self.client_state.config_records["global_eval_metrics"]:
            self.client_state.config_records["global_eval_metrics"]["accuracy_per_exp"] = []
        if "rounds_selected" not in self.client_state.config_records["local_eval_metrics"]:
            self.client_state.config_records["local_eval_metrics"]["rounds_selected"] = []
        if "rounds_selected" not in self.client_state.config_records["global_eval_metrics"]:
            self.client_state.config_records["global_eval_metrics"]["rounds_selected"] = []

        self.cfg = cfg
        self.net = net
        self.partition_id = partition_id
        self.latency_sim = latency_simulator
        self.latency_enabled = bool(
            self.latency_sim
            and getattr(self.latency_sim, "enabled", False)
            and self.latency_sim.has_client(partition_id)
        )
        self.permanent_drop = False
        self._payload_bytes: int | None = None

        self.trainlen_per_exp = list(trainlen_per_exp or [])
        self.testlen_per_exp = list(testlen_per_exp or [])
        self.cl_strategy, self.evaluation = strategy_bundle

        self.adapter = BenchmarkAdapter(benchmark)
        self._train_stream = self.adapter.train_stream
        self._test_stream = self.adapter.test_stream
        self._test_stream_name = self.adapter.test_stream_name
        # Trigger warning if configured experiences diverge from benchmark-provided streams.
        self._stream_len_hint = self.adapter.effective_experience_count(
            self.trainlen_per_exp,
            experience_count,
        )
        self.num_experiences_config = experience_count

        cprint(self.client_state.config_records)

    def get_parameters(self, config: Any) -> List[Any]:
        return get_parameters(self.cl_strategy.model)

    def _ensure_payload_bytes(self, parameters: Sequence[Any]) -> int:
        if self._payload_bytes is None:
            try:
                self._payload_bytes = int(sum(arr.nbytes for arr in parameters))  # type: ignore[attr-defined]
            except AttributeError:
                self._payload_bytes = 0
        return self._payload_bytes or 0

    def fit(self, parameters: Sequence[Any], config: dict[str, Any]):
        set_parameters(self.cl_strategy.model, parameters)
        rnd = config["server_round"]
        num_rounds = config["num_rounds"]

        round_start = time.time()

        cprint("FIT")
        print(f"Client {self.partition_id} Fit on round: {rnd}")

        payload_bytes = self._ensure_payload_bytes(parameters)
        latency_sample = None
        simulated_latency_pre = 0.0
        simulated_latency_post = 0.0
        base_delay_s = 0.0
        download_time_s = 0.0
        upload_time_s = 0.0
        expected_network_time_s = 0.0
        threshold_s = float("inf")
        exceeded_threshold = False

        if self.latency_enabled and self.latency_sim:
            latency_sample = self.latency_sim.sample(self.partition_id, rnd, payload_bytes)
            base_delay_s = latency_sample.base_delay_s
            download_time_s = latency_sample.download_time_s
            upload_time_s = latency_sample.upload_time_s
            expected_network_time_s = latency_sample.total_network_time_s
            threshold_s = latency_sample.threshold_s
            exceeded_threshold = latency_sample.exceeded_threshold
            cprint(
                "Time client took: {:.3f}s (threshold {:.3f}s) ".format(
                    expected_network_time_s, threshold_s
                ),
                "blue",
            )
            if exceeded_threshold and not self.permanent_drop:
                cprint(
                    (
                        f"Client {self.partition_id} latency sample {expected_network_time_s:.3f}s exceeds "
                        f"threshold {threshold_s:.3f}s"
                    ),
                    "yellow",
                )
            simulated_latency_pre = self.latency_sim.sleep_pre_training(latency_sample)

        drop_due_to_latency = bool(
            self.permanent_drop or (self.latency_enabled and exceeded_threshold)
        )
        if (
            drop_due_to_latency
            and not self.permanent_drop
            and self.latency_sim
            and self.latency_sim.should_remove_permanently()
        ):
            cprint(
                f"Client {self.partition_id} marked for permanent removal due to latency",
                "red",
            )
            self.permanent_drop = True

        results: List[Any] = []

        experience_count = max(1, self.num_experiences_config)
        experience_idx = (rnd - 1) % experience_count
        print(
            f"Round {rnd}: Training on experience {experience_idx} "
            f"(cycling through {experience_count} available experiences)"
        )

        training_start = time.time()
        training_duration = 0.0
        if not drop_due_to_latency:
            experience = self.adapter.training_experience(experience_idx)
            print(f"EXP: {getattr(experience, 'current_experience', experience_idx)}")
            results.append(self.cl_strategy.train(experience))
            cprint("Training completed: ")
            training_duration = time.time() - training_start
        else:
            cprint(
                f"Skipping training for client {self.partition_id} due to latency threshold",
                "yellow",
            )

        evaluation_duration = 0.0
        if not drop_due_to_latency:
            eval_start = time.time()
            print(f"Local Evaluation of client {self.partition_id} on round {rnd}")
            results.append(self.cl_strategy.eval(self._test_stream))
            evaluation_duration = time.time() - eval_start

        local_eval_metrics = self.client_state.config_records["local_eval_metrics"]

        if drop_due_to_latency:
            curr_accpexp: List[float] = []
            stream_loss = 0.0
            stream_acc = 0.0
            cm_fmpexp: List[float] = []
            sw_fmpexp: List[float] = []
            cmfm = 0.0
            swfm = 0.0
        else:
            curr_accpexp = []
            for res in results:
                if not isinstance(res, dict):
                    continue
                for exp, acc in res.items():
                    if isinstance(exp, str) and exp.startswith("Top1_Acc_Exp/"):
                        curr_accpexp.append(float(acc))

            last_metrics = self.evaluation.get_last_metrics()
            print("DEBUG: Available metrics keys:", list(last_metrics.keys()))

            stream_loss = self.adapter.extract_stream_metric(
                last_metrics, "Loss_Stream", self._test_stream_name
            )
            stream_acc = self.adapter.extract_stream_metric(
                last_metrics, "Top1_Acc_Stream", self._test_stream_name
            )

            hist_accpexp = local_eval_metrics["accuracy_per_exp"]

            cm_fmpexp = []
            num_experiences = max(1, self.num_experiences_config)
            if curr_accpexp:
                for i, entry in enumerate(hist_accpexp):
                    prev_acc = json.loads(entry)
                    idx = i % num_experiences
                    if idx < len(prev_acc) and idx < len(curr_accpexp):
                        cm_fmpexp.append(prev_acc[idx] - curr_accpexp[idx])
            cmfm = sum(cm_fmpexp) / len(cm_fmpexp) if cm_fmpexp else 0.0

            cprint("Check Cumalative FM", "blue")
            cprint("History of Accuracy per Experience for this client")
            print(json.dumps(hist_accpexp, indent=2))
            print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
            print(f"Cumalative Forgetting per Experience: {json.dumps(cm_fmpexp, indent=4)}")
            print(f"Cumalative Forgetting Measure: {cmfm}")

            sw_fmpexp = []
            if hist_accpexp and curr_accpexp:
                prev_accpexp = json.loads(hist_accpexp[-1]) if hist_accpexp else []
                for prev_acc, curr_acc in zip(prev_accpexp, curr_accpexp):
                    sw_fmpexp.append(prev_acc - curr_acc)
            denominator = max(1, self.num_experiences_config)
            swfm = (sum(sw_fmpexp) / denominator) if sw_fmpexp else 0.0

            cprint("Check StepWise FM", "blue")
            print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
            prev_accpexp = json.loads(hist_accpexp[-1]) if hist_accpexp else []
            print(f"Prev Accuracy per Experience {json.dumps(prev_accpexp, indent=4)}")
            print(f"StepWise Forgetting per Experience: {json.dumps(sw_fmpexp, indent=4)}")
            print(f"StepWise Forgetting Measure: {swfm}")

        total_round_time_wall = time.time() - round_start

        network_sleep_after = 0.0
        if latency_sample is not None and not drop_due_to_latency and self.latency_sim:
            simulated_latency_post = self.latency_sim.sleep_post_training(latency_sample)
            network_sleep_after = upload_time_s

        simulated_latency_total = 0.0
        if latency_sample is not None and not drop_due_to_latency:
            simulated_latency_total = simulated_latency_pre + simulated_latency_post

        reported_round_total = total_round_time_wall + simulated_latency_total

        fit_dict_return = {
            "cumalative_forgetting_measure": float(cmfm),
            "stepwise_forgetting_measure": float(swfm),
            "stream_loss": float(stream_loss),
            "stream_acc": float(stream_acc),
            "stream_disc_usage": float(0.0),
            "accuracy_per_experience": json.dumps(curr_accpexp),
            "stepwise_forgetting_per_exp": json.dumps(sw_fmpexp),
            "cumalative_forgetting_per_exp": json.dumps(cm_fmpexp),
            "pid": self.partition_id,
            "round": rnd,
            "latency/enabled": bool(self.latency_enabled),
            "latency/base_delay_s": float(base_delay_s),
            "latency/download_time_s": float(download_time_s),
            "latency/upload_time_s": float(upload_time_s),
            "latency/expected_network_time_s": float(expected_network_time_s),
            "latency/threshold_s": float(threshold_s),
            "latency/dropped": bool(drop_due_to_latency),
            "latency/upload_sleep_s": float(network_sleep_after),
            "timing/training_s": float(training_duration),
            "timing/evaluation_s": float(evaluation_duration),
            "timing/round_wall_clock_s": float(total_round_time_wall),
            "timing/simulated_latency_s": float(simulated_latency_total),
            "timing/round_total_s": float(reported_round_total),
        }

        if self.latency_sim and getattr(self.latency_sim, "log_round_time_variance", False):
            fit_dict_return["timing/round_without_latency_s"] = float(total_round_time_wall)
            fit_dict_return["timing/round_latency_component_s"] = float(simulated_latency_total)
        cprint("----------------------------Results After Fit--------------------------------")
        print(json.dumps(fit_dict_return, indent=4))
        cprint("-----------------------------------------------------------------------")

        print("Logging Client States")
        if rnd != 0 and not drop_due_to_latency:
            metrics = local_eval_metrics
            metrics.setdefault("accuracy_per_exp", []).append(json.dumps(curr_accpexp))
            metrics.setdefault("stream_accuracy", []).append(stream_acc)
            metrics.setdefault("stream_loss", []).append(stream_loss)
            metrics.setdefault("cumalative_forgetting_measure", []).append(cmfm)
            metrics.setdefault("stepwise_forgetting_measure", []).append(swfm)
            metrics.setdefault("rounds_selected", []).append(rnd)

        cprint("Finished Fit")

        clear_memory()
        print(f"Memory cleared after fit round {rnd}")

        if random.random() < getattr(self.cfg.client, "falloff", 0.0):
            return None
        experience_idx = (rnd - 1) % experience_count
        next_train_len = (
            self.trainlen_per_exp[experience_idx] if self.trainlen_per_exp else 0
        )
        return get_parameters(self.cl_strategy.model), next_train_len, fit_dict_return

    def evaluate(self, parameters: Sequence[Any], config: dict[str, Any]):
        set_parameters(self.net, parameters)
        rnd = config["server_round"]

        cl_strategy, evaluation = make_cl_strat(self.net)

        results: List[Any] = []
        print(
            f"------------------------Local Client {self.partition_id} Evaluation on Updated Global Model--------------------"
        )

        results.append(cl_strategy.eval(self._test_stream))
        last_metrics = evaluation.get_last_metrics()
        stream_loss = self.adapter.extract_stream_metric(
            last_metrics, "Loss_Stream", self._test_stream_name
        )
        stream_acc = self.adapter.extract_stream_metric(
            last_metrics, "Top1_Acc_Stream", self._test_stream_name
        )

        curr_accpexp: List[float] = []
        for res in results:
            if not isinstance(res, dict):
                continue
            for exp, acc in res.items():
                if isinstance(exp, str) and exp.startswith("Top1_Acc_Exp/"):
                    curr_accpexp.append(float(acc))

        global_eval_metrics = self.client_state.config_records["global_eval_metrics"]
        hist_accpexp = global_eval_metrics["accuracy_per_exp"]

        cm_fmpexp = []
        num_experiences = max(1, self.num_experiences_config)
        if curr_accpexp:
            for i, entry in enumerate(hist_accpexp):
                prev_acc = json.loads(entry)
                idx = i % num_experiences
                if idx < len(prev_acc) and idx < len(curr_accpexp):
                    cm_fmpexp.append(prev_acc[idx] - curr_accpexp[idx])
        cmfm = sum(cm_fmpexp) / len(cm_fmpexp) if cm_fmpexp else 0.0

        cprint("Check Cumalative FM", "blue")
        cprint("History of Accuracy per Experience for this client")
        print(json.dumps(hist_accpexp, indent=2))
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Cumalative Forgetting per Experience: {json.dumps(cm_fmpexp, indent=4)}")
        print(f"Cumalative Forgetting Measure: {cmfm}")

        sw_fmpexp = []
        if hist_accpexp:
            prev_accpexp = json.loads(hist_accpexp[-1])
        else:
            prev_accpexp = []
        for prev_acc, curr_acc in zip(prev_accpexp, curr_accpexp):
            sw_fmpexp.append(prev_acc - curr_acc)
        denominator = max(1, self.num_experiences_config)
        swfm = (sum(sw_fmpexp) / denominator) if sw_fmpexp else 0.0

        cprint("Check StepWise FM", "blue")
        print(f"Current Accuracy per Experience: {json.dumps(curr_accpexp, indent=4)}")
        print(f"Prev Accuracy per Experience {json.dumps(prev_accpexp, indent=4)}")
        print(f"StepWise Forgetting per Experience: {json.dumps(sw_fmpexp, indent=4)}")
        print(f"StepWise Forgetting Measure: {swfm}")

        print("Eval of Client: ")
        print("Loss: ", stream_loss)
        print("Acc: ", stream_acc)
        print("Per Exp Acc: ", curr_accpexp)

        eval_dict_return = {
            "stream_accuracy": float(stream_acc),
            "stream_loss": float(stream_loss),
            "accuracy_per_experience": json.dumps(curr_accpexp),
            "stepwise_forgetting_measure": float(swfm),
            "cumalative_forgetting_measure": float(cmfm),
            "stepwise_forgetting_per_experience": json.dumps(sw_fmpexp),
            "cumalative_forgetting_per_experience": json.dumps(cm_fmpexp),
            "server_round": rnd,
            "pid": self.partition_id,
        }

        print(f"Global Distributed Evaluation of Client {self.partition_id}")
        print(json.dumps(eval_dict_return, indent=4))

        cprint("Logging Client States")
        if rnd != 0:
            metrics = global_eval_metrics
            metrics.setdefault("accuracy_per_exp", []).append(json.dumps(curr_accpexp))
            metrics.setdefault("stream_accuracy", []).append(stream_acc)
            metrics.setdefault("stream_loss", []).append(stream_loss)
            metrics.setdefault("cumalative_forgetting_measure", []).append(cmfm)
            metrics.setdefault("stepwise_forgetting_measure", []).append(swfm)
            metrics.setdefault("rounds_selected", []).append(rnd)
        return float(stream_loss), sum(self.testlen_per_exp), eval_dict_return