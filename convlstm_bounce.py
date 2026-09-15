from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
IMAGE_SIZE = 32
SEQUENCE_LENGTH = 10
TRAIN_SIZE = 2000
TEST_SIZE = 200
DATA_SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    group: int
    model_type: str = "convlstm"
    layers: int = 1
    hidden: int = 32
    kernel: int = 3
    input_frames: int = 4
    loss_name: str = "mse"
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    seed: int = 42


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig("baseline", 0),
    "exp1": ExperimentConfig("exp1", 1, layers=2),
    "exp2": ExperimentConfig("exp2", 2, hidden=64),
    "exp3": ExperimentConfig("exp3", 3, kernel=5),
    "exp4": ExperimentConfig("exp4", 4, input_frames=8),
    "exp5": ExperimentConfig("exp5", 5, model_type="flatten_lstm"),
    "exp6": ExperimentConfig("exp6", 6, loss_name="l1"),
}


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def configure_chinese_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "SimHei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def make_sequences(
    n_seq: int = TRAIN_SIZE + TEST_SIZE,
    T: int = SEQUENCE_LENGTH,
    size: int = IMAGE_SIZE,
    r: int = 2,
    seed: int = DATA_SEED,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_seq, T, size, size), dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    for s in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        positions: list[list[float]] = []
        for t in range(T):
            x, y = x + vx, y + vy
            if x < r or x > size - r:
                vx = -vx
            if y < r or y > size - r:
                vy = -vy
            positions.append([float(x), float(y)])
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
        metadata.append(
            {
                "sequence_index": s,
                "initial_speed": float(np.hypot(vx, vy)),
                "positions": positions,
            }
        )
    output = seqs[..., None]
    return (output, metadata) if return_metadata else output


def make_tensors(data: np.ndarray, input_frames: int) -> tuple[torch.Tensor, ...]:
    if data.shape != (TRAIN_SIZE + TEST_SIZE, SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 1):
        raise ValueError(f"正式数据形状错误：{data.shape}")
    if not 1 <= input_frames < SEQUENCE_LENGTH:
        raise ValueError("输入帧数必须在 1 到 9 之间")

    train_slice = slice(0, TRAIN_SIZE)
    test_slice = slice(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE)
    assert train_slice.stop <= test_slice.start
    x = torch.from_numpy(data[:, :input_frames]).permute(0, 1, 4, 2, 3)
    y = torch.from_numpy(data[:, input_frames, :, :, 0])
    return x[train_slice], y[train_slice], x[test_slice], y[test_slice]


class ConvLSTMCell(nn.Module):


    def __init__(self, in_ch: int, hid_ch: int, k: int = 3) -> None:
        super().__init__()
        if k % 2 != 1:
            raise ValueError("卷积核必须为奇数，才能用 k//2 保持空间尺寸")
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int = 1, hid: int = 32, k: int = 3, layers: int = 1) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers 至少为 1")
        channels = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            ConvLSTMCell(channels[index], channels[index + 1], k)
            for index in range(layers)
        )
        self.out = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"ConvLSTM 输入必须是 (B,T,C,H,W)，实际为 {tuple(x.shape)}")
        batch, _, _, height, width = x.shape
        states = [
            (
                x.new_zeros((batch, cell.hid, height, width)),
                x.new_zeros((batch, cell.hid, height, width)),
            )
            for cell in self.cells
        ]
        for t in range(x.shape[1]):
            current = x[:, t]
            for layer_index, cell in enumerate(self.cells):
                states[layer_index] = cell(current, states[layer_index])
                current = states[layer_index][0]
        return self.out(states[-1][0]).squeeze(1)


class FlattenLSTM(nn.Module):


    def __init__(self, size: int = IMAGE_SIZE, hidden: int = 256) -> None:
        super().__init__()
        pixels = size * size
        self.size = size
        self.lstm = nn.LSTM(pixels, hidden, batch_first=True)
        self.out = nn.Linear(hidden, pixels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"FlattenLSTM 输入必须是 (B,T,C,H,W)，实际为 {tuple(x.shape)}")
        batch, steps = x.shape[:2]
        flattened = x.reshape(batch, steps, -1)
        sequence, _ = self.lstm(flattened)
        return self.out(sequence[:, -1]).reshape(batch, self.size, self.size)


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_type == "convlstm":
        return ConvLSTM(hid=config.hidden, k=config.kernel, layers=config.layers)
    if config.model_type == "flatten_lstm":
        return FlattenLSTM()
    raise ValueError(f"未知模型类型：{config.model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predicted = model(inputs)
            difference = predicted - targets
            squared_error += difference.square().sum().item()
            absolute_error += difference.abs().sum().item()
            element_count += targets.numel()
            predictions.append(predicted.cpu())
    mse = squared_error / element_count
    mae = absolute_error / element_count
    if not np.isfinite([mse, mae]).all():
        raise FloatingPointError(f"测试指标非有限值：MSE={mse}, MAE={mae}")
    return mse, mae, torch.cat(predictions)


def _log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def train_experiment(
    config: ExperimentConfig,
    data: np.ndarray,
    device: torch.device,
    output_dir: Path,
    run_name: str | None = None,
    smoke: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float]], torch.Tensor]:
    run_name = run_name or config.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"run_log_{run_name}.txt"
    if log_path.exists():
        log_path.unlink()

    effective = replace(
        config,
        epochs=1 if smoke else config.epochs,
        batch_size=8 if smoke else config.batch_size,
    )
    set_seed(effective.seed)
    train_x, train_y, test_x, test_y = make_tensors(data, effective.input_frames)
    if smoke:
        train_x, train_y = train_x[:32], train_y[:32]
        test_x, test_y = test_x[:16], test_y[:16]

    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(effective.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=effective.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=effective.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = build_model(effective).to(device)
    parameter_count = count_parameters(model)
    loss_fn: nn.Module = nn.MSELoss() if effective.loss_name == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=effective.learning_rate)
    config_record = {
        **asdict(effective),
        "run_name": run_name,
        "smoke": smoke,
        "device": str(device),
        "parameter_count": parameter_count,
        "train_size": len(train_x),
        "test_size": len(test_x),
        "data_seed": DATA_SEED,
        "target_frame_one_based": effective.input_frames + 1,
    }
    (output_dir / f"config_{run_name}.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"配置：{json.dumps(config_record, ensure_ascii=False, sort_keys=True)}", log_path)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    history: list[dict[str, float]] = []
    final_predictions = torch.empty(0)
    for epoch in range(effective.epochs):
        model.train()
        loss_sum = 0.0
        element_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(inputs)
            loss = loss_fn(predicted, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"第 {epoch + 1} 轮出现非有限训练损失")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.numel()
            element_count += targets.numel()
        train_loss = loss_sum / element_count
        test_mse, test_mae, final_predictions = evaluate(model, test_loader, device)
        epoch_record = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "test_mse": float(test_mse),
            "test_mae": float(test_mae),
        }
        history.append(epoch_record)
        _log(
            f"epoch {epoch + 1}/{effective.epochs} "
            f"train_{effective.loss_name.upper()}={train_loss:.8f} "
            f"test_MSE={test_mse:.8f} test_MAE={test_mae:.8f}",
            log_path,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    metrics: dict[str, Any] = {
        **config_record,
        "final_train_loss": history[-1]["train_loss"],
        "test_mse": history[-1]["test_mse"],
        "test_mae": history[-1]["test_mae"],
        "elapsed_seconds": elapsed,
    }
    (output_dir / f"history_{run_name}.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"metrics_{run_name}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        f"完成：test_MSE={metrics['test_mse']:.8f} "
        f"test_MAE={metrics['test_mae']:.8f} 耗时={elapsed:.3f}s",
        log_path,
    )
    plot_history(history, effective, output_dir / curve_filename(run_name))
    return metrics, history, final_predictions


def curve_filename(run_name: str) -> str:
    if run_name == "baseline":
        return "result_baseline.png"
    if run_name.startswith("exp"):
        return f"result_{run_name}.png"
    return f"result_{run_name}.png"


def plot_history(
    history: list[dict[str, float]], config: ExperimentConfig, output_path: Path
) -> None:
    configure_chinese_font()
    epochs = [int(record["epoch"]) for record in history]
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(epochs, [r["train_loss"] for r in history], "o-", label=f"训练 {config.loss_name.upper()}")
    plt.plot(epochs, [r["test_mse"] for r in history], "s-", label="测试 MSE")
    plt.xlabel("Epoch")
    plt.ylabel("损失 / 误差")
    plt.title(f"{config.experiment} 训练曲线")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def choose_best_config(rows: dict[str, dict[str, Any]]) -> tuple[ExperimentConfig, dict[str, Any]]:
    baseline_mse = float(rows["baseline"]["test_mse"])
    chosen: list[str] = []
    config = replace(EXPERIMENTS["baseline"], experiment="best", group=7)

    max_best_parameters = 400_000

    modifications: dict[str, dict[str, Any]] = {
        "exp1": {"layers": 2},
        "exp2": {"hidden": 64},
        "exp3": {"kernel": 5},
        "exp4": {"input_frames": 8},
        "exp6": {"loss_name": "l1"},
    }
    decisions: dict[str, Any] = {"baseline_mse": baseline_mse, "experiments": {}}
    for experiment, changes in modifications.items():
        mse = float(rows[experiment]["test_mse"])
        improved = mse < baseline_mse
        decisions["experiments"][experiment] = {
            "test_mse": mse,
            "improved_over_baseline": improved,
            "changes": changes,
        }
        if improved:
            candidate = replace(config, **changes)
            candidate_parameters = count_parameters(build_model(candidate))
            decisions["experiments"][experiment]["candidate_parameter_count"] = candidate_parameters
            if candidate_parameters <= max_best_parameters:
                config = candidate
                chosen.append(experiment)
            else:
                decisions["experiments"][experiment]["skipped_reason"] = (
                    f"组合参数量 {candidate_parameters} 超过 {max_best_parameters} 上限，"
                    "避免 batch=64 时显存不足。"
                )
    decisions["chosen_experiments"] = chosen
    decisions["max_best_parameters"] = max_best_parameters
    decisions["excluded_structure_control"] = {
        "experiment": "exp5",
        "reason": "全连接 LSTM 是结构对照，不能与 ConvLSTM 内部改进直接组合。",
        "test_mse": float(rows["exp5"]["test_mse"]),
    }
    decisions["best_config"] = asdict(config)
    return config, decisions


def upsert_metrics_csv(output_dir: Path, new_rows: Iterable[dict[str, Any]]) -> None:
    csv_path = output_dir / "metrics.csv"
    rows_by_name: dict[str, dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_by_name[row["run_name"]] = row
    for row in new_rows:
        rows_by_name[str(row["run_name"])] = row
    fields = [
        "run_name", "experiment", "group", "model_type", "layers", "hidden", "kernel",
        "input_frames", "loss_name", "learning_rate", "batch_size", "epochs", "seed",
        "data_seed", "train_size", "test_size", "parameter_count", "device", "smoke",
        "final_train_loss", "test_mse", "test_mae", "elapsed_seconds",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        def sort_key(item: dict[str, Any]) -> tuple[int, str]:
            try:
                return int(item.get("group", 99)), str(item.get("run_name", ""))
            except (TypeError, ValueError):
                return 99, str(item.get("run_name", ""))
        writer.writerows(sorted(rows_by_name.values(), key=sort_key))


def save_best_average(output_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    average = dict(runs[0])
    average["run_name"] = "best_average"
    for field in ("final_train_loss", "test_mse", "test_mae", "elapsed_seconds"):
        average[field] = float(np.mean([float(run[field]) for run in runs]))
    average["parameter_count"] = runs[0]["parameter_count"]
    average["repeat_count"] = 3
    (output_dir / "metrics_best_average.json").write_text(
        json.dumps(average, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return average


def select_visual_samples(metadata: list[dict[str, Any]], input_frames: int) -> list[int]:
    test_meta = metadata[TRAIN_SIZE:]
    speed_index = max(range(TEST_SIZE), key=lambda i: test_meta[i]["initial_speed"])
    boundary_index = min(
        range(TEST_SIZE),
        key=lambda i: min(
            min(x, y, IMAGE_SIZE - x, IMAGE_SIZE - y)
            for x, y in test_meta[i]["positions"][: input_frames + 1]
        ),
    )
    ordinary_index = TEST_SIZE // 2
    selected: list[int] = []
    for candidate in (ordinary_index, speed_index, boundary_index):
        if candidate not in selected:
            selected.append(candidate)
    for candidate in range(TEST_SIZE):
        if len(selected) == 3:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected


def plot_predictions(
    data: np.ndarray,
    predictions: torch.Tensor,
    metadata: list[dict[str, Any]],
    input_frames: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    configure_chinese_font()
    indices = select_visual_samples(metadata, input_frames)
    columns = input_frames + 2
    figure, axes = plt.subplots(3, columns, figsize=(1.75 * columns, 5.6), squeeze=False)
    records: list[dict[str, Any]] = []
    for row, test_index in enumerate(indices):
        global_index = TRAIN_SIZE + test_index
        for t in range(input_frames):
            axes[row, t].imshow(data[global_index, t, :, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, t].set_title(f"输入 {t + 1}", fontsize=9)
        target = data[global_index, input_frames, :, :, 0]
        prediction = predictions[test_index].numpy()
        sample_mse = float(np.mean((prediction - target) ** 2))
        axes[row, input_frames].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, input_frames].set_title("真实下一帧", fontsize=9)
        axes[row, input_frames + 1].imshow(np.clip(prediction, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[row, input_frames + 1].set_title(f"预测帧\nMSE={sample_mse:.6f}", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
        records.append(
            {
                "test_index": test_index,
                "global_sequence_index": global_index,
                "sample_mse": sample_mse,
                "initial_speed": metadata[global_index]["initial_speed"],
            }
        )
    figure.suptitle("输入帧 | 真实下一帧 | 预测帧", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    (output_path.parent / "prediction_samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "device_used": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "chinese_font": configure_chinese_font(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但 PyTorch 检测不到可用 GPU")
    return device


def load_formal_metrics(output_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        path = output_dir / f"metrics_{experiment}.json"
        if not path.exists():
            raise FileNotFoundError(f"缺少正式实验结果：{path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("smoke"):
            raise ValueError(f"不能用 smoke 结果选择最优组合：{path}")
        rows[experiment] = row
    return rows


def run_all(
    data: np.ndarray,
    metadata: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
) -> None:
    base_rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        metrics, _, _ = train_experiment(EXPERIMENTS[experiment], data, device, output_dir)
        base_rows.append(metrics)
        upsert_metrics_csv(output_dir, [metrics])
    row_map = {str(row["experiment"]): row for row in base_rows}
    best_config, decisions = choose_best_config(row_map)
    (output_dir / "best_selection.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_runs: list[dict[str, Any]] = []
    first_predictions = torch.empty(0)
    for run_number in range(1, 4):
        run_name = f"best_run{run_number}"
        metrics, history, predictions = train_experiment(
            best_config, data, device, output_dir, run_name=run_name
        )
        best_runs.append(metrics)
        upsert_metrics_csv(output_dir, [metrics])
        if run_number == 1:
            first_predictions = predictions
            plot_history(history, best_config, output_dir / "result_best.png")
    average = save_best_average(output_dir, best_runs)
    upsert_metrics_csv(output_dir, [average])
    plot_predictions(
        data,
        first_predictions,
        metadata,
        best_config.input_frames,
        output_dir / "result_pred.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=[*EXPERIMENTS.keys(), "all", "best"],
        default="baseline",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true", help="仅跑小数据 1 epoch，不写正式汇总")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke and args.experiment in {"all", "best"}:
        raise SystemExit("smoke 模式请选择 baseline 或 exp1~exp6 中的一组")
    device = resolve_device(args.device)
    output_dir = args.results_dir / "smoke" if args.smoke else args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    info = environment_info(device)
    (output_dir / "environment.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"运行设备：{device}；环境：{json.dumps(info, ensure_ascii=False)}", flush=True)
    generated = make_sequences(return_metadata=True)
    assert isinstance(generated, tuple)
    data, metadata = generated
    if args.experiment == "all":
        run_all(data, metadata, device, output_dir)
        return
    if args.experiment == "best":
        rows = load_formal_metrics(output_dir)
        best_config, decisions = choose_best_config(rows)
        (output_dir / "best_selection.json").write_text(
            json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        best_runs: list[dict[str, Any]] = []
        first_predictions = torch.empty(0)
        first_history: list[dict[str, float]] = []
        for run_number in range(1, 4):
            metrics, history, predictions = train_experiment(
                best_config, data, device, output_dir, run_name=f"best_run{run_number}"
            )
            best_runs.append(metrics)
            upsert_metrics_csv(output_dir, [metrics])
            if run_number == 1:
                first_predictions, first_history = predictions, history
        average = save_best_average(output_dir, best_runs)
        upsert_metrics_csv(output_dir, [average])
        plot_history(first_history, best_config, output_dir / "result_best.png")
        plot_predictions(
            data, first_predictions, metadata, best_config.input_frames, output_dir / "result_pred.png"
        )
        return
    config = EXPERIMENTS[args.experiment]
    metrics, _, _ = train_experiment(config, data, device, output_dir, smoke=args.smoke)
    if not args.smoke:
        upsert_metrics_csv(output_dir, [metrics])


if __name__ == "__main__":
    main()
