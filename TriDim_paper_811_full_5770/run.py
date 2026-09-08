"""Train and evaluate the TriDim paper models.

The public protocol uses subject-aware 8:1:1 splits, seeds 5/42/43,
validation-Accuracy checkpoint selection and sample
standard deviation across seeds.  SleepEDF and SEED-V use the fixed manifests
distributed in ``configs/splits``.

Example (three-seed Full run on FACED):

    python run.py \
        --model eeg_mixer_v11_1_spatial_multilevel --data FACED_new \
        --dataset_paths_yaml ./configs/paper/full/FACED_new_full.yaml \
        --gpu 0 --gpu_idx 0 --num_workers 4 \
        --seeds 5 42 43
"""
from utils.experiment_record import collect_experiment_record
import time
import argparse
import os
import random
import subprocess
import sys

import numpy as np
import psutil
import torch

from exp.exp_classification import Exp_Classification


PAPER_MODELS = (
    "eeg_mixer_v11_1_spatial_multilevel",
    "eeg_mixer_v11_2_no_stem",
    "eeg_mixer_v11_1_spatial_multilevel_noC",
    "eeg_mixer_v11_1_spatial_multilevel_noK",
    "eeg_mixer_v11_1_spatial_multilevel_noT",
    "eeg_mixer_v11_1_spatial_multilevel_onlyC",
    "eeg_mixer_v11_1_spatial_multilevel_onlyK",
    "eeg_mixer_v11_1_spatial_multilevel_onlyT",
    "eeg_mixer_v11_1_spatial_multilevel_noxattn",
    "eeg_mixer_v11_1_spatial_multilevel_sharedffn",
    "eeg_mixer_v11_1_spatial_multilevel_indepattn",
    "eeg_mixer_v11_1_spatial_multilevel_seqckt",
    "eeg_mixer_v11_1_spatial_multilevel_flattf",
    "eeg_mixer_v11_1_spatial_multilevel_crisscross",
)

# YAML parameters accepted by the public runner.  Rejecting unknown keys avoids
# the silent no-op behavior that affected several historical experiment files.
ALLOWED_CONFIG_PARAMS = {
    "batch_size", "canonical_channel_coord_path",
    "canonical_channel_names", "canonical_channels",
    "channel_basis_dim", "class_weight_mode", "d_model", "dataset",
    "drop_path_c", "drop_path_k", "drop_path_mlp", "drop_path_schedule",
    "drop_path_t", "dropout", "electrode_channel_count", "electrode_csv",
    "electrode_montage_used", "eta_min", "eval_freq",
    "external_split_manifest", "focal_gamma", "gpu", "gpu_idx",
    "gradient_clip_norm", "input_channel_coord_path", "itr", "k_basis_dim",
    "label_smoothing", "layer_scale_init", "learning_rate", "lradj", "model",
    "n_heads", "no_channel_prior", "num_class", "num_workers", "optimizer",
    "patch_embed_dim", "patch_len", "patch_stride", "patience", "root_path",
    "seed", "seeds", "seed_start", "select_metric", "seq_len", "split_mode",
    "stem_dropout", "stem_hidden_mult", "stem_kernels", "t_basis_dim",
    "t_layer", "train_epochs", "train_ratio", "use_amp",
    "use_channel_prior", "use_cosine_scheduler",
    "use_focal_loss", "use_gpu", "use_input_norm", "use_multi_gpu",
    "use_multi_level_readout", "use_subject_balanced_sampler",
    "axis_execution_mode", "use_axis_attention", "use_axis_ffn",
    "use_subject_label_for_split", "val_ratio", "warmup_epochs", "weight_decay",
}


def use_cpus(gpus, cpus_per_gpu):
    cpus = []
    for gpu in gpus:
        cpus.extend(list(range(gpu * cpus_per_gpu, (gpu + 1) * cpus_per_gpu)))
    p = psutil.Process()
    try:
        p.cpu_affinity(cpus)
    except Exception:
        # cpu_affinity is unsupported on some platforms (e.g. macOS); ignore.
        return
    print(
        "A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(
            len(cpus)
        )
    )


def _get_cli_overrides(argv):
    """Return arg names explicitly set on CLI, normalized to underscore style."""
    names = set()
    i = 0
    while i < len(argv):
        tok = str(argv[i]).strip()
        if not tok.startswith("--"):
            i += 1
            continue
        if "=" in tok:
            key = tok[2:].split("=", 1)[0]
            names.add(key.replace("-", "_"))
            i += 1
            continue
        key = tok[2:]
        names.add(key.replace("-", "_"))
        i += 1
        if i < len(argv) and not str(argv[i]).startswith("--"):
            i += 1
    return names


def _load_simple_kv_yaml(path):
    """Tiny YAML fallback parser for simple `key: value` files."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]
            mapping[str(key)] = os.path.expanduser(str(value))
    return mapping


def _load_yaml_obj(yaml_path):
    if not yaml_path or not os.path.isfile(yaml_path):
        return None
    try:
        import yaml  # type: ignore

        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return _load_simple_kv_yaml(yaml_path)


def _coerce_value(current_value, new_value):
    if new_value is None:
        return current_value
    if current_value is None:
        return new_value
    if isinstance(current_value, bool):
        if isinstance(new_value, bool):
            return new_value
        txt = str(new_value).strip().lower()
        return txt in ("1", "true", "yes", "y", "on")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(new_value)
    if isinstance(current_value, float):
        return float(new_value)
    if isinstance(current_value, list):
        if isinstance(new_value, list):
            return new_value
        return [int(x.strip()) for x in str(new_value).split(",") if x.strip()]
    return new_value


def _coerce_untyped_yaml_value(value):
    if isinstance(value, list):
        return [_coerce_untyped_yaml_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_untyped_yaml_value(v) for k, v in value.items()}
    if not isinstance(value, str):
        return value
    text = value.strip()
    lower = text.lower()
    if lower in ("true", "yes", "y", "on"):
        return True
    if lower in ("false", "no", "n", "off"):
        return False
    if lower in ("none", "null", "~"):
        return None
    try:
        if text and all(ch in "+-0123456789" for ch in text):
            return int(text)
    except ValueError:
        pass
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
    except ValueError:
        pass
    return value


def _extract_dataset_cfg(yaml_path, data_key):
    """Read a per-dataset yaml. Supports flat / nested / single-dataset schemas."""
    raw = _load_yaml_obj(yaml_path)
    if not isinstance(raw, dict):
        return {}

    cfg = {}

    # Single-dataset schema: top-level "dataset: <name>" + root_path + params.
    if str(raw.get("dataset", "")).strip() == str(data_key):
        if isinstance(raw.get("root_path"), str):
            cfg["root_path"] = os.path.expandvars(os.path.expanduser(raw["root_path"]))
        if isinstance(raw.get("params"), dict):
            cfg["params"] = dict(raw["params"])
        return cfg

    # Nested under dataset_paths.
    if isinstance(raw.get("dataset_paths"), dict):
        node = raw["dataset_paths"].get(data_key)
        if isinstance(node, str):
            return {"root_path": os.path.expandvars(os.path.expanduser(node))}
        if isinstance(node, dict):
            out = {}
            if isinstance(node.get("root_path"), str):
                out["root_path"] = os.path.expandvars(os.path.expanduser(node["root_path"]))
            if isinstance(node.get("params"), dict):
                out["params"] = dict(node["params"])
            return out

    # Direct by dataset key.
    node = raw.get(data_key)
    if isinstance(node, str):
        return {"root_path": os.path.expandvars(os.path.expanduser(node))}
    if isinstance(node, dict):
        out = {}
        if isinstance(node.get("root_path"), str):
            out["root_path"] = os.path.expandvars(os.path.expanduser(node["root_path"]))
        if isinstance(node.get("params"), dict):
            out["params"] = dict(node["params"])
        if "params" not in out:
            direct_params = {k: v for k, v in node.items() if k != "root_path"}
            if direct_params:
                out["params"] = direct_params
        return out

    # Flat map fallback.
    if data_key in raw and isinstance(raw[data_key], str):
        return {"root_path": os.path.expandvars(os.path.expanduser(raw[data_key]))}
    return {}


def _resolve_downstream_dataset_root(downstream_root, data_key):
    if not downstream_root:
        return None
    base = os.path.expanduser(str(downstream_root))
    if not os.path.isdir(base):
        return None
    path = os.path.join(base, str(data_key))
    return path if os.path.isdir(path) else None

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriDim paper experiment runner")

    # model / data
    parser.add_argument(
        "--model",
        type=str,
        default="eeg_mixer_v11_1_spatial_multilevel",
        choices=PAPER_MODELS,
        help="Full TriDim or one of the released axis/block variants",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="paper dataset key, see configs/paper/full/",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/",
        help="root path of the dataset; if omitted, taken from --dataset_paths_yaml",
    )
    parser.add_argument(
        "--dataset_paths_yaml",
        type=str,
        default="./configs/paper/full/FACED_new_full.yaml",
        help="paper YAML supplying the dataset path and all experiment parameters",
    )
    parser.add_argument(
        "--downstream_root",
        type=str,
        default="",
        help="optional shared downstream dir; if set and contains a <data> subdir it overrides yaml root_path",
    )

    # patch / model size
    parser.add_argument("--seq_len", type=int, default=96, help="placeholder, overridden by data")
    parser.add_argument("--patch_len", type=int, default=80)
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=None,
        help="temporal patch stride; defaults to patch_len (no overlap) when omitted",
    )
    parser.add_argument("--enc_in", type=int, default=7, help="placeholder, overridden by data")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)

    # basis-mixer projection dims
    parser.add_argument("--channel_basis_dim", type=int, default=None)
    parser.add_argument("--k_basis_dim", type=int, default=None)
    parser.add_argument("--t_basis_dim", type=int, default=None)
    parser.add_argument("--patch_embed_dim", type=int, default=None)

    parser.add_argument("--canonical_channels", type=int, default=64)
    parser.add_argument("--no_channel_prior", action="store_true")
    parser.add_argument("--input_channel_coord_path", type=str, default=None)
    parser.add_argument("--canonical_channel_coord_path", type=str, default=None)
    parser.add_argument("--canonical_channel_names", type=str, default=None)

    # depth / regularization
    parser.add_argument("--t_layer", type=int, default=3, help="number of TriAxis encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    # optimization
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--itr", type=int, default=1, help="number of seeds, seeds = [seed_start, seed_start+itr-1]")
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="custom seed list, e.g. --seeds 5 42 43. overrides seed_start and itr when set")
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8, help="early stopping patience")
    parser.add_argument(
        "--select_metric",
        type=str,
        default="Accuracy",
        choices=["F1", "Accuracy"],
        help="validation metric used for model selection",
    )

    # subject split
    parser.add_argument(
        "--split_mode",
        type=str,
        default="stratified_random",
        choices=["stratified_random", "external_manifest"],
        help=(
            "subject-aware stratified split, or the immutable per-seed "
            "external manifest specified by the paper YAML"
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--external_split_manifest", type=str, default=None)

    # learning rate
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--lradj",
        type=str,
        default="cosine",
        choices=["cosine", "constant", "binary", "type0", "type05", "type1", "type2", "type3", "type4"],
    )

    # mixed precision (off by default; enable with --use_amp)
    parser.add_argument("--use_amp", action="store_true", default=False)

    # GPU
    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0])
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    
    parser.add_argument("--bsub_script", type=str, default="")
    parser.add_argument("--exp_notes", type=str, default="")
    parser.add_argument(
        "--validate_config_only",
        action="store_true",
        help="load and validate the merged YAML/CLI configuration without training",
    )

    # Optimization/model options supplied by the paper YAML files.  Defining
    # them here gives CLI overrides a type and keeps configuration validation
    # explicit.
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_cosine_scheduler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weight_mode", type=str, default="none")
    parser.add_argument("--use_focal_loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=4.0)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--use_input_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_multi_level_readout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_axis_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_axis_ffn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--axis_execution_mode",
        choices=["parallel", "serial_ckt"],
        default="parallel",
    )
    parser.add_argument("--use_subject_label_for_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_subject_balanced_sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop_path_c", type=float, default=0.0)
    parser.add_argument("--drop_path_k", type=float, default=0.0)
    parser.add_argument("--drop_path_t", type=float, default=0.0)
    parser.add_argument("--drop_path_mlp", type=float, default=0.0)
    parser.add_argument("--drop_path_schedule", type=str, default="linear")
    parser.add_argument("--layer_scale_init", type=float, default=0.0)
    parser.add_argument("--stem_hidden_mult", type=float, default=1.0)
    parser.add_argument("--stem_dropout", type=float, default=0.0)
    parser.add_argument("--stem_kernels", nargs="+", type=int, default=None)
    parser.add_argument("--electrode_csv", type=str, default=None)
    parser.add_argument("--electrode_channel_count", type=int, default=None)
    parser.add_argument("--electrode_montage_used", type=str, default=None)
    parser.add_argument("--num_class", type=int, default=None)

    args = parser.parse_args()
    args.use_channel_prior = not args.no_channel_prior
    # Ensure device visibility is pinned before any CUDA query/initialization.

    # GPU
    if args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        else:
            print(f"[device] keep scheduler CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        args.gpu = 0
    else:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"[device] set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print(f"[device] keep scheduler CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        args.gpu = 0
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    

    cli_overrides = _get_cli_overrides(sys.argv[1:])
    dataset_cfg = _extract_dataset_cfg(args.dataset_paths_yaml, args.data)
    if dataset_cfg:
        cfg_root = dataset_cfg.get("root_path")
        if cfg_root is not None:
            if "root_path" in cli_overrides:
                print(
                    "[dataset_cfg] keep CLI --root_path={}, yaml has {}".format(
                        args.root_path, cfg_root
                    )
                )
            else:
                args.root_path = cfg_root
                print("[dataset_cfg] {} root_path -> {}".format(args.data, args.root_path))

        cfg_params = dataset_cfg.get("params", {})
        if isinstance(cfg_params, dict):
            unknown = sorted(
                str(key).replace("-", "_")
                for key in cfg_params
                if str(key).replace("-", "_") not in ALLOWED_CONFIG_PARAMS
            )
            if unknown:
                raise ValueError(
                    "Unknown YAML parameter(s): {}. Refusing silent no-op config keys.".format(
                        ", ".join(unknown)
                    )
                )
            for key, value in cfg_params.items():
                attr = str(key).replace("-", "_")
                if attr in cli_overrides:
                    continue
                if attr in {
                    "external_split_manifest", "electrode_csv",
                    "input_channel_coord_path", "canonical_channel_coord_path",
                } and isinstance(value, str):
                    value = os.path.expandvars(os.path.expanduser(value))
                if not hasattr(args, attr):
                    setattr(args, attr, _coerce_untyped_yaml_value(value))
                    continue
                current = getattr(args, attr)
                try:
                    coerced = _coerce_value(current, value)
                except Exception:
                    coerced = value
                setattr(args, attr, coerced)
            if cfg_params:
                print(
                    "[dataset_cfg] applied {} params for {}".format(
                        len(cfg_params), args.data
                    )
                )
    elif args.dataset_paths_yaml and os.path.isfile(args.dataset_paths_yaml):
        print(
            "[dataset_cfg] {} not found in {}, using --root_path={}".format(
                args.data, args.dataset_paths_yaml, args.root_path
            )
        )

    if "root_path" not in cli_overrides:
        downstream_root = _resolve_downstream_dataset_root(args.downstream_root, args.data)
        if downstream_root is not None:
            if args.root_path != downstream_root:
                print(
                    "[dataset_cfg] {} root_path -> {} (from --downstream_root)".format(
                        args.data, downstream_root
                    )
                )
            args.root_path = downstream_root

    if args.validate_config_only:
        errors = []
        if not os.path.isdir(args.root_path):
            errors.append(f"dataset root does not exist: {args.root_path}")
        if args.model not in PAPER_MODELS:
            errors.append(f"unsupported model: {args.model}")
        if str(args.select_metric) != "Accuracy":
            errors.append("paper configurations require select_metric=Accuracy")
        manifest = getattr(args, "external_split_manifest", None)
        seeds = list(args.seeds or [])
        is_fixed_loso = (
            manifest
            and "{seed}" not in str(manifest)
            and seeds == [42]
        )
        if seeds != [5, 42, 43] and not is_fixed_loso:
            errors.append(
                "paper configurations require seeds=[5, 42, 43], except "
                "released fixed-fold LOSO controls which use seed 42"
            )
        if manifest:
            for seed in args.seeds:
                path = str(manifest).format(seed=seed)
                if not os.path.isfile(path):
                    errors.append(f"split manifest does not exist: {path}")
        if errors:
            raise ValueError("Invalid paper configuration:\n- " + "\n- ".join(errors))
        print(
            "[CONFIG VALID] dataset={} model={} root={} seeds={} mlr={} manifest={}".format(
                args.data,
                args.model,
                args.root_path,
                args.seeds,
                args.use_multi_level_readout,
                manifest or "generated subject-aware split",
            )
        )
        raise SystemExit(0)

    # Re-check CUDA availability after yaml ingestion: yaml may have flipped
    # use_gpu back to True even when torch wasn't built with CUDA. Without this
    # second guard the model would try to .to('cuda:0') and crash on CPU-only
    # boxes.
    if args.use_gpu and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to CPU.")
        args.use_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Pin CPU affinity per GPU. 12 CPUs/GPU is a sane default; tweak if needed.
    use_cpus(gpus=args.gpu_idx, cpus_per_gpu=12)

    Exp = Exp_Classification
    avg_metrics = []
    # 种子列表生成逻辑：优先使用自定义种子列表，否则沿用原连续种子逻辑
    if args.seeds is not None:
        seed_list = args.seeds
        # 同步更新itr，保证后续均值/标准差计算的循环次数正确
        args.itr = len(seed_list)
    else:
        # 原连续种子逻辑，完全保留
        seed_list = [args.seed_start + ii for ii in range(args.itr)]
    # seed range: [seed_start, seed_start + itr - 1]
    for seed in seed_list:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Force PyTorch to pick deterministic kernels everywhere it can.
        # Pair with CUBLAS_WORKSPACE_CONFIG=:4096:8 (set in shell) for bit-exact cuBLAS.
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == "":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as _e:
            print(f"[determinism] use_deterministic_algorithms failed: {_e}")

        # setting record of experiments
        args.seed = seed
        stride = args.patch_stride if args.patch_stride is not None else args.patch_len
        cb = args.channel_basis_dim if args.channel_basis_dim is not None else "auto"
        kb = args.k_basis_dim if args.k_basis_dim is not None else "auto"
        tb = args.t_basis_dim if args.t_basis_dim is not None else "auto"
        pe = args.patch_embed_dim if args.patch_embed_dim is not None else "auto"
        metric_tag = str(getattr(args, "select_metric", "Accuracy")).replace("/", "_")
        setting = (
            "{}_{}_seed_{}_dm_{}_dp_{}_tl_{}_bs_{}_lr{}_pl_{}"
            "_cb{}_kb{}_tb{}_pe{}_ps{}_cp{}_ml{}_sel{}".format(
                args.model,
                args.data,
                args.seed,
                args.d_model,
                args.dropout,
                args.t_layer,
                args.batch_size,
                args.learning_rate,
                args.patch_len,
                cb,
                kb,
                tb,
                pe,
                stride,
                int(args.use_channel_prior),
                int(getattr(args, "use_multi_level_readout", False)),
                metric_tag,
            )
        )


        git_hash = get_git_hash()
        run_id = time.strftime("%Y%m%d_%H%M%S")
        setting = setting + f"_git{git_hash}_{run_id}"
        exp = Exp(args)
        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))

        # Keep a single evaluation so logs and averages describe the same test run.

        test_metrics = exp.test(setting)

        avg_metrics.append(test_metrics)

        val_metrics = getattr(exp, "last_val_metrics", {})

        record, record_path = collect_experiment_record(
            args=args,
            setting=setting,
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            result_dir="results/runs",
            bsub_script=getattr(args, "bsub_script", ""),
            notes=getattr(args, "exp_notes", ""),
        )

        print(f"[result] Saved experiment record to: {record_path}")
        
        torch.cuda.empty_cache()

    keys = ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC")
    means = [np.mean([avg_metrics[i][k] for i in range(args.itr)]) for k in keys]
    stds = [
        np.std([avg_metrics[i][k] for i in range(args.itr)], ddof=1)
        if args.itr > 1 else 0.0
        for k in keys
    ]
    print(
        f"Mean accuracy: {means[0]:.4f}, precision: {means[1]:.4f},"
        f"recall: {means[2]:.4f}, f1: {means[3]:.4f},"
        f" AUROC: {means[4]:.4f}, AUPRC: {means[5]:.4f}"
    )
    print(
        f"Std accuracy: {stds[0]:.4f}, precision: {stds[1]:.4f},"
        f"recall: {stds[2]:.4f}, f1: {stds[3]:.4f},"
        f" AUROC: {stds[4]:.4f}, AUPRC: {stds[5]:.4f}"
    )
    print("=" * 80)
    print(
        "[CONFIG CHECK] model={} multi_level_readout={} "
        "selection={}".format(
            args.model,
            args.use_multi_level_readout,
            args.select_metric,
        )
    )
    print("=" * 80)
