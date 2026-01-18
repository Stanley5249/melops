"""CLI for exporting NVIDIA Parakeet ASR models to ONNX format."""

import argparse

from melops_export import export_model


def main():
    parser = argparse.ArgumentParser(
        prog="melops-export",
        description="export nvidia parakeet-tdt asr models to onnx format",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="huggingface repository id or path to .nemo file "
        "(default: nvidia/parakeet-tdt-0.6b-v3)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory for onnx model and tokenizer "
        "(default: <system_cache>/melops/models/<REPO_ID>)",
    )
    args = parser.parse_args()

    export_model(
        repo_id=args.repo_id,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
