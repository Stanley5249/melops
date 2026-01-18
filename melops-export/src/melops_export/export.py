"""Export NVIDIA Parakeet-TDT ASR models to ONNX with tokenizer.

NeMo exports create 295+ weight files; this consolidates them into standard
ONNX external data format (single .data file per model).
"""

import tempfile
from pathlib import Path

import onnx
import torch
from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel
from nemo.collections.common.tokenizers import (
    SentencePieceTokenizer as NemoSentencePieceTokenizer,
)
from onnx.external_data_helper import (
    convert_model_to_external_data,
    write_external_data_tensors,
)
from platformdirs import user_cache_dir
from tokenizers.implementations import (
    SentencePieceBPETokenizer as HFSentencePieceBPETokenizer,
)


def get_cache_dir() -> Path:
    """Get system cache directory with melops subdirectory."""
    return Path(user_cache_dir()) / "melops"


def resolve_output_dir(repo_id: str, out_dir: str | None) -> Path:
    """Resolve output directory, defaulting to <system_cache>/melops/models/<repo_id>."""
    if out_dir is not None:
        return Path(out_dir)
    return get_cache_dir() / "models" / repo_id.replace("/", "--")


def load_model(repo_id: str) -> ASRModel:
    """Load ASR model from HuggingFace Hub or local .nemo file."""
    model = ASRModel.from_pretrained(repo_id, map_location=torch.device("cpu"))
    assert isinstance(model, ASRModel), "loaded model is not an ASRModel"
    model.eval()
    return model


def consolidate_external_data(onnx_path: Path, output_dir: Path) -> Path:
    """Consolidate NeMo's 295+ weight files into single .data file.

    NeMo exports create many separate weight files (external data format).
    This loads the model and repacks all weights contiguously into one .data file.
    Required for models >2GB due to protobuf limits.
    """
    model = onnx.load(str(onnx_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx = output_dir / onnx_path.name
    data_file = output_onnx.stem + ".data"
    data_path = output_dir / data_file

    # Remove existing data file to prevent appending
    if data_path.exists():
        data_path.unlink()

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )

    write_external_data_tensors(model, str(output_dir))
    onnx.save(model, str(output_onnx))
    print(f"Saved model to {output_onnx} ({data_file})")
    return output_onnx


def export_onnx_rnnt(
    model: EncDecRNNTBPEModel, temp_dir: Path, output_dir: Path
) -> None:
    """Export RNNT model to ONNX, consolidate weights, move to output_dir.

    Creates encoder-model.onnx + .data and decoder_joint-model.onnx + .data.
    Uses batch_size=1 with dynamic sequence lengths.
    """
    # Dynamic axes for variable-length audio sequences
    # Encoder input: audio_signal (1, 128, time) - time at axis 2
    # Encoder output: outputs (1, 8192, time) - time at axis 2
    # Decoder input: encoder_outputs (1, 8192, time) - time at axis 2
    dynamic_axes = {
        "audio_signal": {2: "time"},
        "outputs": {2: "time"},
        "encoder_outputs": {2: "time"},
    }

    model.export(str(temp_dir / "model.onnx"), dynamic_axes=dynamic_axes)

    output_dir.mkdir(parents=True, exist_ok=True)
    for onnx_file in temp_dir.glob("*.onnx"):
        consolidate_external_data(onnx_file, output_dir)


def convert_sentencepiece_to_tokenizer(
    tokenizer: NemoSentencePieceTokenizer,
) -> HFSentencePieceBPETokenizer:
    """Convert NeMo SentencePiece tokenizer to HuggingFace format."""
    vocab = {k: v for (v, k) in enumerate(tokenizer.vocab)}
    return HFSentencePieceBPETokenizer(vocab, [])


def export_tokenizer(model: EncDecRNNTBPEModel, output_path: Path) -> None:
    """Export tokenizer as tokenizer.json."""
    nemo_tokenizer = model.tokenizer
    assert isinstance(nemo_tokenizer, NemoSentencePieceTokenizer), (
        f"expected SentencePieceTokenizer, got {type(nemo_tokenizer).__name__}"
    )

    hf_tokenizer = convert_sentencepiece_to_tokenizer(nemo_tokenizer)
    tokenizer_path = output_path / "tokenizer.json"
    hf_tokenizer.save(str(tokenizer_path))

    print(f"Saved tokenizer to {tokenizer_path}")


def export_model(
    repo_id: str | None = None,
    out_dir: str | None = None,
) -> None:
    """Export Parakeet-TDT model to ONNX with tokenizer.

    Args:
        repo_id: HuggingFace repo ID or .nemo path (default: nvidia/parakeet-tdt-0.6b-v3)
        out_dir: Output directory (default: <system_cache>/melops/models/<repo_id>)
    """
    if repo_id is None:
        repo_id = "nvidia/parakeet-tdt-0.6b-v3"

    output_path = resolve_output_dir(repo_id, out_dir)
    model = load_model(repo_id)

    if not isinstance(model, EncDecRNNTBPEModel):
        raise TypeError(f"unsupported model type: {type(model).__name__}")

    temp_base = Path(tempfile.gettempdir()) / "melops" / "export"
    temp_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_base) as temp_dir:
        temp_path = Path(temp_dir)
        export_onnx_rnnt(model, temp_path, output_path)
        export_tokenizer(model, output_path)

    print(f"Completed export to {output_path}")
