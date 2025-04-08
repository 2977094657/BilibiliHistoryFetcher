# download model from https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html
# pip install -U huggingface_hub
# $env:HF_ENDPOINT = "https://hf-mirror.com"
# huggingface-cli download --resume-download csukuangfj/sherpa-onnx-whisper-base --local-dir sherpa-onnx-whisper-base

# https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/whisper/test.py
# https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py

# onnxruntime: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

# pip install sherpa-onnx==1.11.1+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
# pip install sherpa-onnx (cpu only)


# req: numpy, soundfile, kaldi_native_fbank, onnxruntime
# optional: librosa
# pip install numpy soundfile kaldi_native_fbank librosa onnxruntime-directml

# test command: python test_onnx_runner.py --encoder sherpa-onnx-whisper-base\base-encoder.onnx --decoder sherpa-onnx-whisper-base\base-decoder.onnx --tokens sherpa-onnx-whisper-base\base-tokens.txt --language zh --task transcribe test.wav 

# package: pyinstaller
# pip install pyinstaller
# pyinstaller test_onnx_runner.py --hidden-import=numpy.core.multiarray --noconfirm
# ./test_onnx_runner.exe --encoder sherpa-onnx-whisper-base\base-encoder.onnx --decoder sherpa-onnx-whisper-base\base-decoder.onnx --tokens sherpa-onnx-whisper-base\base-tokens.txt --language zh --task transcribe test.wav 

# package: nuitka
# pip install nuitka
# python -m nuitka --mode=standalone --noinclude-numba-mode=nofollow test_onnx_runner.py
# ./test_onnx_runner.exe --encoder sherpa-onnx-whisper-base\base-encoder.onnx --decoder sherpa-onnx-whisper-base\base-decoder.onnx --tokens sherpa-onnx-whisper-base\base-tokens.txt --language zh --task transcribe test.wav

import argparse
import base64
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort #FIXME: 打包时导致torch引入
import soundfile as sf
import librosa
from tqdm import tqdm

print("init onnxruntime: ",ort.get_available_providers())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to the tokens",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="""The actual spoken language in the audio.
        Example values, en, de, zh, jp, fr.
        If None, we will detect the language using the first 30s of the
        input audio
        """,
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        type=str,
        default="transcribe",
        help="Valid values are: transcribe, translate",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="Path to the test wave",
    )
    return parser.parse_args()


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        session_opts = ort.SessionOptions()
        # doc: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
        # session_opts.inter_op_num_threads = 1
        # session_opts.intra_op_num_threads = 0 

        # session_opts.log_severity_level = 1 # onnx debug log

        self.session_opts = session_opts

        self.init_encoder(encoder)
        self.init_decoder(decoder)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=ort.get_available_providers()
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.n_mels = int(meta["n_mels"])
        self.sot = int(meta["sot"])
        self.eot = int(meta["eot"])
        self.translate = int(meta["translate"])
        self.transcribe = int(meta["transcribe"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.no_speech = int(meta["no_speech"])
        self.blank = int(meta["blank_id"])

        self.sot_sequence = list(map(int, meta["sot_sequence"].split(",")))
        self.sot_sequence.append(self.no_timestamps)

        self.all_language_tokens = list(
            map(int, meta["all_language_tokens"].split(","))
        )
        self.all_language_codes = meta["all_language_codes"].split(",")
        self.lang2id = dict(zip(self.all_language_codes, self.all_language_tokens))
        self.id2lang = dict(zip(self.all_language_tokens, self.all_language_codes))

        self.is_multilingual = int(meta["is_multilingual"]) == 1

    def init_decoder(self, decoder: str):
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=ort.get_available_providers(),
        )

    def run_encoder(
        self,
        mel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_layer_cross_k, n_layer_cross_v = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: mel,
            },
        )
        return n_layer_cross_k, n_layer_cross_v

    def run_decoder(
        self,
        tokens: np.ndarray,
        n_layer_self_k_cache: np.ndarray,
        n_layer_self_v_cache: np.ndarray,
        n_layer_cross_k: np.ndarray,
        n_layer_cross_v: np.ndarray,
        offset: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
            ],
            {
                self.decoder.get_inputs()[0].name: tokens,
                self.decoder.get_inputs()[1].name: n_layer_self_k_cache,
                self.decoder.get_inputs()[2].name: n_layer_self_v_cache,
                self.decoder.get_inputs()[3].name: n_layer_cross_k,
                self.decoder.get_inputs()[4].name: n_layer_cross_v,
                self.decoder.get_inputs()[5].name: offset,
            },
        )
        return (
            logits,
            out_n_layer_self_k_cache,
            out_n_layer_self_v_cache,
        )

    def get_self_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = 1
        n_layer_self_k_cache = np.zeros(
            (self.n_text_layer,
            batch_size,
            self.n_text_ctx,
            self.n_text_state,),
            dtype=np.float32,
        )
        n_layer_self_v_cache = np.zeros(
            (self.n_text_layer,
            batch_size,
            self.n_text_ctx,
            self.n_text_state,),
            dtype=np.float32,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache

    def suppress_tokens(self, logits, is_initial: bool) -> None:
        # suppress blank
        if is_initial:
            logits[self.eot] = float("-inf")
            logits[self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[self.no_timestamps] = float("-inf")

        logits[self.sot] = float("-inf")
        logits[self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[self.translate] = float("-inf")

    def detect_language(
        self, n_layer_cross_k: np.ndarray, n_layer_cross_v: np.ndarray
    ) -> int:
        tokens = np.array([[self.sot]], dtype=np.int64)
        offset = np.zeros(1, dtype=np.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_self_cache()

        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        logits = logits.reshape(-1)
        mask = np.ones(logits.shape[0], dtype=np.int64)
        mask[self.all_language_tokens] = 0
        logits[mask != 0] = float("-inf")
        lang_id = logits.argmax().item()
        print("detected language: ", self.id2lang[lang_id])
        return lang_id


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str, dim: int = 80) -> np.ndarray:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, wave)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        # f = torch.from_numpy(f)
        features.append(f)

    features = np.stack(features)

    log_spec = np.log10(np.clip(features, min=1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    #mel = torch.nn.functional.pad(mel, (0, 0, 0, 1500), "constant", 0)
    mel = np.pad(mel, pad_width=((0, 1500), (0, 0)), mode='constant', constant_values=0)
    # Note that if it throws for a multilingual model,
    # please use a larger value, say 300

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[: target - 50]
        #mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)
        mel = np.pad(mel, pad_width=((0, 50), (0, 0)), mode='constant', constant_values=0)

    # We don't need to pad it to 30 seconds now!
    #  mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[0]), "constant", 0)

    #mel = mel.T.unsqueeze(0)
    mel = np.expand_dims(mel.T, axis=0)

    return mel


def main():
    args = get_args()

    model = OnnxModel(args.encoder, args.decoder)
    n_mels = model.n_mels

    mel = compute_features(args.sound_file, dim=n_mels)

    n_layer_cross_k, n_layer_cross_v = model.run_encoder(mel)

    if args.language is not None:
        if model.is_multilingual is False and args.language != "en":
            print(f"This model supports only English. Given: {args.language}")
            return

        if args.language not in model.lang2id:
            print(f"Invalid language: {args.language}")
            print(f"Valid values are: {list(model.lang2id.keys())}")
            return

        # [sot, lang, task, notimestamps]
        model.sot_sequence[1] = model.lang2id[args.language]
    elif model.is_multilingual is True:
        print("detecting language")
        lang = model.detect_language(n_layer_cross_k, n_layer_cross_v)
        model.sot_sequence[1] = lang

    if args.task is not None:
        if model.is_multilingual is False and args.task != "transcribe":
            print("This model supports only English. Please use --task=transcribe")
            return
        assert args.task in ["transcribe", "translate"], args.task

        if args.task == "translate":
            model.sot_sequence[2] = model.translate

    n_layer_self_k_cache, n_layer_self_v_cache = model.get_self_cache()

    print(model.sot_sequence)
    tokens = np.array([model.sot_sequence], dtype=np.int64)
    offset = np.zeros(1, dtype=np.int64)
    logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
        tokens=tokens,
        n_layer_self_k_cache=n_layer_self_k_cache,
        n_layer_self_v_cache=n_layer_self_v_cache,
        n_layer_cross_k=n_layer_cross_k,
        n_layer_cross_v=n_layer_cross_v,
        offset=offset,
    )
    offset += len(model.sot_sequence)
    # logits.shape (batch_size, tokens.shape[1], vocab_size)
    logits = logits[0, -1]
    model.suppress_tokens(logits, is_initial=True)
    #  logits = logits.softmax(dim=-1)
    # for greedy search, we don't need to compute softmax or log_softmax
    max_token_id = logits.argmax(axis=-1)
    results = []
    for i in tqdm(range(model.n_text_ctx)):
        if max_token_id == model.eot:
            break
        results.append(max_token_id.item())
        tokens = np.array([[results[-1]]])

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        offset += 1
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=False)
        max_token_id = logits.argmax(axis=-1)
    token_table = load_tokens(args.tokens)
    s = b""
    for i in results:
        if i in token_table:
            s += base64.b64decode(token_table[i])

    print(s.decode().strip())


if __name__ == "__main__":
    main()