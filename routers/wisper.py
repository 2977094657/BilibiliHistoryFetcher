from dataclasses import dataclass
import base64
from typing import Tuple

# third party libs
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

@dataclass
class WhisperModel:
    encoder_model_path: str
    decoder_model_path: str
    tokens_file_path: str
    language: str
    task: str
    device: str

    def __post_init__(self):
        print("init onnxruntime: ", ort.get_available_providers())

        self.session_opts = ort.SessionOptions()
        self.__init_encoder(self.encoder_model_path)
        self.__init_decoder(self.decoder_model_path)

    def transcribe(audio_path):
        return

    def get_state():
        return

    def __init_encoder(self, encoder_model_path: str):
        self.encoder = ort.InferenceSession(
            encoder_model_path,
            sess_options=self.session_opts,
            providers=ort.get_available_providers(),
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

    def __init_decoder(self, decoder_model_path: str):
        self.decoder = ort.InferenceSession(
            decoder_model_path,
            sess_options=self.session_opts,
            providers=ort.get_available_providers(),
        )

    def __run_encoder(
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

    def __run_decoder(
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

    def __get_self_cache(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def __suppress_tokens(self, logits, is_initial: bool) -> None:
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

    def __detect_language(
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
