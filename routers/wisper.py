from dataclasses import dataclass
import base64
from typing import Tuple,Union,Optional

# third party libs
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
from pydub import AudioSegment
import librosa

@dataclass
class WhisperModel:
    encoder: Union[str, bytes]
    decoder: Union[str, bytes]
    tokens_file: str
    language: Optional[str] = "zh"
    task: Optional[str] = "transcribe"
    device: str = "cpu"
    model_size: Optional[str] = "tiny"

    def __post_init__(self):
        print("init onnxruntime: ", ort.get_available_providers())

        self.session_opts = ort.SessionOptions()
        self.__init_encoder()
        self.__init_decoder()

    def run(self, audio_path, language)->Tuple[list, dict]:
        self.language = language

        # 计算音频特征
        mel = compute_features(audio_path, dim=self.n_mels)

        # 运行编码器
        n_layer_cross_k, n_layer_cross_v = self.__run_encoder(mel)

        # 选择语言
        if self.language is not None:
            if self.is_multilingual is False and self.language != "en":
                raise ValueError("This model supports only English.")

            if self.language not in self.lang2id:
                raise ValueError("Invalid language")

            # [sot, lang, task, notimestamps]
            self.sot_sequence[1] = self.lang2id[self.language]
        elif self.is_multilingual is True:
            print("detecting language")
            lang = self.__detect_language(n_layer_cross_k, n_layer_cross_v)
            self.sot_sequence[1] = lang
        
        # 选择任务
        if self.task is not None:
            if self.is_multilingual is False and self.task != "transcribe":
                raise ValueError("This model supports only English.")
            
            assert self.task in ["transcribe", "translate"], self.task

            if self.task == "translate":
                self.sot_sequence[2] = self.translate

        # 开辟KVCache空间
        n_layer_self_k_cache, n_layer_self_v_cache = self.__get_self_cache()

        # 运行解码器
        tokens = np.array([self.sot_sequence], dtype=np.int64)
        offset = np.zeros(1, dtype=np.int64)
        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.__run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        offset += len(self.sot_sequence)
        # logits.shape (batch_size, tokens.shape[1], vocab_size)
        logits = logits[0, -1]
        self.__suppress_tokens(logits, is_initial=True)
        #  logits = logits.softmax(dim=-1)
        # for greedy search, we don't need to compute softmax or log_softmax
        max_token_id = logits.argmax(axis=-1)
        results = []
        for i in range(self.n_text_ctx):
            if max_token_id == self.eot:
                break
            results.append(max_token_id.item())
            tokens = np.array([[results[-1]]])

            logits, n_layer_self_k_cache, n_layer_self_v_cache = self.__run_decoder(
                tokens=tokens,
                n_layer_self_k_cache=n_layer_self_k_cache,
                n_layer_self_v_cache=n_layer_self_v_cache,
                n_layer_cross_k=n_layer_cross_k,
                n_layer_cross_v=n_layer_cross_v,
                offset=offset,
            )
            offset += 1
            logits = logits[0, -1]
            self.__suppress_tokens(logits, is_initial=False)
            max_token_id = logits.argmax(axis=-1)

        # 解码token
        token_table = load_tokens(self.tokens_file)
        s = b""
        for i in results:
            if i in token_table:
                s += base64.b64decode(token_table[i])

        #print(s.decode().strip())

        return s.decode().strip(), {"language":self.language,"duration":float(mel.shape[-1]/10)}

    def get_state():
        return

    def __init_encoder(self):
        self.encoder = ort.InferenceSession(
            self.encoder,
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

    def __init_decoder(self):
        self.decoder = ort.InferenceSession(
            self.decoder,
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

def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens

def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    audio = AudioSegment.from_file(filename,
                                    filename.split(".")[-1],
                                   )
    audio = audio.set_channels(1)
    print(audio.sample_width)
    audio_data = audio.get_array_of_samples()
    data = np.frombuffer(audio_data, dtype=np.int16)
    data = data.astype(np.float32) / 32767.0

    return data, audio.frame_rate

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