# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

from .qwen3_forced_aligner import Qwen3ForcedAligner
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    chunk_list,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except:
    pass


@dataclass
class ASRTranscription:
    """
    One transcription result.

    Attributes:
        language (str):
            Merged language string for the sample, e.g. "Chinese" or "Chinese,English".
            Empty string if unknown or silent audio.
        text (str):
            Transcribed text.
        time_stamps (Optional[Any]):
            Forced aligner output (ForcedAlignResult).
            Present only when return_time_stamps=True.
    """
    language: str
    text: str
    time_stamps: Optional[Any] = None


class Qwen3ASRModel:
    """
    Unified inference wrapper for Qwen3-ASR with two backends:
      - Transformers backend 
      - vLLM backend

    It optionally supports time stamp output via Qwen3-ForcedAligner.

    Notes:
      - Each request uses a context text and exactly one audio.
      - If language is provided, the prompt will force the output to be text-only by appending
        "language {Language}<asr_text>" to the assistant prompt.
    """

    def __init__(
        self,
        backend: str,
        model: Any,
        processor: Any,
        sampling_params: Optional[Any] = None,
        forced_aligner: Optional[Qwen3ForcedAligner] = None,
        max_inference_batch_size: int = -1,
    ):
        self.backend = backend  # "transformers" | "vllm"
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.forced_aligner = forced_aligner
        self.max_inference_batch_size = int(max_inference_batch_size)

        if backend == "transformers":
            self.device = getattr(model, "device", None)
            if self.device is None:
                try:
                    self.device = next(model.parameters()).device
                except StopIteration:
                    self.device = torch.device("cpu")
            self.dtype = getattr(model, "dtype", torch.float32)
        else:
            self.device = None
            self.dtype = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using Transformers backend.

        Args:
            pretrained_model_name_or_path:
                HuggingFace repo id or local directory.
            forced_aligner:
                Optional forced aligner model path/repo id.
            forced_aligner_kwargs:
                Optional kwargs forwarded to Qwen3ForcedAligner.from_pretrained(...).
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            **kwargs:
                Forwarded to AutoModel.from_pretrained(...).

        Returns:
            Qwen3ASRModel
        """

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)

        forced_aligner_model = None
        if forced_aligner is not None:
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="transformers",
            model=model,
            processor=processor,
            sampling_params=None,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
        )

    @classmethod
    def LLM(
        cls,
        model: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: Optional[int] = 8192,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using vLLM backend.

        Import is isolated to keep vLLM optional.

        Args:
            model:
                Model path/repo for vLLM.
            forced_aligner:
                Optional forced aligner model path/repo id.
            forced_aligner_kwargs:
                Optional kwargs forwarded to Qwen3ForcedAligner.from_pretrained(...).
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            max_new_tokens:
                Maximum number of tokens to generate.
            **kwargs:
                Forwarded to vllm.LLM(...).

        Returns:
            Qwen3ASRModel

        Raises:
            ImportError: If vLLM is not installed.
        """
        try:
            from vllm import LLM as vLLM
            from vllm import SamplingParams
        except Exception as e:
            raise ImportError(
                "vLLM is not available. Install with: pip install qwen-asr[vllm]"
            ) from e

        llm = vLLM(model=model, **kwargs)

        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(**({"temperature": 0.0, "max_tokens": max_new_tokens}))
        forced_aligner_model = None
        if forced_aligner is not None:
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="vllm",
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
        )

    def get_supported_languages(self) -> List[str]:
        """
        Returns the supported language list.

        Returns:
            List[str]: Canonical language names.
        """
        return list(SUPPORTED_LANGUAGES)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        Transcribe audio with optional context and optional forced alignment timestamps.

        Args:
            audio:
                Audio input(s). Supported:
                  - str: local path / URL / base64 data url
                  - (np.ndarray, sr)
                  - list of above
            context:
                Context string(s). If scalar, it will be broadcast to batch size.
            language:
                Optional language(s). If provided, it must be in supported languages.
                If scalar, it will be broadcast to batch size.
                If provided, the prompt will force output to be transcription text only.
            return_time_stamps:
                If True, timestamps are produced via forced aligner and merged across chunks.
                This requires forced_aligner initialized.

        Returns:
            List[ASRTranscription]: One result per input audio.

        Raises:
            ValueError:
                - If return_time_stamps=True but forced_aligner is not provided.
                - If language is unsupported.
                - If batch sizes mismatch for context/language.
        """
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("return_time_stamps=True requires `forced_aligner` to be provided at initialization.")

        wavs = normalize_audios(audio)
        n = len(wavs)

        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

        # chunk audios and record mapping
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # run ASR on chunks
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        # parse outputs, prepare for optional alignment
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)

        # forced alignment (optional)
        per_chunk_align: List[Optional[Any]] = [None] * len(chunks)
        if return_time_stamps:
            to_align_audio = []
            to_align_text = []
            to_align_lang = []
            to_align_idx = []

            for idx, (c, txt, lang_pred) in enumerate(zip(chunks, per_chunk_text, per_chunk_lang)):
                if txt.strip() == "":
                    continue
                to_align_audio.append((c.wav, c.sr))
                to_align_text.append(txt)
                to_align_lang.append(lang_pred)
                to_align_idx.append(idx)

            # batch align with max_inference_batch_size
            aligned_results: List[Any] = []
            for a_chunk, t_chunk, l_chunk in zip(
                chunk_list(to_align_audio, self.max_inference_batch_size),
                chunk_list(to_align_text, self.max_inference_batch_size),
                chunk_list(to_align_lang, self.max_inference_batch_size),
            ):
                aligned_results.extend(
                    self.forced_aligner.align(audio=a_chunk, text=t_chunk, language=l_chunk)
                )

            # offset fix
            for k, idx in enumerate(to_align_idx):
                c = chunks[idx]
                r = aligned_results[k]
                per_chunk_align[idx] = self._offset_align_result(r, c.offset_sec)

        # merge chunks back to original samples
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        out_aligns: List[List[Any]] = [[] for _ in range(n)]

        for c, lang, txt, al in zip(chunks, per_chunk_lang, per_chunk_text, per_chunk_align):
            out_langs[c.orig_index].append(lang)
            out_texts[c.orig_index].append(txt)
            if return_time_stamps and al is not None:
                out_aligns[c.orig_index].append(al)

        results: List[ASRTranscription] = []
        for i in range(n):
            merged_text = "".join([t for t in out_texts[i] if t is not None])
            merged_language = merge_languages(out_langs[i])
            merged_align = None
            if return_time_stamps:
                merged_align = self._merge_align_results(out_aligns[i])
            results.append(ASRTranscription(language=merged_language, text=merged_text, time_stamps=merged_align))

        return results

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        Build the string prompt for one request.

        If force_language is provided, "language X<asr_text>" is appended after the generation prompt
        to request text-only output.
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        Run backend inference for chunk-level items.

        Args:
            contexts: List of system context strings.
            wavs: List of mono waveforms (np.ndarray).
            languages: List of forced languages or None.

        Returns:
            List[str]: Raw decoded strings (one per chunk).
        """
        if self.backend == "transformers":
            return self._infer_asr_transformers(contexts, wavs, languages)
        if self.backend == "vllm":
            return self._infer_asr_vllm(contexts, wavs, languages)
        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _infer_asr_transformers(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        outs: List[str] = []

        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            text_ids = self.model.generate(**inputs)

            decoded = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))

        return outs

    def _infer_asr_vllm(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        inputs: List[Dict[str, Any]] = []
        for c, w, fl in zip(contexts, wavs, languages):
            prompt = self._build_text_prompt(context=c, force_language=fl)
            inputs.append({"prompt": prompt, "multi_modal_data": {"audio": [w]}})

        outs: List[str] = []
        for batch in chunk_list(inputs, self.max_inference_batch_size):
            outputs = self.model.generate(batch, sampling_params=self.sampling_params, use_tqdm=False)
            for o in outputs:
                outs.append(o.outputs[0].text)
        return outs

    def _offset_align_result(self, result: Any, offset_sec: float) -> Any:
        """
        Apply time offset to a ForcedAlignResult-like object.

        This function assumes:
          - result has attribute `.items` which is a list of items with start_time/end_time in seconds.
          - dataclasses are frozen in upstream implementation, so we reconstruct by type.

        Args:
            result: ForcedAlignResult
            offset_sec: Offset in seconds

        Returns:
            ForcedAlignResult: New object with shifted timestamps.
        """
        if result is None:
            return None
        items = []
        for it in result.items:
            items.append(type(it)(text=it.text, 
                                  start_time=round(it.start_time + offset_sec, 3), 
                                  end_time=round(it.end_time + offset_sec, 3)))
        return type(result)(items=items)

    def _merge_align_results(self, results: List[Any]) -> Optional[Any]:
        """
        Merge multiple ForcedAlignResult objects into a single one by concatenating items.

        Args:
            results: List of ForcedAlignResult

        Returns:
            ForcedAlignResult or None
        """
        if not results:
            return None
        all_items = []
        for r in results:
            if r is None:
                continue
            all_items.extend(list(r.items))
        if not all_items:
            return None
        return type(results[0])(items=all_items)
