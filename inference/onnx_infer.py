import json
import os
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import librosa
import numpy as np
import onnxruntime as ort

try:
    from .ort_utils import (
        run_session as _ort_run_session,
        create_session as _ort_create_session,
        is_device_lost as _ort_is_device_lost,
    )
except ImportError:  # 作为脚本直接运行时（python onnx_infer.py ...）
    from ort_utils import (
        run_session as _ort_run_session,
        create_session as _ort_create_session,
        is_device_lost as _ort_is_device_lost,
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def median_abs_deviation(x, axis=0, center=np.median, scale=1.0):
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            scale = 0.6744897501960817
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = np.asarray(x)

    if x.size == 0:
        if axis is None:
            return np.nan
        if axis is not None:
            nan_shape = list(x.shape)
            del nan_shape[axis]
            nan_shape = tuple(nan_shape)
            if nan_shape == ():
                return np.nan
            return np.full(nan_shape, np.nan)
        return np.nan

    contains_nan = np.isnan(x).any()

    if contains_nan:
        if axis is None:
            return np.nan
        else:
            result_shape = list(x.shape)
            if axis is not None:
                del result_shape[axis]
            result = np.full(tuple(result_shape), np.nan)
            return result / scale
    else:
        if axis is None:
            med = center(x)
            mad = np.median(np.abs(x - med))
        else:
            med = center(x, axis=axis)
            med_expanded = np.expand_dims(med, axis=axis)
            mad = np.median(np.abs(x - med_expanded), axis=axis)

    return mad / scale


def remove_outliers_per_position(data_series, threshold=1.5):
    processed_values = []
    for position_values in data_series:
        if not position_values:
            processed_values.append(0.0)
            continue
        med = np.median(position_values)
        mad_val = median_abs_deviation(position_values)

        if mad_val == 0:
            processed_values.append(med)
            continue

        z_scores = np.abs((np.array(position_values) - med) / (mad_val * 1.4826))

        retained_values = []
        filtered_out = []
        for x, z in zip(position_values, z_scores):
            if z <= threshold:
                retained_values.append(x)
            else:
                filtered_out.append(x)

        if len(retained_values) > 0:
            final_value = np.mean(retained_values)
        else:
            final_value = med

        processed_values.append(final_value)
    return processed_values


@dataclass
class Phoneme:
    start: float
    end: float
    text: str

    def __init__(self, start: float, end: float, text: str):
        self.start = max(0.0, start)
        self.end = end
        self.text = text

        if not (self.start < self.end):
            raise ValueError(f"Phoneme Invalid: text={self.text} start={self.start}, end={self.end}")

    @property
    def dur(self) -> float:
        """音素时长（秒）"""
        return self.end - self.start


@dataclass
class Word:
    start: float
    end: float
    text: str
    phonemes: List[Phoneme] = field(default_factory=list)

    def __init__(self, start: float, end: float, text: str, init_phoneme: bool = False):
        self.start = max(0.0, start)
        self.end = end
        self.text = text

        if not (self.start < self.end):
            raise ValueError(f"Word Invalid: text={self.text} start={self.start}, end={self.end}")

        self.phonemes: List[Phoneme] = []
        if init_phoneme:
            self.phonemes.append(Phoneme(self.start, self.end, self.text))

    @property
    def dur(self) -> float:
        """词时长（秒）"""
        return self.end - self.start

    def add_phoneme(self, phoneme, log_list: list = None):
        if phoneme.start == phoneme.end:
            warning_msg = f"{phoneme.text} phoneme长度为0，非法"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)
            return
        if phoneme.start >= self.start and phoneme.end <= self.end:
            self.phonemes.append(phoneme)
        else:
            warning_msg = f"{phoneme.text}: phoneme边界超出word，添加失败"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)

    def append_phoneme(self, phoneme, log_list: list = None):
        if phoneme.start == phoneme.end:
            warning_msg = f"{phoneme.text} phoneme长度为0，非法"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)
            return
        if len(self.phonemes) == 0:
            if phoneme.start == self.start:
                self.phonemes.append(phoneme)
                self.end = phoneme.end
            else:
                warning_msg = f"{phoneme.text}: phoneme左边界超出word，添加失败"
                if log_list is not None:
                    log_list.append(f"WARNING: {warning_msg}")
                else:
                    warnings.warn(warning_msg)
        else:
            if phoneme.start == self.phonemes[-1].end:
                self.phonemes.append(phoneme)
                self.end = phoneme.end
            else:
                warning_msg = f"{phoneme.text}: phoneme添加失败"
                if log_list is not None:
                    log_list.append(f"WARNING: {warning_msg}")
                else:
                    warnings.warn(warning_msg)

    def move_start(self, new_start, log_list: list = None):
        if 0 <= new_start < self.phonemes[0].end:
            self.start = new_start
            self.phonemes[0].start = new_start
        else:
            warning_msg = f"{self.text}: start >= first_phone_end，无法调整word边界"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)

    def move_end(self, new_end, log_list: list = None):
        if new_end > self.phonemes[-1].start >= 0:
            self.end = new_end
            self.phonemes[-1].end = new_end
        else:
            warning_msg = f"{self.text}: new_end <= first_phone_start，无法调整word边界"
            if log_list is not None:
                log_list.append(f"WARNING: {warning_msg}")
            else:
                warnings.warn(warning_msg)


class WordList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self._log = []

    def _add_log(self, message: str):
        self._log.append(message)

    def log(self) -> str:
        return "\n".join(self._log)

    def clear_log(self):
        self._log.clear()

    def overlapping_words(self, new_word: Word):
        overlapping_words = []
        for word in self:
            if not isinstance(word, Word):
                continue
            if not (new_word.end <= word.start or new_word.start >= word.end):
                overlapping_words.append(word)
        return overlapping_words

    def append(self, word: Word):
        if len(word.phonemes) == 0:
            warning_msg = f"{word}: phones为空，非法word"
            self._add_log(f"WARNING: {warning_msg}")
            return

        if len(self) == 0:
            super().append(word)
            return

        if not self.overlapping_words(word):
            super().append(word)
        else:
            warning_msg = f"{word}: 区间重叠，无法添加word"
            self._add_log(f"WARNING: {warning_msg}")

    @staticmethod
    def remove_overlapping_intervals(raw_interval, remove_interval):
        r_start, r_end = raw_interval
        m_start, m_end = remove_interval

        if not (r_start < r_end):
            raise ValueError(f"raw_interval.start must be smaller than raw_interval.end")
        if not (m_start < m_end):
            raise ValueError(f"remove_interval.start must be smaller than remove_interval.end")

        overlap_start = max(r_start, m_start)
        overlap_end = min(r_end, m_end)

        if overlap_start >= overlap_end:
            return [raw_interval]

        result = []
        if r_start < overlap_start:
            result.append((r_start, overlap_start))

        if overlap_end < r_end:
            result.append((overlap_end, r_end))

        return result

    def add_AP(self, new_word: Word, min_dur=0.1):
        try:
            if len(new_word.phonemes) == 0:
                warning_msg = f"{new_word.text} phonemes为空，非法word"
                self._add_log(f"WARNING: {warning_msg}")
                return

            if len(self) == 0:
                self.append(new_word)
                return

            overlapping = self.overlapping_words(new_word)
            if not overlapping:
                self.append(new_word)
                self.sort(key=lambda w: w.start)
                return

            ap_intervals = [(new_word.start, new_word.end)]
            for word in self:
                temp_res = []
                for _ap in ap_intervals:
                    temp_res.extend(self.remove_overlapping_intervals(_ap, (word.start, word.end)))
                ap_intervals = temp_res
            ap_intervals = [_ap for _ap in ap_intervals if _ap[1] - _ap[0] >= min_dur]

            for _ap in ap_intervals:
                try:
                    self.append(Word(_ap[0], _ap[1], new_word.text, True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")
                    continue

            self.sort(key=lambda w: w.start)
        except Exception as e:
            self._add_log(f"ERROR in add_AP: {e}")

    def fill_small_gaps(self, wav_length: float, gap_length: float = 0.1):
        try:
            if self[0].start < 0:
                self[0].start = 0

            if 0 < self[0].start < gap_length < self[0].dur:
                self[0].move_start(0, self._log)

            if self[-1].end >= wav_length - gap_length:
                self[-1].move_end(wav_length, self._log)

            for i in range(1, len(self)):
                if 0 < self[i].start - self[i - 1].end <= gap_length:
                    self[i - 1].move_end(self[i].start, self._log)
        except Exception as e:
            self._add_log(f"ERROR in fill_small_gaps: {e}")

    def merge_duplicate_phonemes(self, min_duration=0.05):
        """
        合并相邻的同名音素，当音素自身时长过小时
        将两个同名音素的总时长平分，消除中间的间隙
        注意：同名音素可能跨越不同的word
        """
        try:
            # 收集所有音素
            all_phonemes = []
            for word in self:
                all_phonemes.extend(word.phonemes)
            
            # 按时间排序
            all_phonemes.sort(key=lambda ph: ph.start)
            
            # 检查相邻音素
            i = 0
            while i < len(all_phonemes) - 1:
                current_ph = all_phonemes[i]
                next_ph = all_phonemes[i + 1]
                
                # 检查是否同名
                if current_ph.text == next_ph.text:
                    # 检查两个音素中是否有任何一个的时长小于阈值
                    current_duration = current_ph.end - current_ph.start
                    next_duration = next_ph.end - next_ph.start
                    
                    if current_duration < min_duration or next_duration < min_duration:
                        # 计算总时长（包括间隙）
                        total_duration = next_ph.end - current_ph.start
                        
                        # 计算新的边界，将总时长平分
                        mid_point = current_ph.start + total_duration / 2
                        
                        # 更新两个音素的边界
                        current_ph.end = mid_point
                        next_ph.start = mid_point
                        
                        self._add_log(f"INFO: 平分同名音素 '{current_ph.text}' 时长: {current_ph.start}-{mid_point} 和 {mid_point}-{next_ph.end}")
                
                i += 1

            # 更新word的边界
            for word in self:
                if word.phonemes:
                    word.start = word.phonemes[0].start
                    word.end = word.phonemes[-1].end

        except Exception as e:
            self._add_log(f"ERROR in merge_duplicate_phonemes: {e}")

    def add_SP(self, wav_length, add_phone="SP", min_dur=1e-3):
        try:
            words_res = WordList()
            words_res._log = self._log

            if self[0].start > min_dur:
                try:
                    words_res.append(Word(0, self[0].start, add_phone, init_phoneme=True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")

            words_res.append(self[0])
            for i in range(1, len(self)):
                word = self[i]
                if word.start - words_res[-1].end > min_dur:
                    try:
                        words_res.append(Word(words_res[-1].end, word.start, add_phone, init_phoneme=True))
                    except ValueError as e:
                        self._add_log(f"ERROR: {e}")
                words_res.append(word)

            if wav_length - self[-1].end > min_dur:
                try:
                    words_res.append(Word(self[-1].end, wav_length, add_phone, init_phoneme=True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")

            self.clear()
            self.extend(words_res)
            self.check()
        except Exception as e:
            self._add_log(f"ERROR in add_SP: {e}")

    @property
    def phonemes(self):
        phonemes = []
        for word in self:
            phonemes.extend([ph.text for ph in word.phonemes])
        return phonemes

    @property
    def intervals(self):
        return [[word.start, word.end] for word in self]

    def clear_language_prefix(self):
        for word in self:
            for phoneme in word.phonemes:
                phoneme.text = phoneme.text.split("/")[-1]

    def check(self):
        if len(self) == 0:
            return True

        for i, word in enumerate(self):
            if not isinstance(word, Word):
                warning_msg = f"Element at index {i} is not a Word instance"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if not (word.start < word.end):
                warning_msg = f"Word '{word.text}' has invalid time order: start={word.start}, end={word.end}"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if len(word.phonemes) == 0:
                warning_msg = f"Word '{word.text}' has no phonemes"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if word.phonemes[0].start != word.start:
                warning_msg = f"Word '{word.text}' first phoneme start({word.phonemes[0].start}) != word start({word.start})"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            if word.phonemes[-1].end != word.end:
                warning_msg = f"Word '{word.text}' last phoneme end({word.phonemes[-1].end}) != word end({word.end})"
                self._add_log(f"WARNING: {warning_msg}")
                return False

            for j in range(len(word.phonemes)):
                if not (word.phonemes[j].start < word.phonemes[j].end):
                    warning_msg = f"Word '{word.text}' phoneme '{word.phonemes[j].text}' has invalid time order: start={word.phonemes[j].start}, end={word.phonemes[j].end}"
                    self._add_log(f"WARNING: {warning_msg}")
                    return False

                if j < len(word.phonemes) - 1 and word.phonemes[j].end != word.phonemes[j + 1].start:
                    warning_msg = f"Word '{word.text}' phoneme '{word.phonemes[j].text}' end({word.phonemes[j].end}) != next phoneme '{word.phonemes[j + 1].text}' start({word.phonemes[j + 1].start})"
                    self._add_log(f"WARNING: {warning_msg}")
                    return False

        for i in range(len(self) - 1):
            if self[i].end != self[i + 1].start:
                warning_msg = f"Word '{self[i].text}' end({self[i].end}) != next word '{self[i + 1].text}' start({self[i + 1].start})"
                self._add_log(f"WARNING: {warning_msg}")
                return False

        return True


class AlignmentDecoder:
    def __init__(self, vocab, sample_rate, hop_size):
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_length = hop_size / sample_rate
        self.T = self.ph_seq_id = self.ph_idx_seq = self.ph_frame_pred = self.ph_time_int_pred = None
        self.ph_seq_pred = self.ph_intervals_pred = self.edge_prob = self.pred_words = self.frame_confidence = None

    def decode(self,
               ph_frame_logits,
               ph_edge_logits,
               wav_length: float,
               ph_seq: list[str],
               word_seq: list[str] = None,
               ph_idx_to_word_idx: list[int] = None,
               ignore_sp: bool = True,
               ):
        ph_frame_logits = ph_frame_logits[0]
        ph_edge_logits = ph_edge_logits[0]
        
        ph_seq_id_list = []
        warning_log = []
        for ph in ph_seq:
            if ph in self.vocab["vocab"]:
                ph_seq_id_list.append(self.vocab["vocab"][ph])
            else:
                # 用 SP(id=0) 占位推理，输出时仍保留原音素名
                ph_seq_id_list.append(self.vocab["vocab"]["SP"])
                warning_msg = f"音素 '{ph}' 不在模型词汇表中，推理时用 'SP' 占位，输出仍保留原音素 '{ph}'"
                warning_log.append(warning_msg)
                warnings.warn(warning_msg)
        
        ph_seq_id = np.array(ph_seq_id_list)
        self.ph_seq_id = ph_seq_id

        ph_mask = np.full(self.vocab["vocab_size"], 1e9)
        ph_mask[ph_seq_id], ph_mask[0] = 0, 0

        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        num_frames = int((wav_length * self.sample_rate + 0.5) / self.hop_size)
        ph_frame_logits, ph_edge_logits = ph_frame_logits[:, :num_frames], ph_edge_logits[:num_frames]

        ph_frame_logits_adjusted = ph_frame_logits - ph_mask[:, np.newaxis]
        ph_frame_pred = softmax(ph_frame_logits_adjusted, axis=0).astype("float32")
        ph_prob_log = log_softmax(ph_frame_logits_adjusted, axis=0).astype("float32")
        ph_edge_pred = np.clip(sigmoid(ph_edge_logits), 0.0, 1.0).astype("float32")
        self.ph_frame_pred = ph_frame_pred
        vocab_size, self.T = ph_frame_pred.shape

        edge_diff = np.concatenate((np.diff(ph_edge_pred, axis=0), [0]), axis=0)
        self.edge_prob = (ph_edge_pred + np.concatenate(([0], ph_edge_pred[:-1]))).clip(0, 1)

        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            ph_prob_log,
            self.edge_prob,
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        self.ph_idx_seq = ph_idx_seq
        self.ph_time_int_pred = np.array(ph_time_int_pred, dtype="int32")
        self.frame_confidence = frame_confidence

        ph_time_fractional = (edge_diff[self.ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = self.frame_length * (
            np.concatenate([self.ph_time_int_pred.astype("float32") + ph_time_fractional, [self.T]])
        )
        ph_time_pred = np.clip(ph_time_pred, 0, None)
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        word = None
        words: WordList = WordList()

        ph_seq_decoded = []
        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            ph_seq_decoded.append(ph_seq[ph_idx])
            if ph_seq[ph_idx] == 'SP' and ignore_sp:
                continue
            phoneme = Phoneme(ph_intervals[i, 0], ph_intervals[i, 1], ph_seq[ph_idx])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word.append_phoneme(phoneme)
            else:
                word = Word(ph_intervals[i, 0], ph_intervals[i, 1], word_seq[word_idx])
                word.add_phoneme(phoneme)
                words.append(word)
                word_idx_last = word_idx
        self.ph_seq_pred, self.ph_intervals_pred, self.pred_words = words.phonemes, words.intervals, words
        return words, total_confidence, warning_log

    @staticmethod
    def forward_pass(T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id, prob3_pad_len=2):
        backtrack_s = np.full_like(dp, -1, dtype=np.int32)
        edge_prob_log = np.log(edge_prob + 1e-6)
        not_edge_prob_log = np.log(1 - edge_prob + 1e-6)
        mask_reset = (ph_seq_id == 0)
        T_div_S = T / S  # 缓存比例因子

        # 预分配临时数组，避免每帧创建
        prob1 = np.empty(S, dtype=np.float32)
        prob2 = np.full(S, -np.inf, dtype=np.float32)
        prob3 = np.full(S, -np.inf, dtype=np.float32)
        stacked_probs = np.empty((3, S), dtype=np.float32)  # 复用，避免 np.vstack

        i_vals_prob3 = np.arange(prob3_pad_len, S)
        idx_arr = np.clip(i_vals_prob3 - prob3_pad_len + 1, 0, S - 1)
        mask_cond_prob3 = (idx_arr >= S - 1) | (ph_seq_id[idx_arr] == 0)
        # 预计算切片范围
        slice_pad = slice(0, S - prob3_pad_len)
        slice_skip1 = slice(0, S - 1)
        skip1_view = prob2[1:]  # 复用视图
        prob3_view = prob3[i_vals_prob3]

        for t in range(1, T):
            prob_log_t = prob_log[:, t]
            edge_log_t = edge_prob_log[t]
            not_edge_log_t = not_edge_prob_log[t]
            dp_prev = dp[:, t - 1]

            # prob1: 停留在当前音素
            prob1[:] = dp_prev
            prob1 += prob_log_t
            prob1 += not_edge_log_t

            # prob2: 前进1个音素
            np.add(dp_prev[slice_skip1], prob_log_t[slice_skip1], out=skip1_view)
            skip1_view += edge_log_t
            np.add(skip1_view, curr_ph_max_prob_log[slice_skip1] * T_div_S, out=skip1_view)

            # prob3: 跳过1个音素（前进2个）
            candidate_vals = dp_prev[slice_pad] + prob_log_t[slice_pad] + edge_log_t + curr_ph_max_prob_log[slice_pad] * T_div_S
            # 用布尔索引替代 np.where，减少中间数组
            prob3_view[:] = -np.inf
            prob3_view[mask_cond_prob3] = candidate_vals[mask_cond_prob3]

            # 复用 stacked_probs，避免 np.vstack 分配
            stacked_probs[0] = prob1
            stacked_probs[1] = prob2
            stacked_probs[2] = prob3
            max_indices = np.argmax(stacked_probs, axis=0)
            dp[:, t] = stacked_probs[max_indices, np.arange(S)]
            backtrack_s[:, t] = max_indices

            # 更新 curr_ph_max_prob_log
            mask_type0 = (max_indices == 0)
            np.maximum(curr_ph_max_prob_log, prob_log_t, out=curr_ph_max_prob_log, where=mask_type0)
            np.copyto(curr_ph_max_prob_log, prob_log_t, where=~mask_type0)
            curr_ph_max_prob_log[mask_reset] = 0.0

            # 重置 prob2/prob3 的边界
            skip1_view.fill(-np.inf)
            prob3_view.fill(-np.inf)
        return dp, backtrack_s, curr_ph_max_prob_log

    def _decode(self,
                ph_seq_id,
                ph_prob_log,
                edge_prob):
        vocab_size, T = ph_prob_log.shape
        S = len(ph_seq_id)
        prob_log = ph_prob_log[ph_seq_id, :]

        curr_ph_max_prob_log = np.full(S, -np.inf)
        dp = np.full((S, T), -np.inf, dtype="float32")

        dp[0, 0] = prob_log[0, 0]
        curr_ph_max_prob_log[0] = prob_log[0, 0]
        if ph_seq_id[0] == 0 and S > 1:
            dp[1, 0] = prob_log[1, 0]
            curr_ph_max_prob_log[1] = prob_log[1, 0]

        dp, backtrack_s, curr_ph_max_prob_log = self.forward_pass(
            T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id,
            prob3_pad_len=2 if S >= 2 else 1
        )

        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []

        if S == 1:
            s = 0
        else:
            s = S - 2 if dp[-2, -1] > dp[-1, -1] and ph_seq_id[-1] == 0 else S - 1

        # 用 while 循环替代 np.arange(T-1, -1, -1)，避免创建大数组
        t = T - 1
        while t >= 0:
            assert backtrack_s[s, t] >= 0 or t == 0
            frame_confidence.append(dp[s, t])

            bt = backtrack_s[s, t]
            if bt != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)
                if bt == 1:
                    s -= 1
                elif bt == 2:
                    s -= 2
            t -= 1

        ph_idx_seq.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(np.diff(np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1))
        return ph_idx_seq, ph_time_int, frame_confidence


class NonLexicalDecoder:
    def __init__(self, vocab, class_names: list[str], sample_rate: int, hop_size: int):
        self.vocab = vocab
        self.non_lexical_phs = class_names
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_length = hop_size / sample_rate
        self.cvnt_probs = None

    def decode(self,
               cvnt_logits,
               wav_length: float | None = None,
               non_lexical_phonemes: list[str] = None,
               ) -> list[WordList]:
        non_lexical_phonemes = non_lexical_phonemes or []
        if wav_length is not None:
            num_frames = int((wav_length * self.sample_rate + 0.5) / self.hop_size)
            cvnt_logits = cvnt_logits[:, :, :num_frames]
        self.cvnt_probs = softmax(cvnt_logits, axis=1)[0]

        non_lexical_words = []
        for ph in non_lexical_phonemes:
            i = self.non_lexical_phs.index(ph)
            tag_words: list[Word] = self.non_lexical_words(self.cvnt_probs[i], tag=ph)
            non_lexical_words.append(tag_words)
        return non_lexical_words

    def non_lexical_words(self, prob, threshold=0.5, max_gap=5, mix_frames=10, tag=""):
        words, start, gap_count = [], None, 0

        for i in range(len(prob)):
            if prob[i] >= threshold:
                if start is None:
                    start = i
                gap_count = 0
            elif start is not None:
                if gap_count < max_gap:
                    gap_count += 1
                else:
                    end = i - gap_count - 1
                    if end > start and (end - start) >= mix_frames:
                        word = Word(start * self.frame_length, end * self.frame_length, tag)
                        word.add_phoneme(Phoneme(start * self.frame_length, end * self.frame_length, tag))
                        words.append(word)
                    start, gap_count = None, 0

        if start is not None and (len(prob) - start) >= mix_frames:
            word = Word(start * self.frame_length, (len(prob) - 1) * self.frame_length, tag)
            word.add_phoneme(Phoneme(start * self.frame_length, (len(prob) - 1) * self.frame_length, tag))
            words.append(word)
        return words


class StopInference(Exception):
    """推理被用户手动停止时抛出。"""
    pass


def resolve_non_lexical(requested, supported):
    """解析「自动添加非语言音素」设置。

    Args:
        requested: 逗号分隔的音素名，如 ``"AP,SP"``；空字符串表示不添加。
        supported: 模型 vocab 中的 non_lexical_phonemes（如 ``['AP','EP']``）。

    Returns:
        ``(走 NLL 检测的音素, 是否用 SP 补全空白, 不被支持的音素)``。
    """
    names = [item.strip() for item in str(requested or "").split(",") if item.strip()]
    supported = {str(item) for item in (supported or [])}
    nll_names = [name for name in names if name in supported]
    unsupported = [name for name in names if name not in supported]
    add_sp = "SP" in unsupported
    return nll_names, add_sp, unsupported


class InferenceOnnx:
    def __init__(self, onnx_path: Path):
        self.model = None
        self.model_path = onnx_path
        self.model_folder = onnx_path.parent
        self.vocab = None
        self.mel_cfg = None
        self.vocab_folder = None
        self.nll_decoder = None
        self.fa_decoder = None
        self.dataset = []
        self.predictions = []
        self.failed_files = []  # [(wav_path, error_traceback)] 推理失败的音频
        self.device = 'cpu'     # 当前使用的推理设备（load_model 时写入）
        self._cpu_fallback = False  # 是否已因设备丢失而永久回退 CPU
        self._device_recover_count = 0  # 设备丢失后已尝试恢复的次数
        self.progress_callback = None

    def load_config(self):
        vocab_file = self.model_folder / 'vocab.json'
        config_file = self.model_folder / 'config.json'
        version_file = self.model_folder / 'VERSION'

        assert vocab_file.exists(), f"{vocab_file} does not exist"
        assert config_file.exists(), f"{config_file} does not exist"
        assert version_file.exists(), f"{version_file} does not exist"

        with open(version_file, 'r', encoding='utf-8') as f:
            assert int(f.readline().strip()) == 5, f"onnx model version must be 5."
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.loads(f.read())
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.loads(f.read())

        self.mel_cfg = config['mel_spec_config']
        self.vocab_folder = self.model_folder

    # 实际用到的输出。ctc_logits 没有任何解码器使用，不请求它可以让 ORT
    # 直接剪掉这条分支，省一次 144 类头的前向计算和对应的输出张量。
    _USED_OUTPUTS = ('ph_frame_logits', 'ph_edge_logits', 'cvnt_logits')

    def load_model(self, device='cpu'):
        self.device = (device or 'cpu').lower()
        self.model = self.create_session(self.model_folder / 'model.onnx', device=device)
        # 缓存输出名称，避免每次 run_onnx 都查询；且只保留解码器真正消费的几路
        available = [output.name for output in self.model.get_outputs()]
        wanted = [name for name in self._USED_OUTPUTS if name in available]
        self._output_names = wanted or available
        self.device_info = self._get_device_info()

    def _recover_from_device_lost(self) -> bool:
        """GPU 设备被驱动重置（DXGI_ERROR_DEVICE_REMOVED / TDR）后恢复推理。

        先用同一设备重建 session（D3D 设备会被重新创建）；若仍失败，
        则永久回退到 CPU，保证整批任务能跑完。返回 True 表示可以重试。
        """
        if self._cpu_fallback:
            return False

        current = (self.device or 'cpu').lower()
        targets = []
        # 第一次恢复重建同设备 session；若再次丢失则直接回退 CPU
        if current != 'cpu' and self._device_recover_count == 0:
            targets.append((current, False))
        targets.append(('cpu', True))
        self._device_recover_count += 1

        for target, is_fallback in targets:
            try:
                self.load_model(device=target)
                self.init_decoder()
                if is_fallback:
                    self._cpu_fallback = True
                action = '回退 CPU' if is_fallback else f'重建 {target.upper()} session'
                msg = f"⚠ GPU 设备已被驱动重置（设备丢失），已{action}，重试当前音频"
                print(msg)
                if self.progress_callback:
                    self.progress_callback(msg)
                return True
            except Exception as exc:
                msg = f"⚠ 重建 {target.upper()} session 失败：{exc}"
                print(msg)
                if self.progress_callback:
                    self.progress_callback(msg)
        return False

    def init_decoder(self):
        self.nll_decoder = NonLexicalDecoder(vocab=self.vocab,
                                             class_names=['None', *self.vocab['non_lexical_phonemes']],
                                             sample_rate=self.mel_cfg["sample_rate"], hop_size=self.mel_cfg["hop_size"])
        self.fa_decoder = AlignmentDecoder(vocab=self.vocab, sample_rate=self.mel_cfg["sample_rate"],
                                           hop_size=self.mel_cfg["hop_size"])

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def get_dataset(self, wav_folder, language, g2p="dictionary", dictionary_path=None, in_format="lab",
                    insert_sp=True):
        if dictionary_path is None:
            dictionary_path = self.vocab_folder / self.vocab["dictionaries"].get(language, "")

        if g2p == "dictionary":
            assert os.path.exists(dictionary_path), f"{Path(dictionary_path).absolute()} does not exist."
            g2p = DictionaryG2P(language, dictionary_path, self.vocab.get("silent_phonemes", []),
                                insert_sp=insert_sp)
        elif g2p == "phoneme":
            g2p = PhonemeG2P(language)
        else:
            raise f"g2p - {g2p} is not supported, which should be 'dictionary' or 'phoneme'."

        wav_paths = Path(wav_folder).rglob("*.wav")
        for wav_path in wav_paths:
            try:
                lab_path = wav_path.with_suffix("." + in_format)
                if lab_path.exists():
                    with open(lab_path, "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    ph_seq, word_seq, ph_idx_to_word_idx = g2p(lab_text)
                    self.dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
                else:
                    warnings.warn(f"{Path(wav_path).absolute()} does not exist.")
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
        msg = f"Loaded {len(self.dataset)} samples."
        print(msg)
        if self.progress_callback:
            self.progress_callback(msg)

    def get_dataset_from_list(self, wav_paths, language, g2p="dictionary", dictionary_path=None, in_format="lab",
                              insert_sp=True):
        """从指定的 wav 文件列表加载数据集（用于并行推理时每个子进程加载自己的子集）"""
        if dictionary_path is None:
            dictionary_path = self.vocab_folder / self.vocab["dictionaries"].get(language, "")

        if g2p == "dictionary":
            assert os.path.exists(dictionary_path), f"{Path(dictionary_path).absolute()} does not exist."
            g2p_obj = DictionaryG2P(language, dictionary_path, self.vocab.get("silent_phonemes", []),
                                    insert_sp=insert_sp)
        elif g2p == "phoneme":
            g2p_obj = PhonemeG2P(language)
        else:
            raise ValueError(f"g2p - {g2p} is not supported, which should be 'dictionary' or 'phoneme'.")

        for wav_path in wav_paths:
            wav_path = Path(wav_path)
            try:
                lab_path = wav_path.with_suffix("." + in_format)
                if lab_path.exists():
                    with open(lab_path, "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    ph_seq, word_seq, ph_idx_to_word_idx = g2p_obj(lab_text)
                    self.dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
                else:
                    warnings.warn(f"{lab_path} does not exist.")
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
        msg = f"Loaded {len(self.dataset)} samples."
        print(msg)
        if self.progress_callback:
            self.progress_callback(msg)

    def _run_model(self, input_feed):
        """运行 ONNX 模型（原生错误信息无法解码时由 ort_utils 还原后抛出）。"""
        return _ort_run_session(self.model, self._output_names, input_feed)

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        # 使用缓存的输出名称，避免每次查询
        results = dict(zip(
            self._output_names,
            self._run_model({'waveform': [padded_wav]})
        ))

        words, _, warning_log = self.fa_decoder.decode(
            ph_frame_logits=results['ph_frame_logits'][:, :, padded_frames:],
            ph_edge_logits=results['ph_edge_logits'][:, padded_frames:],
            wav_length=wav_length, ph_seq=ph_seq, word_seq=word_seq, ph_idx_to_word_idx=ph_idx_to_word_idx
        )

        non_lexical_words = self.nll_decoder.decode(cvnt_logits=results['cvnt_logits'][:, :, padded_frames:],
                                                    wav_length=wav_length, non_lexical_phonemes=non_lexical_phonemes)
        return words, non_lexical_words, warning_log

    @staticmethod
    def run_onnx(session, input_dict):
        output_names = [output.name for output in session.get_outputs()]
        return dict(zip(output_names, _ort_run_session(session, output_names, input_dict)))

    @staticmethod
    def create_session(onnx_path, device='cpu'):
        # 根据用户选择设置providers
        if device.lower() == 'dml':
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("INFO: 用户选择 DirectML (GPU加速) 进行推理")
        elif device.lower() == 'cuda':
            providers = ['CUDAExecutionProvider','DmlExecutionProvider', 'CPUExecutionProvider']
            print("INFO: 用户选择 CUDA (GPU加速) 进行推理")
        elif device.lower() == 'webgpu':
            providers = ['WebGpuExecutionProvider', 'CPUExecutionProvider']
            print("INFO: 用户选择 WebGPU 进行推理")
        else:
            providers = ['CPUExecutionProvider']
            print("INFO: 用户选择 CPU 进行推理")
        
        available_providers = ort.get_available_providers()
        print(f"INFO: 可用推理架构: {available_providers}")
        
        # 检测可用的provider
        enabled_providers = [p for p in providers if p in available_providers]
        
        if not enabled_providers:
            # 如果所有指定的provider都不可用，使用CPU
            enabled_providers = ['CPUExecutionProvider']
            print("WARNING: 指定的执行提供程序都不可用，使用CPUExecutionProvider")
        else:
            # 输出实际使用的设备信息
            primary_device = enabled_providers[0]
            if primary_device == 'DmlExecutionProvider':
                print("INFO: 实际使用 DirectML (GPU加速) 进行推理")
            elif primary_device == 'CUDAExecutionProvider':
                print("INFO: 实际使用 CUDA (GPU加速) 进行推理")
            elif primary_device == 'WebGpuExecutionProvider':
                print("INFO: 实际使用 WebGPU 进行推理")
            else:
                print("INFO: 实际使用 CPU 进行推理")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 禁用内存模式复用，避免 DML 下的内存碎片问题
        options.enable_mem_pattern = False
        # 禁用 CPU 内存 arena，减少内存占用
        options.enable_cpu_mem_arena = False

        if 'DmlExecutionProvider' in enabled_providers:
            # DML 不支持真正的并行执行，使用 SEQUENTIAL 模式避免调度开销
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # DML 下 inter_op 线程数设低，减少不必要的线程切换开销
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 0  # 让 ORT 自动选择
            print("INFO: 已启用DirectML优化 (SEQUENTIAL模式, 单线程调度)")

        elif 'CUDAExecutionProvider' in enabled_providers:
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            options.inter_op_num_threads = 2
            options.intra_op_num_threads = 0
            print("INFO: 已启用CUDA优化")

        elif 'WebGpuExecutionProvider' in enabled_providers:
            # WebGPU 使用 GPU 计算，采用 SEQUENTIAL 模式避免线程竞争
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 0
            print("INFO: 已启用WebGPU优化 (SEQUENTIAL模式)")

        else:
            # CPU 模式
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            options.inter_op_num_threads = min(cpu_count, 8)
            options.intra_op_num_threads = cpu_count
            print(f"INFO: 已启用CPU优化 (inter_op={options.inter_op_num_threads}, intra_op={options.intra_op_num_threads})")

        return _ort_create_session(str(onnx_path), sess_options=options, providers=enabled_providers)

    def _get_device_info(self):
        """返回当前 session 实际使用的设备信息"""
        if self.model is None:
            return {'available_providers': ort.get_available_providers(), 
                    'enabled_providers': [], 'primary_device': 'N/A'}
        enabled = self.model.get_providers()
        return {
            'available_providers': ort.get_available_providers(),
            'enabled_providers': enabled,
            'primary_device': enabled[0] if enabled else 'CPUExecutionProvider'
        }

    def infer(self, non_lexical_phonemes, pad_times=1, pad_length=5, merge_phonemes=True, stop_event=None):
        nll_phonemes, add_sp, unsupported = resolve_non_lexical(
            non_lexical_phonemes, self.vocab.get('non_lexical_phonemes', []))
        if unsupported:
            msg = f"非语言音素 {', '.join(unsupported)} 不在模型词表中"
            if add_sp and 'SP' in unsupported:
                msg += "（SP 将用于补全空白区域）"
            print(msg)
            if self.progress_callback:
                self.progress_callback(msg)
        if not nll_phonemes and not add_sp:
            msg = "未启用自动添加非语言音素（输出不含 AP/SP 标签）"
        else:
            msg = (f"自动添加非语言音素: NLL 检测={', '.join(nll_phonemes) or '无'}"
                   f"，SP 补全空白={'是' if add_sp else '否'}")
        print(msg)
        if self.progress_callback:
            self.progress_callback(msg)

        # ── 输出实际推理使用的设备信息 ──
        primary_device = self.device_info['primary_device']
        device_name_map = {
            'DmlExecutionProvider': 'DirectML (GPU加速)',
            'CUDAExecutionProvider': 'CUDA (NVIDIA GPU)',
            'WebGpuExecutionProvider': 'WebGPU (GPU加速)',
            'CPUExecutionProvider': 'CPU',
        }
        device_display = device_name_map.get(primary_device, primary_device)
        provider_list = ', '.join(self.device_info['enabled_providers'])
        device_msg = f"实际推理设备: {device_display} | Providers: [{provider_list}]"
        print("=" * 60)
        print(device_msg)
        print("=" * 60)
        if self.progress_callback:
            self.progress_callback(device_msg)

        _infer_start_time = time.time()

        pad_lengths = [round(pad_length / pad_times * i, 1) for i in range(0, pad_times)] if pad_times > 1 else [0]
        total = len(self.dataset)

        # 预计算 pad 参数（所有 pad_length 共享同一组模板）
        pad_params = []
        for pl in pad_lengths:
            padded_samples = int(pl * self.mel_cfg['sample_rate'])
            padded_frames = int(padded_samples / self.mel_cfg['hop_size'])
            pad_params.append((pl, padded_samples, padded_frames))

        # ── 逐文件推理（按需加载音频） ──
        for i, (wav_path, ph_seq, word_seq, ph_idx_to_word_idx) in enumerate(self.dataset):
            # 检查用户是否请求强制停止
            if stop_event is not None and stop_event.is_set():
                raise StopInference()

            if (i + 1) % 10 == 0 or i == total - 1:
                msg = f"Processing {i + 1}/{total}..."
                print(msg)
                if self.progress_callback:
                    self.progress_callback(msg)

            # 单个文件处理（封装成函数，便于设备丢失后重试）
            def _run_one(wav_path=wav_path, ph_seq=ph_seq, word_seq=word_seq,
                         ph_idx_to_word_idx=ph_idx_to_word_idx):
                # 按需加载当前音频
                wav, sr = librosa.load(wav_path, sr=self.mel_cfg['sample_rate'], mono=True)
                wav_length = len(wav) / self.mel_cfg['sample_rate']

                words_list: list[WordList] = []
                all_warning_logs = []

                for pl, padded_samples, padded_frames in pad_params:
                    padded_wav = np.pad(wav, (padded_samples, 0), mode='constant', constant_values=0)

                    words, non_lexical_words, warning_log = self._infer(
                        padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx,
                        wav_length, nll_phonemes)
                    if warning_log:
                        all_warning_logs.extend(warning_log)
                    for _words in non_lexical_words:
                        for word in _words:
                            words.add_AP(word)
                    words.clear_language_prefix()
                    words_list.append(words)

                if all_warning_logs:
                    wav_name = os.path.basename(wav_path)
                    warning_msg = f"{wav_name}:\n" + "\n".join(all_warning_logs)
                    print(warning_msg)
                    if self.progress_callback:
                        self.progress_callback(warning_msg)

                # 多轮推理结果一致性检查与合并
                if len(words_list) == 1:
                    # pad_times=1 时无需去重，直接使用
                    result_word = words_list[0]
                else:
                    ph_list = [words.phonemes for words in words_list]
                    duplicate_indices = find_all_duplicate_phonemes(ph_list)

                    if len(duplicate_indices) == 0:
                        wav_name = os.path.basename(wav_path)
                        warning_msg = f"{wav_name}: 多次推理结果不一致，使用第一次推理的结果"
                        print(warning_msg)
                        if self.progress_callback:
                            self.progress_callback(warning_msg)
                        words_list = [words_list[0]]
                    else:
                        words_list = [words_list[i] for i in duplicate_indices]

                    phonemes_all = []
                    result_word = WordList()
                    for w_idx in range(len(words_list[0])):
                        phonemes = []
                        for ph_idx in range(len(words_list[0][w_idx].phonemes)):
                            ph_start = remove_outliers_per_position(
                                [[words[w_idx].phonemes[ph_idx].start for words in words_list]])[0]
                            ph_end = remove_outliers_per_position(
                                [[words[w_idx].phonemes[ph_idx].end for words in words_list]])[0]
                            ph_start = max(ph_start, phonemes_all[-1].end if len(phonemes_all) > 0 else 0)
                            ph_end = max(ph_start + 0.0001, ph_end)
                            phonemes.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                            phonemes_all.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                        word = Word(phonemes[0].start, phonemes[-1].end, words_list[0][w_idx].text)
                        for ph in phonemes:
                            word.append_phoneme(ph)
                        result_word.append(word)

                # 先闭合词与词之间的小缝隙（同原版 post_processing），
                # 避免后续 add_SP 生成 10ms 级别的碎片 SP
                result_word.fill_small_gaps(wav_length)

                if add_sp:
                    result_word.add_SP(wav_length, "SP")

                if merge_phonemes:
                    result_word.merge_duplicate_phonemes(min_duration=0.05)

                # 获取后处理日志并推送到前端
                warning_log = result_word.log()
                if warning_log:
                    warnings.warn(warning_log)
                    if self.progress_callback:
                        self.progress_callback(warning_log)

                self.predictions.append((wav_path, wav_length, result_word))

            # 单个文件失败不影响其余文件；设备丢失时重建 session 后重试
            attempt = 0
            while True:
                try:
                    _run_one()
                    break
                except StopInference:
                    raise
                except Exception as e:
                    if attempt < 2 and _ort_is_device_lost(e) and self._recover_from_device_lost():
                        attempt += 1
                        continue
                    tb = traceback.format_exc()
                    self.failed_files.append((str(wav_path), tb))
                    fail_msg = f"❌ 推理失败: {Path(wav_path).absolute()}\n{tb}"
                    print(fail_msg)
                    if self.progress_callback:
                        self.progress_callback(fail_msg)
                    break

        # ── 输出推理总耗时 ──
        elapsed = time.time() - _infer_start_time
        time_msg = f"推理完成，总耗时: {elapsed:.1f}s  (共 {total} 个文件，平均 {elapsed / max(total, 1):.2f}s/个)"
        print(time_msg)
        if self.progress_callback:
            self.progress_callback(time_msg)

        # ── 汇总失败文件 ──
        if self.failed_files:
            failed_names = "\n".join(
                f"  {i + 1}. {Path(p).absolute()}" for i, (p, _) in enumerate(self.failed_files))
            summary = (f"⚠ 共 {len(self.failed_files)}/{total} 个音频推理失败:\n{failed_names}")
            print(summary)
            if self.progress_callback:
                self.progress_callback(summary)

    def export(self, output_folder, output_format=None):
        if output_format is None:
            output_format = ['textgrid']
        self._export_textgrid(output_folder)
        msg = "Output files are saved to the same folder as the input wav files."
        print(msg)
        if self.progress_callback:
            self.progress_callback(msg)

    def _export_textgrid(self, output_folder=None):
        for wav_path, wav_length, words in self.predictions:
            # TextGrid 一律写在各自 wav 同目录下，
            # 这样无论输入是文件夹还是零散选择的文件都能正确定位。
            tg_path = Path(wav_path).with_suffix('.TextGrid')
            
            phonemes = []
            for word in words:
                phonemes.extend(word.phonemes)
            
            with open(tg_path, 'w', encoding='utf-8') as f:
                f.write('File type = "ooTextFile"\n')
                f.write('Object class = "TextGrid"\n\n')
                f.write(f'xmin = 0\n')
                f.write(f'xmax = {wav_length}\n')
                f.write('tiers? <exists>\n')
                f.write('size = 2\n')
                f.write('item []:\n')
                f.write('    item [1]:\n')
                f.write('        class = "IntervalTier"\n')
                f.write(f'        name = "words"\n')
                f.write(f'        xmin = 0\n')
                f.write(f'        xmax = {wav_length}\n')
                f.write(f'        intervals: size = {len(words)}\n')
                for i, word in enumerate(words):
                    f.write(f'        intervals [{i+1}]:\n')
                    f.write(f'            xmin = {word.start}\n')
                    f.write(f'            xmax = {word.end}\n')
                    f.write(f'            text = "{word.text}"\n')
                f.write('    item [2]:\n')
                f.write('        class = "IntervalTier"\n')
                f.write(f'        name = "phones"\n')
                f.write(f'        xmin = 0\n')
                f.write(f'        xmax = {wav_length}\n')
                f.write(f'        intervals: size = {len(phonemes)}\n')
                for i, phoneme in enumerate(phonemes):
                    f.write(f'        intervals [{i+1}]:\n')
                    f.write(f'            xmin = {max(0, phoneme.start)}\n')
                    f.write(f'            xmax = {phoneme.end}\n')
                    f.write(f'            text = "{phoneme.text}"\n')


def find_all_duplicate_phonemes(ph_list):
    if len(ph_list) == 0:
        return []
    if len(ph_list) == 1:
        return [0]
    index_dict = defaultdict(list)
    for idx, sublist in enumerate(ph_list):
        key = tuple(sublist)
        index_dict[key].append(idx)

    duplicate_phonemes = {key: indices for key, indices in index_dict.items() if len(indices) > 1}

    if not duplicate_phonemes:
        return []

    sorted_groups = sorted(duplicate_phonemes.items(), key=lambda x: (-len(x[1]), -len(x[0])))
    best_key, best_indices = sorted_groups[0]
    return best_indices


class DictionaryG2P:
    def __init__(self, language, dictionary_path, silent_phonemes=None, insert_sp=True):
        self.language = language
        self.dictionary_path = dictionary_path
        self.silent_phonemes = set(silent_phonemes) if silent_phonemes else set()
        self.insert_sp = insert_sp
        self.dictionary = self._load_dictionary()

    def _load_dictionary(self):
        dictionary = {}
        with open(self.dictionary_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    phonemes = parts[1:]
                    dictionary[word] = phonemes
        return dictionary

    def __call__(self, text):
        # 与原版 HubertFA 保持一致：首尾以及每个词之间都插入 SP 音素。
        # 这些 SP 会让强制对齐把静音/换气段分配给 SP，而不是硬塞进相邻的音节，
        # 这样 NLL 检测到的 AP 才不会被单词覆盖而被裁掉。
        words = text.split()
        ph_seq = ["SP"] if self.insert_sp else []
        word_seq = []
        ph_idx_to_word_idx = [-1] if self.insert_sp else []

        for word in words:
            if word in self.dictionary:
                # 语言前缀已直接写在词典文件里(如 ja/j), 不再自动补充
                phonemes = [ph for ph in self.dictionary[word] if ph]
                # 词首/词尾的静音音素不参与对齐（同原版行为）
                while phonemes and phonemes[0] in self.silent_phonemes:
                    phonemes.pop(0)
                while phonemes and phonemes[-1] in self.silent_phonemes:
                    phonemes.pop()
            else:
                # 未收录的词：直接把原词当音素使用，避免整句丢失
                phonemes = [word]

            if not phonemes:
                continue

            word_idx = len(word_seq)
            word_seq.append(word)
            ph_seq.extend(phonemes)
            ph_idx_to_word_idx.extend([word_idx] * len(phonemes))

            if self.insert_sp and ph_seq and ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_idx_to_word_idx.append(-1)

        return ph_seq, word_seq, ph_idx_to_word_idx


class PhonemeG2P:
    def __init__(self, language):
        self.language = language

    def __call__(self, text):
        phonemes = list(text)
        ph_idx_to_word_idx = list(range(len(phonemes)))
        return phonemes, phonemes, ph_idx_to_word_idx


def parallel_worker(worker_id, onnx_path_str, wav_folder_str, language, dict_path_str,
                    device, pad_times, pad_length, merge_phonemes, non_lexical_phonemes,
                    wav_files_list, msg_queue=None):
    """
    并行推理的工作进程函数（支持 DML / WebGPU 等多设备）。
    每个子进程加载自己的模型实例，处理一半的 wav 文件，输出 TextGrid 到同一目录。
    由于各进程独立，不会冲突。

    Args:
        msg_queue: multiprocessing.Queue，用于将进度消息传回主进程（GUI 显示）
    """
    import onnxruntime as _ort
    _ort.set_default_logger_severity(3)  # 抑制 ORT 日志

    onnx_path = Path(onnx_path_str)
    wav_folder = Path(wav_folder_str)

    inference = InferenceOnnx(onnx_path)
    inference.load_config()
    inference.load_model(device=device)
    inference.init_decoder()

    def _send(msg):
        """向主进程发送消息（同时打印到子进程自己的控制台）"""
        print(msg)
        if msg_queue is not None:
            try:
                msg_queue.put_nowait(msg)
            except Exception:
                pass

    inference.set_progress_callback(lambda m: _send(f"[Worker {worker_id}] {m}"))

    try:
        _send(f"[Worker {worker_id}] Loading {len(wav_files_list)} files...")
        inference.get_dataset_from_list(
            wav_files_list, language=language,
            g2p="dictionary", dictionary_path=dict_path_str, in_format="lab",
            insert_sp=bool(str(non_lexical_phonemes or "").strip())
        )
        inference.infer(
            non_lexical_phonemes=non_lexical_phonemes,
            pad_times=pad_times, pad_length=pad_length,
            merge_phonemes=merge_phonemes
        )
        inference.export(str(wav_folder))
    except Exception:
        tb = traceback.format_exc()
        _send(f"❌ [Worker {worker_id}] 推理失败:\n{tb}")
        return

    if inference.failed_files:
        failed_names = "\n".join(
            f"  {i + 1}. {Path(p).absolute()}" for i, (p, _) in enumerate(inference.failed_files))
        _send(f"⚠ [Worker {worker_id}] 共 {len(inference.failed_files)} 个音频推理失败:\n{failed_names}")

    _send(f"[Worker {worker_id}] ✅ 完成处理 {len(wav_files_list)} 个文件，成功 {len(inference.predictions)} 个")


# 兼容旧名称
dml_worker = parallel_worker


if __name__ == '__main__':
    import argparse
# python onnx_infer.py --onnx_path ./HubertFA_model/1218_hfa_model/model.onnx --language ja --dictionary ./HubertFA_model/1218_hfa_model/ja-all.txt --wav_folder F:\Download\SVDBCreator_Release_1.0.0\baini_JP\wav\C4 --out_path F:\Download\SVDBCreator_Release_1.0.0\baini_JP\wav\C4
    parser = argparse.ArgumentParser(description='ONNX Inference for Forced Alignment')
    parser.add_argument('--onnx_path', '-m', type=Path, required=True, help='Path to ONNX model')
    parser.add_argument('--wav_folder', '-wf', type=Path, default='segments', help='Input folder path')
    parser.add_argument('--out_path', '-o', type=str, default=None, help='Path to output label')
    parser.add_argument('--g2p', '-g', type=str, default='dictionary', help='G2P class name')
    parser.add_argument('--non_lexical_phonemes', '-np', type=str, default='AP', help='Non-speech phonemes, e.g. AP,EP')

    parser.add_argument('--language', '-l', type=str, default='zh', help='Dictionary language')
    parser.add_argument('--dictionary', '-d', type=Path, default=None, help='Custom dictionary path')

    parser.add_argument('--pad_times', '-pt', type=int, default=1, help='Number of times to pad blank audio')
    parser.add_argument('--pad_length', '-pl', type=int, default=5, help='Max length of blank audio padding')

    args = parser.parse_args()

    assert args.onnx_path.exists() and args.onnx_path.is_file() and args.onnx_path.suffix == '.onnx', \
        f"Path {args.onnx_path} does not exist or is not an ONNX file."

    inference = InferenceOnnx(onnx_path=args.onnx_path)
    inference.load_config()
    inference.init_decoder()
    inference.load_model()
    
    print("="*50)
    print("ONNX Runtime Device Information:")
    print(f"Available Providers: {inference.device_info['available_providers']}")
    print(f"Enabled Providers: {inference.device_info['enabled_providers']}")
    print(f"Primary Device: {inference.device_info['primary_device']}")
    print("="*50)
    
    inference.get_dataset(wav_folder=args.wav_folder, language=args.language, g2p=args.g2p, dictionary_path=args.dictionary)
    inference.infer(non_lexical_phonemes=args.non_lexical_phonemes, pad_times=args.pad_times, pad_length=args.pad_length)
    inference.export(output_folder=args.wav_folder if args.out_path is None else args.out_path)