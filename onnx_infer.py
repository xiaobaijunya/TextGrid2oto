import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import librosa
import numpy as np
import onnxruntime as ort


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

    def add_SP(self, wav_length, add_phone="SP"):
        try:
            words_res = WordList()
            words_res._log = self._log

            if self[0].start > 0:
                try:
                    words_res.append(Word(0, self[0].start, add_phone, init_phoneme=True))
                except ValueError as e:
                    self._add_log(f"ERROR: {e}")

            words_res.append(self[0])
            for i in range(1, len(self)):
                word = self[i]
                if word.start > words_res[-1].end:
                    try:
                        words_res.append(Word(words_res[-1].end, word.start, add_phone, init_phoneme=True))
                    except ValueError as e:
                        self._add_log(f"ERROR: {e}")
                words_res.append(word)

            if self[-1].end < wav_length:
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
                ph_seq_id_list.append(self.vocab["vocab"]["SP"])
                warning_msg = f"音素 '{ph}' 不在模型词汇表中，已替换为 'SP'"
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
        edge_prob_log, not_edge_prob_log = np.log(edge_prob + 1e-6), np.log(1 - edge_prob + 1e-6)
        mask_reset = (ph_seq_id == 0)

        prob1 = np.empty(S, dtype=np.float32)
        prob2 = np.full(S, -np.inf, dtype=np.float32)
        prob3 = np.full(S, -np.inf, dtype=np.float32)

        i_vals_prob3 = np.arange(prob3_pad_len, S)
        idx_arr = np.clip(i_vals_prob3 - prob3_pad_len + 1, 0, S - 1)
        mask_cond_prob3 = (idx_arr >= S - 1) | (ph_seq_id[idx_arr] == 0)

        for t in range(1, T):
            prob_log_t, edge_log_t, not_edge_log_t = prob_log[:, t], edge_prob_log[t], not_edge_prob_log[t]
            dp_prev = dp[:, t - 1]

            prob1[:] = dp_prev + prob_log_t + not_edge_log_t

            prob2[1:] = dp_prev[:S - 1] + prob_log_t[:S - 1] + edge_log_t + curr_ph_max_prob_log[:S - 1] * (T / S)

            candidate_vals = dp_prev[:S - prob3_pad_len] + prob_log_t[
                :S - prob3_pad_len] + edge_log_t + curr_ph_max_prob_log[:S - prob3_pad_len] * (T / S)
            prob3[i_vals_prob3] = np.where(mask_cond_prob3, candidate_vals, -np.inf)

            stacked_probs = np.vstack((prob1, prob2, prob3))
            max_indices = np.argmax(stacked_probs, axis=0)
            dp[:, t], backtrack_s[:, t] = stacked_probs[max_indices, np.arange(S)], max_indices

            mask_type0 = (max_indices == 0)
            np.maximum(curr_ph_max_prob_log, prob_log_t, out=curr_ph_max_prob_log, where=mask_type0)
            np.copyto(curr_ph_max_prob_log, prob_log_t, where=~mask_type0)
            curr_ph_max_prob_log[mask_reset] = 0.0

            prob2[1:], prob3[i_vals_prob3] = -np.inf, -np.inf
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

        ph_idx_seq, ph_time_int, frame_confidence = [], [], []

        if S == 1:
            s = 0
        else:
            s = S - 2 if dp[-2, -1] > dp[-1, -1] and ph_seq_id[-1] == 0 else S - 1

        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[s, t] >= 0 or t == 0
            frame_confidence.append(dp[s, t])

            if backtrack_s[s, t] != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)

                if backtrack_s[s, t] == 1:
                    s -= 1
                elif backtrack_s[s, t] == 2:
                    s -= 2

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

    def load_model(self, device='cpu'):
        self.model = self.create_session(self.model_folder / 'model.onnx', device=device)
        self.device_info = self._get_device_info()

    def init_decoder(self):
        self.nll_decoder = NonLexicalDecoder(vocab=self.vocab,
                                             class_names=['None', *self.vocab['non_lexical_phonemes']],
                                             sample_rate=self.mel_cfg["sample_rate"], hop_size=self.mel_cfg["hop_size"])
        self.fa_decoder = AlignmentDecoder(vocab=self.vocab, sample_rate=self.mel_cfg["sample_rate"],
                                           hop_size=self.mel_cfg["hop_size"])

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def get_dataset(self, wav_folder, language, g2p="dictionary", dictionary_path=None, in_format="lab"):
        if dictionary_path is None:
            dictionary_path = self.vocab_folder / self.vocab["dictionaries"].get(language, "")
        language = language if self.vocab['language_prefix'] else None

        if g2p == "dictionary":
            assert os.path.exists(dictionary_path), f"{Path(dictionary_path).absolute()} does not exist."
            g2p = DictionaryG2P(language, dictionary_path)
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

    def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length, non_lexical_phonemes):
        results = self.run_onnx(self.model, {'waveform': [padded_wav]})

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
        return dict(zip(output_names, session.run(output_names, input_dict)))

    @staticmethod
    def create_session(onnx_path, device='cpu'):
        # 根据用户选择设置providers
        if device.lower() == 'dml':
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("INFO: 用户选择 DirectML (GPU加速) 进行推理")
        else:
            providers = ['CPUExecutionProvider']
            print("INFO: 用户选择 CPU 进行推理")
        
        available_providers = ort.get_available_providers()
        
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
            else:
                print("INFO: 实际使用 CPU 进行推理")
        
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), options, providers=enabled_providers)

    @staticmethod
    def _get_device_info():
        providers = [ 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        enabled_providers = []
        
        for provider in providers:
            if provider in available_providers:
                enabled_providers.append(provider)
        
        device_info = {
            'available_providers': available_providers,
            'enabled_providers': enabled_providers,
            'primary_device': enabled_providers[0] if enabled_providers else 'CPUExecutionProvider'
        }
        return device_info

    def infer(self, non_lexical_phonemes, pad_times=1, pad_length=5):
        non_lexical_phonemes = [ph.strip() for ph in non_lexical_phonemes.split(",") if ph.strip()]
        assert set(non_lexical_phonemes).issubset(set(self.vocab['non_lexical_phonemes'])), \
            f"The non_lexical_phonemes contain elements that are not included in the vocab."

        pad_lengths = [round(pad_length / pad_times * i, 1) for i in range(0, pad_times)] if pad_times > 1 else [0]

        for i in range(len(self.dataset)):
            if (i + 1) % 10 == 0 or i == len(self.dataset) - 1:
                msg = f"Processing {i + 1}/{len(self.dataset)}..."
                print(msg)
                if self.progress_callback:
                    self.progress_callback(msg)

            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = self.dataset[i]

            wav, sr = librosa.load(wav_path, sr=self.mel_cfg['sample_rate'], mono=True)
            wav_length = len(wav) / self.mel_cfg['sample_rate']

            words_list: list[WordList] = []
            all_warning_logs = []
            for pl in pad_lengths:
                padded_samples = int(pl * self.mel_cfg['sample_rate'])
                padded_frames = int(padded_samples / self.mel_cfg['hop_size'])
                padded_wav = np.pad(wav, (padded_samples, 0), mode='constant', constant_values=0)

                words, non_lexical_words, warning_log = self._infer(padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx,
                                                       wav_length, non_lexical_phonemes)
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

            ph_list = [words.phonemes for words in words_list]
            words_list = [words_list[i] for i in find_all_duplicate_phonemes(ph_list)]

            phonemes_all = []
            result_word = WordList()
            for w_idx in range(len(words_list[0])):
                phonemes = []
                for ph_idx in range(len(words_list[0][w_idx].phonemes)):
                    ph_start = \
                        remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].start for words in words_list]])[0]
                    ph_end = \
                        remove_outliers_per_position([[words[w_idx].phonemes[ph_idx].end for words in words_list]])[0]
                    ph_start = max(ph_start, phonemes_all[-1].end if len(phonemes_all) > 0 else 0)
                    ph_end = max(ph_start + 0.0001, ph_end)
                    phonemes.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                    phonemes_all.append(Phoneme(ph_start, ph_end, words_list[0][w_idx].phonemes[ph_idx].text))
                word = Word(phonemes[0].start, phonemes[-1].end, words_list[0][w_idx].text)
                for ph in phonemes:
                    word.append_phoneme(ph)
                result_word.append(word)
            result_word.fill_small_gaps(wav_length)
            result_word.merge_duplicate_phonemes(min_duration=0.05)  # 2. 再处理音素重复
            result_word.add_SP(wav_length)
            
            # 获取后处理日志并推送到前端
            warning_log = result_word.log()
            if warning_log:
                warnings.warn(warning_log)
                if self.progress_callback:
                    self.progress_callback(warning_log)
            
            self.predictions.append((wav_path, wav_length, result_word))

    def export(self, output_folder, output_format=None):
        if output_format is None:
            output_format = ['textgrid']
        self._export_textgrid(output_folder)
        msg = "Output files are saved to the same folder as the input wav files."
        print(msg)
        if self.progress_callback:
            self.progress_callback(msg)

    def _export_textgrid(self, output_folder):
        for wav_path, wav_length, words in self.predictions:
            relative_path = wav_path.relative_to(output_folder)
            tg_path = Path(output_folder) / relative_path.with_suffix('.TextGrid')
            
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
    if len(ph_list) == 1:
        return [0]
    index_dict = defaultdict(list)
    for idx, sublist in enumerate(ph_list):
        key = tuple(sublist)
        index_dict[key].append(idx)

    duplicate_phonemes = {key: indices for key, indices in index_dict.items() if len(indices) > 1}

    if not duplicate_phonemes:
        raise Exception("No duplicate groups")

    sorted_groups = sorted(duplicate_phonemes.items(), key=lambda x: (-len(x[1]), -len(x[0])))
    best_key, best_indices = sorted_groups[0]
    return best_indices


class DictionaryG2P:
    def __init__(self, language, dictionary_path):
        self.language = language
        self.dictionary_path = dictionary_path
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
        words = text.split()
        ph_seq = []
        word_seq = []
        ph_idx_to_word_idx = []

        for word_idx, word in enumerate(words):
            if word in self.dictionary:
                phonemes = self.dictionary[word]
                if self.language:
                    phonemes = [f"{self.language}/{ph}" for ph in phonemes]
                ph_seq.extend(phonemes)
                word_seq.append(word)
                for _ in phonemes:
                    ph_idx_to_word_idx.append(word_idx)
            else:
                ph_seq.append(word)
                word_seq.append(word)
                ph_idx_to_word_idx.append(word_idx)

        return ph_seq, word_seq, ph_idx_to_word_idx


class PhonemeG2P:
    def __init__(self, language):
        self.language = language

    def __call__(self, text):
        phonemes = list(text)
        ph_idx_to_word_idx = list(range(len(phonemes)))
        return phonemes, phonemes, ph_idx_to_word_idx


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