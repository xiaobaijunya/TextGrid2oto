"""纯数学计算函数模块

这些函数只包含纯数学计算，可以安全地用Cython编译加速。
"""

import numpy as np


def sigmoid(x):
    """计算sigmoid函数
    
    Args:
        x: 输入数组
    
    Returns:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """计算softmax函数
    
    Args:
        x: 输入数组
        axis: 归一化轴
    
    Returns:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """计算log_softmax函数
    
    Args:
        x: 输入数组
        axis: 归一化轴
    
    Returns:
        log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def median_abs_deviation(x, axis=0, center=np.median, scale=1.0):
    """计算中位数绝对偏差 (Median Absolute Deviation)
    
    Args:
        x: 输入数组
        axis: 计算轴
        center: 中心统计量函数
        scale: 缩放因子
    
    Returns:
        MAD值
    """
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
    """移除每个位置的异常值
    
    Args:
        data_series: 数据序列列表，每个元素是一个位置的多个值
        threshold: Z分数阈值
    
    Returns:
        处理后的值列表
    """
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


def find_all_duplicate_phonemes(ph_list):
    """查找所有重复的音素序列
    
    Args:
        ph_list: 音素序列列表
    
    Returns:
        重复次数最多的音素序列的索引列表
    """
    from collections import defaultdict
    
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


def forward_pass(T, S, prob_log, edge_prob, curr_ph_max_prob_log, dp, ph_seq_id, prob3_pad_len=2):
    """动态规划前向传播算法（核心算法）
    
    Args:
        T: 时间帧数
        S: 音素序列长度
        prob_log: 概率对数 [S, T]
        edge_prob: 边界概率 [T]
        curr_ph_max_prob_log: 当前音素最大概率对数 [S]
        dp: 动态规划表 [S, T]
        ph_seq_id: 音素序列ID [S]
        prob3_pad_len: prob3的填充长度
    
    Returns:
        dp: 更新后的动态规划表
        backtrack_s: 回溯表
        curr_ph_max_prob_log: 更新后的当前音素最大概率对数
    """
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