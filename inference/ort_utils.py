# -*- coding: utf-8 -*-
"""ONNX Runtime 调用辅助：还原被编码问题掩盖的原生错误信息。

中文 Windows 下 ONNX Runtime 的 C++ 异常文本（含 DirectML / Windows API 的
本地化提示）是 GBK 编码，pybind11 转成 Python str 时按 UTF-8 解码，于是抛出::

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc9 in position 242

真正的报错原因（显存不足 / 设备丢失 / 参数非法 …）就藏在这个异常的
``.object`` 里，本模块负责把它按本地编码还原后再抛出，方便定位问题。
"""

from __future__ import annotations

import onnxruntime as ort


# DXGI_ERROR_DEVICE_REMOVED / TDR 等「设备被重置」类错误的特征
_DEVICE_LOST_MARKERS = (
    "887a0005",          # DXGI_ERROR_DEVICE_REMOVED
    "887a0006",          # DXGI_ERROR_DEVICE_HUNG
    "device_removed",
    "device removed",
    "device lost",
    "设备实例",         # 「GPU 设备实例已经暂停」
    "设备已被移除",
    "设备丢失",
    "已暂停",
)


def is_device_lost(err: BaseException) -> bool:
    """判断异常是否为 GPU 设备丢失/驱动重置（DirectML TDR 等）。"""
    msg = str(err).lower()
    return any(marker in msg for marker in _DEVICE_LOST_MARKERS)


def restore_native_message(err: UnicodeDecodeError) -> str:
    """从 UnicodeDecodeError.object 中还原原始（大多是 GBK）错误文本。"""
    raw = getattr(err, "object", b"") or b""
    if not raw:
        return str(err)
    for enc in ("gbk", "mbcs", "utf-8"):
        try:
            return raw.decode(enc, errors="replace").strip()
        except Exception:
            continue
    return repr(raw)


def run_session(session, output_names, input_feed):
    """等价于 ``session.run(output_names, input_feed)``。

    原生错误信息无法按 UTF-8 解码时，还原为 RuntimeError 再抛出。
    """
    try:
        return session.run(output_names, input_feed)
    except UnicodeDecodeError as e:
        raise RuntimeError(
            "ONNX Runtime 执行失败，原始错误信息不是 UTF-8（已按 GBK 还原）：\n"
            + restore_native_message(e)
        ) from None


def create_session(*args, **kwargs):
    """等价于 ``ort.InferenceSession(*args, **kwargs)``，同样还原无法解码的错误信息。"""
    try:
        return ort.InferenceSession(*args, **kwargs)
    except UnicodeDecodeError as e:
        raise RuntimeError(
            "ONNX Runtime 加载模型失败，原始错误信息不是 UTF-8（已按 GBK 还原）：\n"
            + restore_native_message(e)
        ) from None
