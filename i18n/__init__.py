"""多语言支持模块"""
import json
import sys
from pathlib import Path


# ── 路径：始终使用 exe / 项目根目录下的 i18n 文件夹 ──────────
def _get_app_root() -> Path:
    """与 GUI.py 一致的万能获取程序根目录"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    # 开发模式：i18n/__init__.py → 项目根
    return Path(__file__).resolve().parent.parent

I18N_DIR = _get_app_root() / "i18n"
CONFIG_FILE = _get_app_root() / "config" / "lang_pref.json"

# 可用语言列表
LANGUAGES = {
    "zh": "中文",
    "en":    "English",
    "ja":    "日本語",
}

# 应用版本号
APP_VERSION = "v2.0.2"


# ── 语言偏好持久化 ────────────────────────────────────────────
def _load_pref() -> str:
    """读取上次保存的语言设置，文件不存在返回空"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                code = data.get("lang", "")
                if code in LANGUAGES:
                    return code
    except Exception:
        pass
    return ""


def _save_pref(lang_code: str):
    """保存语言偏好到配置文件"""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"lang": lang_code}, f, ensure_ascii=False)
    except Exception:
        pass


# ── 翻译引擎（单例）───────────────────────────────────────────
class Lang:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if self._loaded:
            return
        self._loaded = True
        # 优先读取保存的偏好，否则默认英文
        saved = _load_pref()
        default = saved if saved else "en"
        self._current = default
        self._strings = {}
        self._load_lang(default)

    @property
    def current(self) -> str:
        return self._current

    def switch(self, lang_code: str):
        lang_file = I18N_DIR / f"{lang_code}.json"
        if not lang_file.exists():
            raise FileNotFoundError(f"语言文件不存在: {lang_file}")
        self._load_lang(lang_code)
        _save_pref(lang_code)  # 切换时自动保存

    def tr(self, key: str, default: str = "") -> str:
        return self._strings.get(key, default or key)

    def __call__(self, key: str, default: str = "") -> str:
        return self.tr(key, default)

    def _load_lang(self, lang_code: str):
        lang_file = I18N_DIR / f"{lang_code}.json"
        if lang_file.exists():
            with open(lang_file, "r", encoding="utf-8") as f:
                self._strings = json.load(f)
            self._current = lang_code
        else:
            self._strings = {}
            self._current = lang_code


# ── 全局函数 ─────────────────────────────────────────────────
_LANG = Lang()

def _(key: str, default: str = "") -> str:
    return _LANG.tr(key, default)

def switch_lang(lang_code: str):
    _LANG.switch(lang_code)

def current_lang() -> str:
    return _LANG.current


# ── 控件注册 & 实时刷新 ──────────────────────────────────────
# 每个条目: (widget, key, kind, extra)
_WIDGETS: list = []

def register(widget, key: str, kind: str = "label", extra=None):
    """注册一个可翻译控件

    kind 说明:
      'label'         — wx.StaticText           → SetLabel(tr)
      'button'        — wx.Button               → SetLabel(tr)
      'checkbox'      — wx.CheckBox             → SetLabel(tr)
      'tooltip'       — 任意控件                → SetToolTip(tr)
      'title'         — wx.Frame                → SetTitle(tr)
      'notebook_tab'  — wx.Notebook             → SetPageText(extra, tr)
                            extra = page_index
      'choice'        — wx.Choice 整体替换       → Clear + 循环 Append
                            extra = [(item_key, clientData), ...]
      'choice_tooltip'— wx.Choice               → SetToolTip(tr)
    """
    _WIDGETS.append((widget, key, kind, extra))


def update_all():
    """刷新所有已注册控件的文本"""
    from i18n import _
    for widget, key, kind, extra in _WIDGETS:
        text = _(key)
        try:
            if kind in ("label", "button", "checkbox"):
                widget.SetLabel(text)
            elif kind == "tooltip":
                widget.SetToolTip(text)
            elif kind == "title":
                widget.SetTitle(text)
            elif kind == "notebook_tab":
                widget.SetPageText(extra, text)
            elif kind == "choice":
                widget.Clear()
                for item_key, client_data in extra:
                    widget.Append(_(item_key), client_data)
                if extra:
                    widget.SetSelection(0)
            elif kind == "choice_tooltip":
                widget.SetToolTip(text)
        except Exception:
            pass
