"""把版本号写入 i18n/__init__.py 的 APP_VERSION。

用法：
    python scripts/stamp_version.py 1.2.3
    python scripts/stamp_version.py v1.2.3   # 开头的 v 会被自动去掉

主要用于 CI 打包时把 tag 版本号注入到程序里（GUI 会显示该版本号）。

注意：控制台输出统一为 ASCII，并把 stdout/stderr 强制为 UTF-8，
否则在 Windows CI（默认 cp1252）打印 emoji / 中文会抛 UnicodeEncodeError。
"""
import re
import sys
from pathlib import Path

# Windows CI 的 stdout 默认可能是 cp1252，强制 UTF-8 防止编码异常
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "i18n" / "__init__.py"

PATTERN = re.compile(r'APP_VERSION\s*=\s*["\'][^"\']*["\']')


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python scripts/stamp_version.py <version>", file=sys.stderr)
        return 1

    version = sys.argv[1].strip().lstrip("vV")
    if not version:
        print("Error: empty version", file=sys.stderr)
        return 1

    if not TARGET.exists():
        print(f"Error: file not found: {TARGET}", file=sys.stderr)
        return 1

    text = TARGET.read_text(encoding="utf-8")
    new_text, count = PATTERN.subn(f'APP_VERSION = "v{version}"', text, count=1)

    if count == 0:
        print("Error: APP_VERSION not found in i18n/__init__.py", file=sys.stderr)
        return 1

    TARGET.write_text(new_text, encoding="utf-8")
    print(f"[OK] APP_VERSION -> v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
