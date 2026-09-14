"""把版本号写入 i18n/__init__.py 的 APP_VERSION。

用法：
    python scripts/stamp_version.py 1.2.3
    python scripts/stamp_version.py v1.2.3   # 开头的 v 会被自动去掉

主要用于 CI 打包时把 tag 版本号注入到程序里（GUI 左下角会显示）。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "i18n" / "__init__.py"

PATTERN = re.compile(r'APP_VERSION\s*=\s*["\'][^"\']*["\']')


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("用法: python scripts/stamp_version.py <version>", file=sys.stderr)
        return 1

    version = sys.argv[1].strip().lstrip("vV")
    if not version:
        print("错误: 版本号为空", file=sys.stderr)
        return 1

    if not TARGET.exists():
        print(f"错误: 找不到 {TARGET}", file=sys.stderr)
        return 1

    text = TARGET.read_text(encoding="utf-8")
    new_text, count = PATTERN.subn(f'APP_VERSION = "v{version}"', text, count=1)

    if count == 0:
        print("错误: 在 i18n/__init__.py 中未找到 APP_VERSION", file=sys.stderr)
        return 1

    TARGET.write_text(new_text, encoding="utf-8")
    print(f"✅ APP_VERSION -> v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
