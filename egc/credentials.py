"""Interactive local credential confirmation; never write secrets to stdout/logs."""
import getpass
import os
import sys


def read_key(ask_key=False, show_key=False):
    key = getpass.getpass("DeepSeek API key (hidden): ") if ask_key else os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("Use --ask-key or set DEEPSEEK_API_KEY in this terminal")
    if any(c.isspace() or ord(c) < 32 for c in key):
        raise ValueError("API key contains whitespace/control characters; paste it again without spaces")
    if show_key:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError("--show-key requires an interactive terminal, without output redirection")
        # Bypass stdout/stderr so application logs and Tee-Object never receive the key.
        console = "CONOUT$" if os.name == "nt" else "/dev/tty"
        with open(console, "w", encoding="utf-8") as terminal:
            terminal.write("API key (local console only): " + key + "\n")
            terminal.flush()
        if input("Check the displayed key. Enter y to use it, anything else to cancel: ").strip().lower() != "y":
            raise ValueError("Key confirmation cancelled; no API requests sent")
    return key
