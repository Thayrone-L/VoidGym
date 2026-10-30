from __future__ import annotations

from collections.abc import Callable


class ConsoleUI:
    """Short Portuguese prompts with explicit safe defaults."""

    def __init__(self, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> None:
        self._input = input_fn
        self._output = output_fn

    def say(self, message: str) -> None:
        self._output(message)

    def ask(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        answer = self._input(prompt + suffix + ": ").strip()
        return default if not answer and default is not None else answer

    def confirm(self, prompt: str, *, default: bool) -> bool:
        choice = "S/n" if default else "s/N"
        answer = self._input(f"{prompt} ({choice}): ").strip().casefold()
        if not answer:
            return default
        return answer in {"s", "sim", "y", "yes"}

    def choose_int(self, prompt: str, *, minimum: int, maximum: int, default: int) -> int:
        while True:
            raw = self.ask(prompt, str(default))
            try:
                value = int(raw)
            except ValueError:
                self.say("Digite um número inteiro.")
                continue
            if minimum <= value <= maximum:
                return value
            self.say(f"Escolha um valor entre {minimum} e {maximum}.")
