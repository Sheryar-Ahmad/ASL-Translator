from collections import deque
from typing import Any, Dict, Optional


class TextBuffer:
    def __init__(
        self,
        space_label: str = "space",
        delete_label: str = "delete",
        nothing_label: str = "nothing",
    ):
        self.space_label = space_label.lower().strip()
        self.delete_label = delete_label.lower().strip()
        self.nothing_label = nothing_label.lower().strip()
        self.parts = []

    @property
    def text(self) -> str:
        return "".join(self.parts).strip()

    def clear(self) -> None:
        self.parts = []

    def backspace(self) -> bool:
        if not self.parts:
            return False

        self.parts.pop()
        return True

    def commit_label(self, label: Optional[str], confidence: float = 1.0) -> bool:
        if not label:
            return False

        normalized = str(label).lower().strip()

        if normalized == self.nothing_label:
            return False

        if normalized == self.delete_label:
            return self.backspace()

        if normalized == self.space_label:
            if self.parts and self.parts[-1] != " ":
                self.parts.append(" ")
                return True
            return False

        if len(normalized) == 1:
            self.parts.append(normalized)
        else:
            self.parts.append(normalized)
            self.parts.append(" ")

        return True


class AutoCommitController:
    def __init__(self, asl_cfg: Dict[str, Any]):
        auto_cfg = asl_cfg.get("auto", {})
        labels_cfg = asl_cfg.get("labels", {})

        self.window = max(1, int(auto_cfg.get("vote_window", 9)))
        self.stable_ms = int(auto_cfg.get("stable_ms", 900))
        self.reset_ms = int(auto_cfg.get("reset_ms", 1400))
        self.threshold = float(asl_cfg.get("confidence_threshold", 0.65))
        self.nothing_label = str(labels_cfg.get("nothing_label", "nothing")).lower()

        self.history = deque(maxlen=self.window)
        self.candidate: Optional[str] = None
        self.candidate_since = 0
        self.last_commit = 0

    def reset(self) -> None:
        """Discard votes when recognition is paused or settings change."""
        self.history.clear()
        self.candidate = None
        self.candidate_since = 0
        self.last_commit = 0

    def update(
        self, label: Optional[str], confidence: float, now_ms: int
    ) -> Optional[str]:
        normalized = str(label).lower().strip() if label else self.nothing_label

        if confidence < self.threshold:
            normalized = self.nothing_label

        self.history.append(normalized)

        if not self.history:
            return None

        counts = {}

        for item in self.history:
            counts[item] = counts.get(item, 0) + 1

        majority_label, majority_count = max(
            counts.items(),
            key=lambda item: item[1],
        )

        required_majority = len(self.history) // 2 + 1

        if majority_label == self.nothing_label or majority_count < required_majority:
            self.candidate = None
            self.candidate_since = 0
            return None

        if majority_label != self.candidate:
            self.candidate = majority_label
            self.candidate_since = now_ms
            return None

        stable_for = now_ms - self.candidate_since
        since_last_commit = now_ms - self.last_commit

        if stable_for >= self.stable_ms and since_last_commit >= self.reset_ms:
            committed = self.candidate
            self.last_commit = now_ms
            self.candidate = None
            self.candidate_since = 0
            self.history.clear()
            return committed

        return None
