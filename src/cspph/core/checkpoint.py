"""Atomic candidate-batch checkpointing for resumable extraction."""
from __future__ import annotations
import gzip, json, os
from pathlib import Path
from typing import Iterator, Mapping


class CandidateCheckpointWriter:
    def __init__(self, root: str | Path, target_id: str, batch_size: int = 16):
        self.root=Path(root); self.target_id=str(target_id); self.batch_size=max(1,int(batch_size)); self.rows=[]; self.batch_index=0
        (self.root / "target_shards").mkdir(parents=True,exist_ok=True)
        committed = sorted((self.root / "target_shards").glob(f"{self.target_id}.batch_*.jsonl.gz"))
        if committed:
            self.batch_index = max(int(path.name.split(".batch_")[1].split(".")[0]) for path in committed) + 1

    def append(self, row: Mapping[str, object]) -> None:
        self.rows.append(dict(row))
        if len(self.rows)>=self.batch_size: self.flush()

    def flush(self) -> Path | None:
        if not self.rows: return None
        final=self.root/"target_shards"/f"{self.target_id}.batch_{self.batch_index:06d}.jsonl.gz"; temp=final.with_suffix(final.suffix+f".tmp.{os.getpid()}")
        with gzip.open(temp,"wt",encoding="utf-8") as h:
            for row in self.rows: h.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
        os.replace(temp,final); self.rows=[]; self.batch_index+=1; return final

    def close(self) -> None: self.flush()


def iter_checkpoint_rows(root: str | Path, target_id: str) -> Iterator[dict]:
    """Yield rows from atomically committed batches for one target."""
    directory = Path(root) / "target_shards"
    for path in sorted(directory.glob(f"{target_id}.batch_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def completed_candidate_ids(root: str | Path, target_id: str, key: str = "candidate_id") -> set[str]:
    """Return committed candidate IDs so interrupted jobs can resume safely."""
    return {str(row[key]) for row in iter_checkpoint_rows(root, target_id) if key in row}


__all__=["CandidateCheckpointWriter", "iter_checkpoint_rows", "completed_candidate_ids"]
