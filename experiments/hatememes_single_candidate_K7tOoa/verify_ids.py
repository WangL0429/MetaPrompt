"""Recover the Arrow row IDs from the source split manifests, verifying all rows."""
import csv
import json
from pathlib import Path

import pyarrow as pa

DATA = Path("/disk1/longw/Datasets/hatememes")


def main():
    with (DATA / "data.jsonl").open() as stream:
        metadata = {str(r["id"]): r for r in map(json.loads, stream)}
    seen = set()
    splits = {}
    for split, csv_name, arrow_name, expected in (("train","train","train",8500),("val","valid","dev",500),
                                                   ("test","test","test",1000)):
        with (DATA / (csv_name + ".csv")).open() as stream:
            ids = [r["id"] for r in csv.DictReader(stream)]
        assert len(ids) == len(set(ids)) == expected and seen.isdisjoint(ids)
        seen.update(ids)
        with pa.memory_map(str(DATA / f"hatememes_{arrow_name}.arrow"),"r") as source:
            t = pa.ipc.open_file(source).read_all()
            assert len(t) == len(ids)
            for i, key in enumerate(ids):
                r = metadata[key]
                assert t["label"][i].as_py() == int(r["label"]) and t["text"][i].as_py() == [r["text"]]
                assert (DATA / "image" / r["image"]).read_bytes() == t["image"][i].as_py(), (split, i, key)
        splits[split] = ids
    out = Path(__file__).resolve().parent / "verified_ids.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(splits))
    tmp.replace(out)
    print({k:len(v) for k,v in splits.items()})


if __name__ == "__main__":
    main()
