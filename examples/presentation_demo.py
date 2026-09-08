import json, pathlib, subprocess, tempfile
from xpref.trace import read_trace
from xpref.prefetch import ExpertLayout, Prefetcher
trace = read_trace("samples/k3-q4-128tok.bin")
result = subprocess.run([".venv/bin/xpref","eval","--trace","samples/k3-q4-128tok.bin","--json"],text=True,capture_output=True,check=True)
print("Synthetic trace evaluation; t/s fields are formula projections:")
print(result.stdout.strip())
with tempfile.TemporaryDirectory(prefix="xpref-demo-") as folder:
    checkpoint=pathlib.Path(folder)/"synthetic.bin"
    checkpoint.write_bytes(bytes(16384))
    layout=ExpertLayout.uniform(1,4,16384)
    with Prefetcher(checkpoint,layout) as prefetcher:
        result=prefetcher.prefetch(0,1)
        print(json.dumps({"synthetic_checkpoint_bytes":16384,"hint_api_available":prefetcher.available,"hinted_offset":result[0],"hinted_bytes":result[1]}))
