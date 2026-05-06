from trl import SFTConfig
import inspect
sig = inspect.signature(SFTConfig.__init__)
params = [p for p in sig.parameters.keys()]
for p in params:
    if 'seq' in p.lower() or 'max' in p.lower() or 'length' in p.lower() or 'dataset' in p.lower() or 'pack' in p.lower():
        print(p)
