from trl import SFTTrainer
import inspect
sig = inspect.signature(SFTTrainer.__init__)
for p in sig.parameters.keys():
    print(p)
