from datasets import load_dataset, DatasetDict
from huggingface_hub import notebook_login

notebook_login()


common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "ir", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "ir", split="test"
)

print(common_voice)
