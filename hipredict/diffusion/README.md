# HiPredict Diffusion

Pretrained weights are available on Hugging Face at [syding9/HiPredict_Diffusion](https://huggingface.co/syding9/HiPredict_Diffusion).

| File | Description |
| --- | --- |
| [`FM_Condition/latest.pt`](https://huggingface.co/syding9/HiPredict_Diffusion/blob/main/FM_Condition/latest.pt) | Bidirectional flow-matching model |
| [`FM_Condition_gradient/latest.pt`](https://huggingface.co/syding9/HiPredict_Diffusion/blob/main/FM_Condition_gradient/latest.pt) | Bidirectional flow-matching model with gradient loss |
| [`brightness/model.pt`](https://huggingface.co/syding9/HiPredict_Diffusion/blob/main/brightness/model.pt) | Optional brightness correction head |

Download a checkpoint and load it with HiPredict:

```python
from huggingface_hub import hf_hub_download
from hipredict.diffusion import load_checkpoint

checkpoint = hf_hub_download(
    repo_id="syding9/HiPredict_Diffusion",
    filename="FM_Condition/latest.pt",
)
model, model_config, training_config, step = load_checkpoint(
    checkpoint, device="cuda"
)
```

Use the downloaded checkpoint path as `--ckpt_path` for inference or evaluation. See the [diffusion documentation](../../docs/diffusion.md) for usage.
