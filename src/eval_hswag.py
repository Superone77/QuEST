import json
import os
from typing import List, Tuple, Union, Optional

import fire
from huggingface_hub import snapshot_download
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("/projects/0/prjs1462/wanqi/repo/QuEST/src/hadamard_transform")
from hadamard_transformer_helper import naive_hadamard_transform_with_scale as hadamard_transform
from transformers import AutoTokenizer
from lm_eval.api.model import LM
from lm_eval import evaluator
from tqdm import tqdm

from optim.utils import load_checkpoint
from models.utils import get_model
from models.quantization.base_linear import OPTIMAL_GAUSSIAN_SCALES, HadamardTrustQuantizer, QuantizedLinear


class NanoLlamaLM(LM):
    def __init__(self, model, device="cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.model.tokenizer = self.tokenizer

    @property
    def tokenizer_name(self) -> str:
        return "meta-llama/Llama-2-7b-hf"

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Helper function to encode a context-continuation pair."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.model.tokenizer.encode(context + continuation)
        context_enc = self.model.tokenizer.encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        print("######### running loglikelihood")
        results = []

        for request in tqdm(requests):
            context, continuation = request.args

            context_enc, continuation_enc = self._encode_pair(context, continuation)

            if not continuation_enc:
                results.append((0.0, True))
                print("[WARNING] empty continuation")
                continue

            tokens = torch.tensor(context_enc + continuation_enc, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(tokens, get_logits=True, all_logits=True)
                logits = output["logits"]

                if logits is None or logits.size(1) == 0:
                    results.append((float("-inf"), False))
                    print("[WARNING] empty logits")
                    continue

                relevant_logits = logits[0, len(context_enc) - 1 : len(context_enc) + len(continuation_enc) - 1]
                if len(relevant_logits) == 0:
                    results.append((float("-inf"), False))
                    print("[WARNING] empty relevant logits")
                    continue

                log_probs = F.log_softmax(relevant_logits, dim=-1)

                continuation_tensor = torch.tensor(continuation_enc, device=self.device)
                token_log_probs = log_probs[range(len(continuation_tensor)), continuation_tensor]

                total_logprob = token_log_probs.sum().item()

                is_greedy = True
                for logits_i, token_i in zip(relevant_logits, continuation_tensor):
                    if torch.argmax(logits_i).item() != token_i.item():
                        is_greedy = False
                        break

                results.append((total_logprob, is_greedy))
        print("done loglikelihood")
        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        results = []

        for request in tqdm(requests):
            text = request.args[0]
            if not text:
                results.append(0.0)
                continue

            tokens = torch.tensor(self.model.tokenizer.encode(text), dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(tokens, get_logits=True)
                logits = output["logits"]

                if logits is None or logits.size(1) == 0:
                    results.append(float("-inf"))
                    continue

                log_probs = F.log_softmax(logits[0, :-1], dim=-1)
                token_log_probs = log_probs[range(len(tokens[0]) - 1), tokens[0, 1:]]
                total_logprob = token_log_probs.sum().item()
                results.append(total_logprob)

        return results

    def generate_until(self, requests) -> List[str]:
        results = []

        for request in tqdm(requests):
            context, gen_kwargs = request.args

            context_tokens = torch.tensor(
                self.model.tokenizer.encode(context), dtype=torch.long, device=self.device
            ).unsqueeze(0)

            max_tokens = gen_kwargs.get("max_gen_toks", 128)
            temperature = gen_kwargs.get("temperature", 1.0)
            top_k = gen_kwargs.get("top_k", None)
            until = gen_kwargs.get("until", None)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    context_tokens, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k
                )

            output_text = self.model.tokenizer.decode(output_tokens[0].tolist())

            if until is not None:
                for stop_seq in until:
                    stop_idx = output_text.find(stop_seq)

            results.append(output_text)

        return results

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        return ""


class Linear4bit(nn.Module):
    def __init__(self, quantizer_linear):
        super().__init__()

        assert isinstance(quantizer_linear.weight_quantizer, HadamardTrustQuantizer), f"quantizer_linear.weight_quantizer is not a HadamardTrustQuantizer, {type(quantizer_linear.weight_quantizer)}"
        assert isinstance(quantizer_linear.activation_quantizer, HadamardTrustQuantizer), f"quantizer_linear.activation_quantizer is not a HadamardTrustQuantizer, {type(quantizer_linear.activation_quantizer)}"

        self.activation_quantizer = quantizer_linear.activation_quantizer

        wq = dequantize_dense(*quantize_pack_hadamard_dense(quantizer_linear.weight, quantizer_linear.weight_quantizer))
        self.register_buffer("wq", wq)
        self.bias = quantizer_linear.bias

    def forward(self, x):
        x = dequantize_dense(*quantize_pack_hadamard_dense(x, self.activation_quantizer))
        return F.linear(x, self.wq, self.bias)


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as ex:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'") from ex

    def __setattr__(self, key, value):
        self[key] = value


class PseudoDdp(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._orig_mod = nn.ModuleDict(
            {
                "module": model,
            }
        )


class PseudoLoader:
    def load_state_dict(self, *args, **kwargs):
        pass


def quantize_pack_hadamard_dense(x: torch.Tensor, quantizer: HadamardTrustQuantizer):
    assert quantizer.centered
    x_had = hadamard_transform(x.reshape(-1, 128), scale=2 ** (-7 / 2)).reshape(x.shape)

    std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
    scale = OPTIMAL_GAUSSIAN_SCALES[quantizer.bits] * std

    step = 2 * scale / (quantizer.n_levels - 1)
    x_clip = torch.clamp(x_had, -scale, scale)
    xq = torch.round((x_clip + scale) / step)

    assert xq.min() >= 0 and xq.max() < quantizer.n_levels
    return xq, scale, step
    # ^ note: xq is in rotated space!


def dequantize_dense(xq, scale, step):
    return xq * step - scale


def replace_linears(model):
    for name, module in model.named_children():
        if isinstance(module, QuantizedLinear):
            model._modules[name] = Linear4bit(module)
        else:
            replace_linears(module)
    return model


def main(model_name: str, ckpts_dir: str = "./ckpts", limit: Optional[int] = None, log_samples: bool = False, no_wrap: bool = False):
    if os.path.exists(os.path.join(model_name, "summary.json")):
        ckpt_dir = model_name
    elif os.path.exists(os.path.join(ckpts_dir, model_name, "summary.json")):
        ckpt_dir = os.path.join(ckpts_dir, model_name)
    else:
        ckpt_dir = os.path.join(ckpts_dir, model_name.replace('/', '-'))
        snapshot_download(repo_id=model_name, local_dir=ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, "main.pt")

    with open(os.path.join(ckpt_dir, "summary.json"), "r") as f:
        config = json.load(f)
    # print(config)

    model = PseudoDdp(get_model(DotDict(config["args"]))) if not no_wrap else get_model(DotDict(config["args"]))
    load_checkpoint(model, PseudoLoader(), PseudoLoader(), ckpt_path, "cuda")
    model = model.cuda()
    if not no_wrap:
        model = model._orig_mod["module"]

    # todo: proper quantizer should be configured using config['args']
    if not no_wrap and not (config['args']['a_quant'] == 'NoQuantizer' and config['args']['w_quant'] == 'NoQuantizer'):
        model = replace_linears(model)

    eval_model = NanoLlamaLM(model)

    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=100000,
        log_samples=log_samples,
        verbosity="INFO",
    )

    write_out = results.pop("samples", None)

    print(results)
    with open(os.path.join(ckpt_dir, "hellaswag_results.json"), "w") as f:
        json.dump(results, f)

    if log_samples:
        with open(os.path.join(ckpt_dir, "hellaswag_samples.json"), "w") as f:
            json.dump(write_out, f)


if __name__ == "__main__":
    fire.Fire(main)
