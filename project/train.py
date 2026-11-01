"""Small, reproducible Qwen3 LoRA pipeline demonstration (not a quality benchmark)."""
import json
import os
from pathlib import Path
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

OUT = Path(os.environ.get('OUTPUT_DIR', '/job/out'))
OUT.mkdir(parents=True, exist_ok=True)
set_seed(42)
assert torch.cuda.is_available(), 'A CUDA GPU is required'
model_id = 'Qwen/Qwen3-8B'
dataset_id = 'HuggingFaceH4/ultrachat_200k'
api = HfApi()
model_revision = api.model_info(model_id).sha
data_revision = api.dataset_info(dataset_id).sha
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
tokenizer.pad_token = tokenizer.eos_token

def prepare(split, count):
    source = load_dataset(dataset_id, revision=data_revision, split=split, streaming=True)
    source = source.shuffle(seed=42, buffer_size=2000)
    rows = []
    for item in source:
        messages = item['messages']
        if len(messages) < 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            continue
        prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
        answer = tokenizer(messages[1]['content'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        if len(prefix) > 192:
            continue
        ids = (prefix + answer)[:256]
        if len(ids) - len(prefix) < 16:
            continue
        rows.append({'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': [-100]*len(prefix) + ids[len(prefix):]})
        if len(rows) == count:
            break
    assert len(rows) == count
    return Dataset.from_list(rows)

train_data = prepare('train_sft', 1000)
eval_data = prepare('test_sft', 32)
train_data.save_to_disk(str(OUT / 'train_tokens'))
eval_data.save_to_disk(str(OUT / 'eval_tokens'))
model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
model.print_trainable_parameters()

def collate(rows):
    size = max(len(row['input_ids']) for row in rows)
    return {key: torch.tensor([row[key] + [pad]*(size-len(row[key])) for row in rows]) for key,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}

args = TrainingArguments(output_dir=str(OUT / 'checkpoints'), num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=8, per_device_eval_batch_size=1, learning_rate=1e-4, warmup_ratio=0.05, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False}, logging_steps=5, save_steps=25, save_total_limit=2, report_to='none', optim='adamw_torch', seed=42, dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=eval_data, data_collator=collate)
before = trainer.evaluate()
result = trainer.train()
after = trainer.evaluate()
trainer.save_model(str(OUT / 'adapter'))
tokenizer.save_pretrained(OUT / 'adapter')
trainer.save_state()
summary = {'model':model_id, 'model_revision':model_revision, 'dataset':dataset_id, 'dataset_revision':data_revision, 'train_examples':1000, 'eval_examples':32, 'max_tokens':256, 'train_metrics':result.metrics, 'before':before, 'after':after, 'purpose':'Pipeline demo; short first-turn responses, not a general quality benchmark.'}
(OUT / 'metrics.json').write_text(json.dumps(summary, indent=2))
model.eval()
model.config.use_cache = True
prompt = 'Explain why the sky is blue in two sentences.'
inputs = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt').cuda()
with torch.no_grad():
    generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
(OUT / 'sample.json').write_text(json.dumps({'prompt':prompt,'response':tokenizer.decode(generated[0,inputs.shape[1]:], skip_special_tokens=True)}, indent=2))
assert (OUT / 'adapter' / 'adapter_model.safetensors').stat().st_size > 0
print('DEMO COMPLETE', json.dumps(summary), flush=True)
