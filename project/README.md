# YAAD general LoRA demo

Qwen/Qwen3-8B (Apache 2.0) trained on 1,000 filtered first-turn examples from HuggingFaceH4/ultrachat_200k train_sft (MIT). Evaluation uses 32 examples from the separate test_sft split before and after training.

One epoch, BF16, rank-8 LoRA on attention query/value projections, 256-token limit, effective batch size 8. Prompts and padding are excluded from loss. Long prompts are skipped; answers may be truncated. This is a pipeline demonstration, not evidence of general model improvement.

Run `bash run.sh` in the YAAD NVIDIA PyTorch container. Results go to `/job/out`: adapter and tokenizer, two most recent resumable checkpoints, metrics, sample response, exact tokenized subsets, model/dataset commit IDs, and package versions. The adapter requires the original base model; base weights are not copied into results.

Model: https://huggingface.co/Qwen/Qwen3-8B
Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
