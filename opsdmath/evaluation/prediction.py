"""Generate prediction JSONL for the single-budget training diagnostics."""

import time

from opsdmath.io import write_rows
from opsdmath.training.prompts import prompt
from opsdmath.training.rollout import batch_suffixes
from opsdmath.training.runtime import log, memory


def predict(args, model, tokenizer, rows, device):
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    predictions = []
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index:index + args.batch_size]
        start = time.perf_counter()
        suffixes = batch_suffixes(
            model, tokenizer, [prompt(tokenizer, row, thinking=args.student_thinking) for row in batch],
            args.cap, False, device,
        )
        token_counts = []
        for row, (suffix, hit_cap) in zip(batch, suffixes, strict=True):
            token_counts.append(len(suffix))
            prediction = {
                "id": row["id"], "response": tokenizer.decode(suffix, skip_special_tokens=True),
                "generated_tokens": len(suffix), "hit_cap": hit_cap,
            }
            if args.save_token_ids:
                prediction["token_ids"] = suffix
            predictions.append(prediction)
        log({"event": "prediction_batch", "count": len(batch), "tokens": sum(token_counts),
             "seconds": time.perf_counter() - start, **memory(device)}, args.output)
    write_rows(args.output / f"{args.prediction_name}.jsonl", predictions)
