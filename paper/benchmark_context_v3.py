"""Synthetic context memory/throughput probe, never an accuracy experiment."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    lock=ROOT/'artifacts/.gpu_lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'context benchmark pid={os.getpid()}'.encode());os.close(fd)
    result=dict(synthetic_repeated_development_prompt=True,context_tokens=768,
                forced_generated_tokens=32,accuracy_interpretation='none',measurements=[])
    destination=ROOT/f'artifacts/v3/engineering/context_probe_{time.time_ns()}.json'
    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
        from peft import PeftModel
        torch.set_num_threads(8);set_seed(42)
        modelpath=ROOT/'models/Qwen3-4B-Instruct'
        tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,
            dtype=torch.bfloat16,device_map={'':'cuda:0'})
        model=PeftModel.from_pretrained(model,ROOT/'artifacts/v3/smoke/qwen4_attempt1',is_trainable=False)
        model.eval()
        row=json.loads((ROOT/'artifacts/benchmarks/mathhard/val.jsonl').read_text(encoding='utf-8').split('\n')[0])
        original=tok.apply_chat_template([dict(role='user',content=row['prompt'])],add_generation_prompt=True)
        repeated=(original*((768+len(original)-1)//len(original)))[:768]
        result.update(model='Qwen3-4B-Instruct',adapter='qwen4_attempt1',original_task_id=row['task_id'],
                      input_ids=repeated,torch_version=torch.__version__)
        # Fixed sequence, repeated in reverse to expose warmup/order effects.
        for iteration,batch in enumerate((1,2,4,8,8,4,2,1)):
            inputs=torch.tensor([repeated]*batch,device='cuda:0')
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(inputs,attention_mask=torch.ones_like(inputs),
                    max_new_tokens=32,min_new_tokens=32,do_sample=False,repetition_penalty=1.0,
                    pad_token_id=tok.pad_token_id,disable_compile=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            result['measurements'].append(dict(iteration=iteration,batch=batch,seconds=elapsed,
                aggregate_tokens_per_second=batch*32/elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),generated_ids=output[:,768:].tolist()))
            destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(iteration,batch,round(elapsed,3),flush=True)
            del inputs,output
        result['complete']=True
        destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
