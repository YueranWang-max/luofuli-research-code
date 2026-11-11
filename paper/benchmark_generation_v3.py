"""Development-only speed diagnosis; no confirmation outcomes or training updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter',required=True)
    a=ap.parse_args();out=ROOT/'artifacts/v3/engineering';out.mkdir(exist_ok=True)
    path=out/f'generate_benchmark_{a.model}_{time.time_ns()}.json'
    lock=ROOT/'artifacts/.gpu_lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'benchmark pid={os.getpid()}'.encode());os.close(fd)
    results=[]
    try:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        from peft import PeftModel
        torch.set_num_threads(8)
        tok=AutoTokenizer.from_pretrained(ROOT/'models'/a.model,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        row=json.loads((ROOT/'artifacts/benchmarks/mathhard/val.jsonl').read_text(encoding='utf-8').split('\n')[0])
        inp=tok.apply_chat_template([dict(role='user',content=row['prompt'])],add_generation_prompt=True,return_tensors='pt').to('cuda:0')
        model=AutoModelForCausalLM.from_pretrained(ROOT/'models'/a.model,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'})
        model.eval()
        def measure(label,batch,sampling):
            x=inp.repeat(batch,1);set_seed(42)
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():
                y=model.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=32,min_new_tokens=32,
                    do_sample=sampling,temperature=1.15 if sampling else None,top_p=.95 if sampling else None,
                    top_k=0 if sampling else None,repetition_penalty=1.0,pad_token_id=tok.pad_token_id,disable_compile=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            results.append(dict(label=label,batch=batch,sampling=sampling,seconds=seconds,
                aggregate_tokens_per_second=32*batch/seconds,ids=y[:,x.shape[1]:].tolist()))
            path.write_text(json.dumps(dict(model=a.model,adapter=a.adapter,task_id=row['task_id'],results=results),indent=2))
            print(label,batch,sampling,round(seconds,3),flush=True)
        measure('base_warmup',1,False)
        measure('base',1,False);measure('base',8,False);measure('base',8,True)
        model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False);model.eval()
        measure('lora',1,False);measure('lora',8,False);measure('lora',8,True)
        # Engineering comparison only: merged/unloaded model is discarded after this script.
        model=model.merge_and_unload();model.eval()
        measure('merged',8,False);measure('merged',8,True)
    finally:lock.unlink()

if __name__=='__main__':main()
