"""Resume frozen greedy evaluations with exact IDs and predeclared budget prefixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rewards.v3 import verify_final
from environments.math_env import MathVerifier
from evaluation.answer_contracts import verify_formatted_final

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=['Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct'])
    ap.add_argument('--adapter')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    modelpath=ROOT/'models'/a.model
    adapter=Path(a.adapter).resolve() if a.adapter else None
    files=list(modelpath.glob('*.json'))+list(modelpath.glob('*.safetensors'))
    if adapter:
        assert (adapter/'completed.json').exists(),'Training incomplete'
        files+=list(adapter.glob('*.json'))+list(adapter.glob('*.safetensors'))
    files += [Path(__file__),ROOT/'src/rewards/v3.py',ROOT/'src/evaluation/answer_contracts.py',ROOT/'paper/CONFIRMATORY_PROTOCOL_V3.md']
    jobs=[]
    for dataset in ('math','gsm'):
        p=ROOT/f'artifacts/v3/data/confirm_{dataset}.jsonl';files.append(p)
        rows=[json.loads(l) for l in p.read_text(encoding='utf-8').split('\n') if l.strip()]
        assert len(rows)==300
        for row in rows:
            for style in ('compact','unconstrained'):
                prompt=row['prompt'] if style=='compact' else row['question']+"\n\nSolve this problem, then end your reply with exactly 'Answer: <final number>'. The final answer must be a plain number (no units, no commas)."
                jobs.append(dict(key=f"{dataset}/{row['task_id']}/{style}",dataset=dataset,
                    task_id=row['task_id'],prompt=prompt,prompt_style=style,reference_answer=row['reference_answer']))
    frozen=dict(model=a.model,adapter=str(adapter) if adapter else None,jobs=jobs,
        hashes={str(p):sha(p) for p in files})
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text(encoding='utf-8'))==frozen,'Evaluation inputs changed'
    else:mp.write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    rp=out/'rows.jsonl'
    done={}
    if rp.exists():
        records=[json.loads(l) for l in rp.read_text(encoding='utf-8').split('\n') if l.strip()]
        done={r['key']:r for r in records};assert len(done)==len(records)
    assert set(done)<={j['key'] for j in jobs}
    if len(done)==len(jobs) and (out/'completed.json').exists():return
    lock=ROOT/'artifacts/.gpu_lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'v3 eval pid={os.getpid()} {out}'.encode());os.close(fd)
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        torch.set_num_threads(8);set_seed(20261017)
        tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'})
        if adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        model.eval()
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        # The first eight MATH questions, both prompt styles: explicit 256 verification for each base.
        check_tasks={j['task_id'] for j in jobs[:16]}
        for j in jobs:
            if j['key'] in done:continue
            inp=tok.apply_chat_template([dict(role='user',content=j['prompt'])],add_generation_prompt=True,return_tensors='pt').to('cuda:0')
            def generate(cap):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():
                    output=model.generate(inp,attention_mask=torch.ones_like(inp),max_new_tokens=cap,do_sample=False,
                        repetition_penalty=1.0,pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                return output[0,inp.shape[1]:].tolist(),time.perf_counter()-start,torch.cuda.max_memory_allocated()
            ids,seconds,peak=generate(1024)
            check=None
            if adapter is None and j['task_id'] in check_tasks:
                short,short_seconds,_=generate(256)
                check=dict(ids=short,seconds=short_seconds,match=short==ids[:256])
                if not check['match']:
                    (out/'prefix_failure.json').write_text(json.dumps(dict(job=j,long_ids=ids,check=check)))
                    raise RuntimeError('Prefix check failed: protocol requires actual separate budget runs')
            scores=[]
            for cap in (256,512,1024):
                part=ids[:cap];text=tok.decode(part,skip_special_tokens=True)
                strict,extracted=verify_final(text,j['reference_answer'])
                legacy=MathVerifier().verify(text,j['reference_answer'])
                tolerant=verify_formatted_final(text,j['reference_answer'])[0]
                scores.append(dict(cap=cap,text=text,generated_tokens=len(part),strict_passed=strict,
                    extracted=extracted,legacy_passed=legacy.passed,formatted_final_passed=tolerant,
                    stop_reason='eos' if part[-1] in eos else 'length',derived_prefix=cap<1024))
            rec=dict(**j,input_ids=inp[0].tolist(),generated_ids=ids,scores=scores,
                generation_seconds=seconds,peak_allocated_bytes=peak,prefix_check=check,
                recorded_unix=time.time(),torch_version=torch.__version__,transformers_version=transformers.__version__)
            with rp.open('a',encoding='utf-8') as f:
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            done[j['key']]=rec
            print(f'[{len(done)}/1200] {j["key"]}',flush=True)
        (out/'completed.json').write_text(json.dumps(dict(n=len(done),finished_unix=time.time())))
    except BaseException:
        (out/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:lock.unlink()

if __name__=='__main__':main()
