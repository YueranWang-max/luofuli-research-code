"""Fixed-batch greedy evaluation; resume never changes batch membership."""
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
from evaluation.answer_contracts import verify_formatted_final
from environments.math_env import MathVerifier

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=['Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct'])
    ap.add_argument('--adapter');ap.add_argument('--output',required=True)
    ap.add_argument('--development-smoke',action='store_true')
    ap.add_argument('--prompt-styles',nargs='+',choices=['compact','unconstrained'],
                    default=['compact','unconstrained'])
    a=ap.parse_args();batch_size=8
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    modelpath=ROOT/'models'/a.model
    adapter=Path(a.adapter).resolve() if a.adapter else None
    files=[p for p in modelpath.iterdir() if p.suffix in ('.json','.safetensors','.txt','.model','.jinja')]
    if adapter:
        assert (adapter/'completed.json').exists(),'Training incomplete'
        endpoint=json.loads((adapter/'completed.json').read_text(encoding='utf-8'))
        if not a.development_smoke:
            assert endpoint['steps']==100 and endpoint['endpoint_hashes'],'Invalid formal endpoint'
            assert all(sha(adapter/name)==value for name,value in endpoint['endpoint_hashes'].items()),'Adapter changed after training'
        files+=list(adapter.glob('adapter*.json'))+list(adapter.glob('*.safetensors'))
    files += [Path(__file__),ROOT/'src/rewards/v3.py',ROOT/'src/evaluation/answer_contracts.py',
        ROOT/'src/environments/math_env.py',ROOT/'paper/CONFIRMATORY_PROTOCOL_V3.md']
    jobs=[]
    datasets=('development',) if a.development_smoke else ('math','gsm')
    for dataset in datasets:
        p=ROOT/'artifacts/benchmarks/mathhard/val.jsonl' if dataset=='development' else ROOT/f'artifacts/v3/data/confirm_{dataset}.jsonl'
        files.append(p)
        rows=[json.loads(l) for l in p.read_text(encoding='utf-8').split('\n') if l.strip()]
        if dataset=='development':rows=rows[:8]
        else:assert len(rows)==300
        for row in rows:
            question=row.get('question',row['prompt'].split('\n\nSolve this problem')[0])
            for style in a.prompt_styles:
                prompt=row['prompt'] if style=='compact' else question+"\n\nSolve this problem, then end your reply with exactly 'Answer: <final number>'. The final answer must be a plain number (no units, no commas)."
                jobs.append(dict(key=f"{dataset}/{row['task_id']}/{style}",dataset=dataset,
                    task_id=row['task_id'],prompt=prompt,prompt_style=style,reference_answer=row['reference_answer']))
    manifest=dict(model=a.model,adapter=str(adapter) if adapter else None,jobs=jobs,
        batch_size=batch_size,development_smoke=a.development_smoke,prompt_styles=a.prompt_styles,
        decoding=dict(do_sample=False,max_new_tokens=1024,repetition_penalty=1.0),
        hashes={str(p):sha(p) for p in files})
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text(encoding='utf-8'))==manifest,'Frozen evaluation changed'
    else:mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    rp=out/'rows.jsonl';done={}
    if rp.exists():
        records=[json.loads(l) for l in rp.read_text(encoding='utf-8').split('\n') if l.strip()]
        done={r['key']:r for r in records};assert len(done)==len(records)
    assert set(done)<={j['key'] for j in jobs}
    if len(done)==len(jobs) and (out/'completed.json').exists():return
    lock=ROOT/'artifacts/.gpu_lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'v3 batched eval pid={os.getpid()} {out}'.encode());os.close(fd)
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        torch.set_num_threads(8);set_seed(20261017)
        tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        tok.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'})
        if adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        model.eval()
        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
        for batch_start in range(0,len(jobs),batch_size):
            batch=jobs[batch_start:batch_start+batch_size]
            if all(j['key'] in done for j in batch):continue
            chats=[[dict(role='user',content=j['prompt'])] for j in batch]
            inputs=tok.apply_chat_template(chats,add_generation_prompt=True,padding=True,return_tensors='pt',return_dict=True).to('cuda:0')
            width=inputs['input_ids'].shape[1]
            def generate(cap):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
                with torch.inference_mode():
                    raw=model.generate(**inputs,max_new_tokens=cap,do_sample=False,repetition_penalty=1.0,pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                sequences=[]
                for seq in raw[:,width:].tolist():
                    end=next((i+1 for i,t in enumerate(seq) if t in eos),len(seq))
                    sequences.append(seq[:end])
                return sequences,seconds,torch.cuda.max_memory_allocated()
            sequences,seconds,peak=generate(1024)
            short=None;check_seconds=None
            if (adapter is None and batch_start<16) or a.development_smoke:
                short,check_seconds,_=generate(256)
                if any(s!=long[:256] for s,long in zip(short,sequences)):
                    (out/'prefix_failure.json').write_text(json.dumps(dict(batch=batch,long=sequences,short=short)))
                    raise RuntimeError('Batched prefix mismatch: separate-budget evaluation required')
            with (out/'batch_times.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(batch_start=batch_start,unix=time.time(),n=len(batch),seconds=seconds,
                    short_check_seconds=check_seconds,peak_allocated_bytes=peak))+'\n');f.flush();os.fsync(f.fileno())
            for i,(j,ids) in enumerate(zip(batch,sequences)):
                if j['key'] in done:
                    assert done[j['key']]['generated_ids']==ids,'Resume regeneration changed output'
                    continue
                scores=[]
                for cap in (256,512,1024):
                    part=ids[:cap];text=tok.decode(part,skip_special_tokens=True)
                    strict,extracted=verify_final(text,j['reference_answer'])
                    scores.append(dict(cap=cap,text=text,generated_tokens=len(part),strict_passed=strict,
                        extracted=extracted,legacy_passed=MathVerifier().verify(text,j['reference_answer']).passed,
                        formatted_final_passed=verify_formatted_final(text,j['reference_answer'])[0],
                        stop_reason='eos' if part[-1] in eos else 'length',derived_prefix=cap<1024))
                input_ids=inputs['input_ids'][i][inputs['attention_mask'][i].bool()].tolist()
                record=dict(**j,input_ids=input_ids,generated_ids=ids,scores=scores,batch_start=batch_start,
                    batch_size=len(batch),generation_seconds_allocated=seconds/len(batch),
                    prefix_check_ids=short[i] if short is not None else None,recorded_unix=time.time(),
                    torch_version=torch.__version__,transformers_version=transformers.__version__)
                with rp.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                done[j['key']]=record
            print(f'[{len(done)}/{len(jobs)}] batch_seconds={seconds:.2f}',flush=True)
        (out/'completed.json').write_text(json.dumps(dict(n=len(done),finished_unix=time.time(),development_smoke=a.development_smoke)))
    except BaseException:
        (out/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:lock.unlink()

if __name__=='__main__':main()
