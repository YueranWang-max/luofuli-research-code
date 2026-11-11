"""Frozen 2 model x 2 prompt x 2 budget x 16 problem exploratory pilot."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from environments.math_env import MathVerifier

OUT = ROOT/'artifacts/protocol_pilot_20261017'
LOCK = ROOT/'artifacts/.gpu_lock'
MODEL_NAMES = ('Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p = ROOT/'artifacts/benchmarks/mathhard/val.jsonl'
    pool = [json.loads(l) for l in p.read_text(encoding='utf-8').split('\n') if l.strip()]
    rng = random.Random(20261017)
    selected=[]
    for level in ('L4','L5'):
        subset=sorted((r for r in pool if r['difficulty']==level),key=lambda r:r['task_id'])
        assert len(subset)>=8
        selected.extend(rng.sample(subset,8))
    jobs=[]
    for model in MODEL_NAMES:
        block=[]
        for row in selected:
            q=row['prompt'].split('\n\nSolve this problem')[0]
            prompts={
                'compact':row['prompt'],
                'unconstrained':q+"\n\nSolve this problem, then end your reply with exactly 'Answer: <final number>'. The final answer must be a plain number (no units, no commas).",
            }
            for style,prompt in prompts.items():
                for cap in (256,512):
                    block.append(dict(key=f"{model}/{row['task_id']}/{style}/{cap}",model=model,
                        task_id=row['task_id'],difficulty=row['difficulty'],prompt_style=style,
                        max_new_tokens=cap,prompt=prompt,reference_answer=row['reference_answer']))
        rng.shuffle(block)
        jobs.extend(block)
    manifest=dict(seed=20261017,source_sha256=sha(p),protocol_sha256=sha(ROOT/'paper/EXPERIMENT_PROTOCOL_20261017.md'),
        script_sha256=sha(Path(__file__)),jobs=jobs,
        note='Exploratory pilot; model blocks fixed, randomized within model; timing is not a model-speed benchmark.')
    path=OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text())==manifest, 'Frozen manifest differs; use a new version, never silently resume.'
    else:
        path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    manifest=prepare()
    if args.prepare_only:
        print(f"Frozen {len(manifest['jobs'])} requests: {OUT/'manifest.json'}")
        return
    # Exclusive creation, unlike check-then-replace locking.
    try:
        fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'GPU lock exists: {LOCK}; inspect owner before retrying')
    os.write(fd,f'protocol_pilot pid={os.getpid()}'.encode())
    os.close(fd)
    try:
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if any(int(x.strip())>2000 for x in used.splitlines()):
            raise RuntimeError('GPU occupied; no inference started')
        import torch
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        torch.set_num_threads(8)
        set_seed(20261017)
        assert torch.cuda.is_available()
        results_path=OUT/'rows.jsonl'
        done=set()
        if results_path.exists():
            done={json.loads(l)['key'] for l in results_path.read_text(encoding='utf-8').split('\n') if l.strip()}
        started=time.time()
        for model_name in MODEL_NAMES:
            jobs=[j for j in manifest['jobs'] if j['model']==model_name and j['key'] not in done]
            if not jobs:
                continue
            model_path=ROOT/'models'/model_name
            fingerprints={p.name:sha(p) for p in sorted(model_path.iterdir()) if p.is_file() and (p.suffix=='.json' or p.suffix=='.safetensors')}
            identity=OUT/(model_name+'_identity.json')
            if identity.exists():
                assert json.loads(identity.read_text())==fingerprints
            else:
                identity.write_text(json.dumps(fingerprints,indent=2))
            print(f'[load] {model_name}',flush=True)
            tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
            if tok.pad_token_id is None:
                tok.pad_token=tok.eos_token
            model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
                device_map={'':'cuda:0'},local_files_only=True,use_safetensors=True)
            model.eval()
            for j in jobs:
                input_ids=tok.apply_chat_template([{'role':'user','content':j['prompt']}],
                    add_generation_prompt=True,return_tensors='pt').to('cuda:0')
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0=time.perf_counter()
                with torch.inference_mode():
                    generated=model.generate(input_ids,attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=j['max_new_tokens'],do_sample=False,pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-t0
                ids=generated[0,input_ids.shape[1]:].tolist()
                text=tok.decode(ids,skip_special_tokens=True)
                eos=model.generation_config.eos_token_id
                eos_ids=eos if isinstance(eos,list) else [eos]
                stop='eos' if ids and ids[-1] in eos_ids else ('length' if len(ids)>=j['max_new_tokens'] else 'other')
                grade=MathVerifier().verify(text,j['reference_answer'])
                record={**j,'completion':text,'input_ids':input_ids[0].tolist(),'generated_ids':ids,
                    'input_tokens':input_ids.shape[1],'generated_tokens':len(ids),'stop_reason':stop,
                    'passed_legacy_verifier':grade.passed,'extracted':grade.extracted,
                    'generation_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'torch_version':torch.__version__,'transformers_version':transformers.__version__,
                    'gpu':torch.cuda.get_device_name(0),'recorded_unix':time.time()}
                with results_path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    f.flush()
                    os.fsync(f.fileno())
                done.add(j['key'])
                print(f"[{len(done)}/128] {j['key']} pass={grade.passed} tokens={len(ids)} stop={stop} sec={elapsed:.1f}",flush=True)
            del model,tok
            gc.collect()
            torch.cuda.empty_cache()
        (OUT/'completed.json').write_text(json.dumps({'n':len(done),'elapsed_seconds':time.time()-started,'finished_unix':time.time()},indent=2))
    except Exception as exc:
        (OUT/f'failure_{time.time_ns()}.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'time':time.time()},indent=2))
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__=='__main__':
    main()
