"""Isolated, instrumented prospective training; one immutable attempt per directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from rewards.v3 import ARMS,reward_group,verify_final
from environments.math_env import MathVerifier

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=['Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct'])
    ap.add_argument('--arm',required=True,choices=ARMS)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--output',required=True)
    ap.add_argument('--resume-from')
    a=ap.parse_args()
    out=Path(a.output).resolve()
    lock=ROOT/'artifacts/.gpu_lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'v3 pid={os.getpid()} output={out}'.encode());os.close(fd)
    start=time.time()
    created=False
    try:
        if a.resume_from:
            assert out.is_dir(), 'Resume output directory must already exist'
            resume=Path(a.resume_from).resolve()
            assert resume.is_dir() and resume.parent==out, 'Checkpoint must belong to output directory'
            created=True
        else:
            out.mkdir(parents=True,exist_ok=False);created=True
            resume=None
        import torch
        import transformers
        import trl
        import peft
        import accelerate
        from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig,GRPOTrainer
        torch.set_num_threads(8)
        set_seed(a.seed)
        data=ROOT/'artifacts/v3/data/train.jsonl'
        rows=[json.loads(l) for l in data.read_text(encoding='utf-8').split('\n') if l.strip()]
        lookup={r['task_id']:r for r in rows}
        assert len(lookup)==len(rows)==1500
        sources=[Path(__file__),ROOT/'src/rewards/v3.py',ROOT/'src/environments/math_env.py',data,ROOT/'paper/CONFIRMATORY_PROTOCOL_V3.md']
        request=dict(args=vars(a),started=start,pid=os.getpid(),hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
            versions=dict(torch=torch.__version__,transformers=transformers.__version__,trl=trl.__version__,
                peft=peft.__version__,accelerate=accelerate.__version__,python=sys.version),
            device=dict(name=torch.cuda.get_device_name(0),total_memory=torch.cuda.get_device_properties(0).total_memory,
                cuda=torch.version.cuda,torch_cpu_threads=torch.get_num_threads()))
        (out/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
        modelpath=ROOT/'models'/a.model
        tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},use_safetensors=True)
        group_file=out/'groups.jsonl'
        group_serial=sum(1 for line in group_file.open(encoding='utf-8') if line.strip()) if group_file.exists() else 0
        def measured_reward(prompts,completions,completion_ids,task_id,trainer_state,**kwargs):
            nonlocal group_serial
            assert len(completions)%4==0
            result=[]
            with (out/'groups.jsonl').open('a',encoding='utf-8') as f:
                for j in range(0,len(completions),4):
                    ids=task_id[j:j+4]
                    assert len(set(ids))==1,'Noncontiguous generation group'
                    texts=[x[0]['content'] if isinstance(x,list) else x for x in completions[j:j+4]]
                    token_ids=[list(x) for x in completion_ids[j:j+4]]
                    ref=lookup[ids[0]]['reference_answer']
                    passed=[verify_final(t,ref)[0] for t in texts]
                    legacy=[MathVerifier().verify(t,ref).passed for t in texts]
                    reward=reward_group(passed,[len(x) for x in token_ids],a.arm)
                    rec=dict(group=group_serial,global_step=trainer_state.global_step,task_id=ids[0],
                        texts=texts,completion_ids=token_ids,passed=passed,legacy_passed=legacy,
                        truncated=[x[-1] not in (tok.eos_token_id,tok.pad_token_id) for x in token_ids],**reward)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    group_serial+=1;result.extend(reward['rewards'])
                f.flush();os.fsync(f.fileno())
            return result
        cfg=GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=1e-5,max_steps=a.steps,max_completion_length=768,num_generations=4,
            temperature=1.15,top_p=.95,epsilon=.2,epsilon_high=.28,gradient_checkpointing=True,
            beta=0.,save_steps=50,logging_steps=1,bf16=True,seed=a.seed,data_seed=a.seed,
            mask_truncated_completions=True,loss_type='dapo',scale_rewards='group',report_to=[])
        (out/'effective_config.json').write_text(cfg.to_json_string(),encoding='utf-8')
        ds=Dataset.from_list([dict(prompt=[dict(role='user',content=r['prompt'])],task_id=r['task_id']) for r in rows])
        class TimedGRPOTrainer(GRPOTrainer):
            def _generate(self,prompts):
                begin=time.perf_counter()
                with (out/'phase_times.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(event='generation_start',step=self.state.global_step,
                        unix=time.time(),n_prompts=len(prompts)))+'\n')
                result=super()._generate(prompts)
                with (out/'phase_times.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(event='generation_end',step=self.state.global_step,
                        unix=time.time(),seconds=time.perf_counter()-begin))+'\n')
                return result
        trainer=TimedGRPOTrainer(model=model,processing_class=tok,reward_funcs=measured_reward,args=cfg,train_dataset=ds,
            peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,task_type='CAUSAL_LM',
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        (out/'generation_config_training.json').write_text(trainer.generation_config.to_json_string(),encoding='utf-8')
        dtype_counts={}
        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                key=str(parameter.dtype);dtype_counts[key]=dtype_counts.get(key,0)+parameter.numel()
        (out/'runtime_model.json').write_text(json.dumps(dict(trainable_parameters_by_dtype=dtype_counts,
            attention_implementation=trainer.model.config._attn_implementation,
            tokenizer_eos=tok.eos_token_id,tokenizer_pad=tok.pad_token_id),indent=2),encoding='utf-8')
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer.save_model(str(out))
        trainer.state.save_to_json(str(out/'trainer_state.json'))
        endpoint_files=list(out.glob('adapter*.json'))+list(out.glob('*.safetensors'))
        assert any(p.suffix=='.safetensors' for p in endpoint_files),'Missing final adapter weights'
        (out/'completed.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-start,groups=group_serial,
            steps=trainer.state.global_step,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            endpoint_hashes={p.name:sha(p) for p in endpoint_files}),indent=2))
    except BaseException:
        if created:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        lock.unlink()

if __name__=='__main__':main()
