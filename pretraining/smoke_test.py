"""CPU interface/gradient checks with reduced synthetic model sizes, no EEG data."""
import argparse,importlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def test(family,upstream):
    import torch
    import torch.nn.functional as F
    from run import build_job
    torch.set_num_threads(1);torch.manual_seed(0)
    job=build_job(family,Path('/synthetic-not-read'),upstream,Path('/synthetic-not-written'))
    sys.path[:0]=job['pythonpath']
    module=importlib.import_module(Path(job['command'][6]).stem)
    sys.argv=job['command'][6:]
    args=module.parse_args()
    assert args.epochs==40 and args.max_subjects==2000 and args.seed==42
    print(f'PASS {family}: complete launch preset accepted by retained parser',flush=True)
    if family=='cbramod':
        from basis_pretrain_wrapper_v10 import CBraModV11
        model=CBraModV11(n_layer=1,seq_len=3,n_channels=21,dropout=0)
        x=torch.randn(2,21,3,200);mask=(torch.rand(2,21,3)>.5).long()
        prediction=model(x,mask=mask)
        assert prediction.shape==x.shape
        loss=F.mse_loss(prediction[mask==1],x[mask==1])
    elif family=='csbrain':
        args.n_layer=1;args.seq_len=3;args.dropout=0
        model,sorted_indices=module.build_model(args)
        x=torch.randn(2,21,3,200)
        raw_mask=module.generate_mask(2,21,3,mask_ratio=args.mask_ratio,device=x.device)
        sorted_idx=torch.as_tensor(sorted_indices,dtype=torch.long)
        mask=raw_mask.index_select(1,sorted_idx)
        target=x.index_select(1,sorted_idx)
        prediction=model(x,mask=mask)
        assert prediction.shape==target.shape
        loss=F.mse_loss(prediction[mask==1],target[mask==1])
    else:
        args.embed_dim=64;args.decoder_embed_dim=64;args.encoder_depth=1;args.decoder_depth=1
        args.heads=4;args.head_dim=16;args.n_channels=3;args.window_size=600
        model=module.build_model(args,module.build_config(args))
        loss=model(torch.randn(2,3,600),torch.randn(2,3,3))
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print(f'PASS {family}: finite masked-reconstruction loss and gradients (reduced CPU model)',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream-root',type=Path,default=ROOT/'upstream')
    p.add_argument('--family',choices=['cbramod','csbrain','reve'])
    a=p.parse_args();upstream=a.upstream_root.resolve()
    if a.family:test(a.family,upstream)
    else:
        for family in ['cbramod','csbrain','reve']:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream-root',str(upstream),'--family',family],check=True)
        print('All three pretraining integrations passed. No historical checkpoint or full training was evaluated.')
if __name__=='__main__':main()
