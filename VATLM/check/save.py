from pathlib import Path
import torch
from vatlm_copy import (
    Config,
    TaskConfig,
    MyVATLM,
)


def main():
    model_size = 'large'
    if model_size == 'base':
        ckpt_path = '/home/minami/vatlm_data/pretrain_base_vox2.pt'
        cfg = Config(model_size='base')
        task_cfg = TaskConfig(model_size='base')
        vatlm = MyVATLM(
            cfg=cfg,
            task_cfg=task_cfg,
            dictionaries=None,
        )
    elif model_size == 'large':
        ckpt_path = '/home/minami/vatlm_data/pretrain_large_vox2.pt'
        cfg = Config(model_size='large')
        task_cfg = TaskConfig(model_size='large')
        vatlm = MyVATLM(
            cfg=cfg,
            task_cfg=task_cfg,
            dictionaries=None,
        )

    state = torch.load(ckpt_path, map_location=torch.device('cpu'))
    pretrained_dict = state['model']
    model_dict = vatlm.state_dict()
    match_dict = {name: params for name, params in pretrained_dict.items() if name in model_dict}
    match_dict = {name: params for name, params in match_dict.items() if not ('final_proj' in name)}
    vatlm.load_state_dict(match_dict, strict=False)

    for name, param in vatlm.named_parameters():
        if not torch.equal(param, pretrained_dict[name]):
            print(name)

    ckpt_path_new = Path(ckpt_path)
    new_filename = ckpt_path_new.stem + '_torch'
    ckpt_path_new = str(ckpt_path_new).replace(ckpt_path_new.stem, new_filename).replace(ckpt_path_new.suffix, '.ckpt')
    torch.save({'vatlm': vatlm.state_dict(),}, ckpt_path_new)


if __name__ == '__main__':
    main()