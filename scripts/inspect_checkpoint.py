import torch
state = torch.load("checkpoints/changeformer/ChangeFormer_LEVIR.pth", map_location="cpu", weights_only=False)
sd = state["model_G_state_dict"]
for k in sorted(sd.keys()):
    if k.startswith("TDec_x2."):
        print(f"{k}  ->  {tuple(sd[k].shape)}")