from .layers import *
from typing import Union,Optional,Callable

class FusionPhysBase(nn.Module):
    def __init__(self, input1:Union[SubmodalityEmbed,RFEncoder],input2:Optional[Union[SubmodalityEmbed,RFEncoder]],
                 blocks:PhysiologicalInteraction,predictor:Union[Predictor,PredictorF3],dim:int):
        super().__init__()
        self.input1=input1
        self.input2=input2
        self.fusion=blocks
        self.predictor = predictor
        self.type_embedding=None if self.input2 is None else nn.Embedding(2,dim)

    def forward(self,x1:torch.Tensor,x2:Optional[torch.Tensor]):
        x1=self.input1(x1)
        if self.input2 is not None:
            x2=self.input2(x2)
            x1=x1+self.type_embeddings(torch.zeros(x1.shape[0],x1.shape[2],x1.shape[3],dtype=torch.long,device='cuda')).permute((0,3,1,2))
            x2=x2+self.type_embeddings(torch.ones(x1.shape[0],x1.shape[2],x1.shape[3],dtype=torch.long,device='cuda')).permute((0,3,1,2))

        x = torch.concat([x1, x2], dim=-1) if self.input2 is not None else x1
        x=self.fusion(x)
        x=self.predictor(x)
        return x

    @property
    def is_fusion(self):
        return False if self.input2 is None else True

class FusionPhys:
    @staticmethod
    def camera_only(in_ch:int,dim:int=64):
        return FusionPhysBase(
            SubmodalityEmbed(in_ch,dim),
            None,
            PhysiologicalInteraction(dim),
            Predictor(),
            dim
        )

    @staticmethod
    def camera_camera_fusion(in_chs:Tuple[int,int],dim:int=64):
        return FusionPhysBase(
            SubmodalityEmbed(in_chs[0],dim),
            SubmodalityEmbed(in_chs[1],dim),
            PhysiologicalInteraction(dim),
            PredictorF3(),
            dim
        )

    @staticmethod
    def camera_rf_fusion(in_chs:Tuple[int,int],dim:int=64):
        return FusionPhysBase(
            SubmodalityEmbed(in_chs[0],dim),
            RFEncoder(in_chs[1]),
            PhysiologicalInteraction(dim),
            Predictor(),
            dim
        )