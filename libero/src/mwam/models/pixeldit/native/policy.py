"""Native flow matching, including the explicit independent-time recipe."""
from __future__ import annotations
import torch
from ..t2i_mwam import T2IFastWAM
from ..t2i_libero import LiberoT2IFastWAM
from .interaction import NativeMoT


class NativePolicy(T2IFastWAM):
    # Keep the real prefix of an episode-tail action chunk. The existing action
    # objective divides by its valid-token count, never by the padded horizon.
    ALLOW_ACTION_PADDING = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        # Legacy checkpoints append proprio to Action trajectories.
        if not self.action_expert.proprio_in_context:
            self.proprio_encoder = None

    @staticmethod
    def _resolve_method_contract(value):
        if value is None or value.get("schema_version") != "mwam_native_t2i_method_v1":
            raise ValueError("Native T2I method contract required")
        return dict(value)

    def _validate_contract_alignment(self):
        if not isinstance(self.dit,NativeMoT): raise TypeError("NativeMoT required")
        if (self.dit.world_depth,self.dit.pixel_depth,self.dit.action_depth)!=(14,2,16):
            raise ValueError("Native T2I requires 14+2 World and 16 Action blocks")
        if self.action_expert.action_dim != self.action_normalization_contract.feature_dim:
            raise ValueError("Action normalization dimension mismatch")
        if self.proprio_dim != self.proprio_normalization_contract.feature_dim:
            raise ValueError("Proprio normalization dimension mismatch")
        if self.normalization_contract.future_offsets != self.kflow_contract.future_offsets:
            raise ValueError("Flow normalization horizon mismatch")
        expected = "independent" if self.action_expert.proprio_in_context else "shared"
        if self.timestep_coupling != expected: raise ValueError("Timestep coupling differs from native recipe")

    def _prepare_context(self, context, context_mask, proprio, *, batch_size, include_proprio=True):
        if self.action_expert.proprio_in_context:
            ac, mask, _, _ = super()._prepare_context(context,context_mask,proprio,
                batch_size=batch_size,include_proprio=include_proprio)
            return ac, mask, ac, mask
        return super()._prepare_context(context,context_mask,None,batch_size=batch_size,include_proprio=False)

    def _prepare_training_sample(self, sample, *, include_action=True):
        inputs = super()._prepare_training_sample(sample,include_action=include_action)
        if include_action:
            raw = sample["proprio"].to(device=self.device,dtype=torch.float32)
            inputs["proprio_normalized"] = self.proprio_normalization_contract.normalize(raw)
        return inputs

    @staticmethod
    def model_timestep(timestep, *, scheduler):
        sigma = timestep.float()/float(scheduler.num_train_timesteps)
        if not bool(torch.isfinite(sigma).all()) or bool(((sigma<0)|(sigma>1)).any()):
            raise ValueError("Scheduler sigma must be finite and in [0,1]")
        return sigma * 1000.

    _pixel_clean_timestep = model_timestep
    _action_noise_fraction_timestep = model_timestep


    def _compute_momap_velocity_loss(self,prediction,target_velocity,*,inputs,timestep_weight):
        if self.world_expert.views==2:
            return LiberoT2IFastWAM._compute_momap_velocity_loss(self,prediction,target_velocity,
                inputs=inputs,timestep_weight=timestep_weight)
        return super()._compute_momap_velocity_loss(prediction,target_velocity,inputs=inputs,timestep_weight=timestep_weight)


    def compute_joint_loss(self,sample,*,momap_noise=None,action_noise=None,
                           momap_timestep=None,action_timestep=None,return_audit=False,tiled=False):
        if tiled or self.training_stage!="joint":
            raise ValueError("Native T2I entrypoint uses untiled joint training")
        inputs = self._prepare_training_sample(sample)
        target = self._normalize_target_flow_for_training(inputs["target_flow_meters"],
            sample_scale_meters=inputs["source_geometry_scale_meters"],inputs=inputs)
        action = inputs["action_normalized"]
        wn = self._coerce_noise(momap_noise,reference=target,name="momap_noise")
        an = self._coerce_noise(action_noise,reference=action,name="action_noise")
        tw,ta = self._resolve_training_timesteps(batch_size=action.shape[0],dtype=torch.float32,
            momap_timestep=momap_timestep,action_timestep=action_timestep)
        wb = self.momap_objective.prepare(target,noise=wn,timestep=tw)
        noised_action = self.train_action_scheduler.add_noise(action,an,ta)
        av = self.train_action_scheduler.training_target(action,an,ta)
        aw = self.train_action_scheduler.training_weight(ta).to(self.device)
        mt = self.model_timestep(tw,scheduler=self.train_world_scheduler)
        mt_action = self.model_timestep(ta,scheduler=self.train_action_scheduler)
        outputs = self.dit(world_condition=inputs["world_condition"],noised_flow=wb.noised_flow,
            t_world=mt,noised_action=noised_action,t_action=mt_action,context=inputs["context"],
            context_mask=inputs["context_mask"],proprio=inputs["proprio_normalized"],
            world_context=inputs["world_context"],world_context_mask=inputs["world_context_mask"],
            source_support_mask=inputs["source_support_mask"])
        wl = self._compute_momap_velocity_loss(outputs["world_velocity"],wb.target_velocity,inputs=inputs,timestep_weight=wb.timestep_weight)
        ar,al,ap = self._action_losses(outputs["action_velocity"],av,inputs["action_is_pad"],aw)
        std = torch.tensor(self.action_normalization_contract.std,device=action.device)
        adr,adl,_ = self._action_losses(outputs["action_velocity"],av,inputs["action_is_pad"],aw,feature_scale=std)
        loss = self.loss_lambda_momap*wl.weighted+self.loss_lambda_action*al
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Native joint loss is not finite")
        values = {"loss/total":loss,"loss/momap_raw":wl.raw,"loss/momap_weighted":wl.weighted,
            "loss/action_raw":ar,"loss/action_weighted":al,"loss/action_denormalized_raw":adr,
            "loss/action_denormalized_weighted":adl,"fm_timestep/momap":tw.mean(),
            "fm_timestep/action":ta.mean(),"fm_sigma/world":(tw/self.train_world_scheduler.num_train_timesteps).mean(),
            "fm_sigma/action":(ta/self.train_action_scheduler.num_train_timesteps).mean(),
            "model_timestep/world":mt.mean(),"model_timestep/action":mt_action.mean(),"valid_ratio/by_horizon":wl.valid_ratio_by_horizon.mean()}
        metrics={k:float(v.detach()) for k,v in values.items()}
        metrics.update(getattr(self,"_last_hybrid_world_loss_metrics",{}))
        if return_audit:
            return loss,metrics,dict(model_timestep=mt.detach(),model_timestep_action=mt_action.detach(),world_timestep=tw.detach(),
                action_timestep=ta.detach(),predicted_momap_velocity=outputs["world_velocity"].detach(),
                predicted_action_velocity=outputs["action_velocity"].detach())
        return loss,metrics
