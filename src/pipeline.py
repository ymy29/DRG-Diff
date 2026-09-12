from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline, rescale_noise_cfg, retrieve_timesteps

class Pipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        image_encoder=None,
        feature_extractor=None,
        force_zeros_for_empty_prompt=True,
        add_watermarker=None
    ):
        super().__init__(
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            scheduler,
            image_encoder,
            feature_extractor,
            force_zeros_for_empty_prompt,
            add_watermarker,
        )
 
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        concept_images: torch.Tensor = None,
        boxes: torch.Tensor = None,
        phrases: List[List[str]] = None,
        phrase_idxes: List[List[List[int]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        use_updated_noise_for_scheduler: bool = False,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)


        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device


        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )


        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)


        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        negative_add_time_ids = negative_add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if boxes is not None:
            boxes = torch.tensor(boxes).to(self.device, prompt_embeds.dtype)
            if phrases is not None:
                drop_grounding_tokens = [0] * batch_size
                batch_boxes = boxes.view(batch_size * boxes.shape[1], -1)

                phrase_input_ids = []
                for phrase in phrases:
                    if not isinstance(phrase, list):
                        phrase = list(phrase)
                    phrase_input_id = self.tokenizer(phrase, 
                                                     max_length=self.tokenizer.model_max_length,
                                                     padding="max_length", 
                                                     truncation=True,
                                                     return_tensors="pt").input_ids
                    phrase_input_ids.append(phrase_input_id)
                phrase_input_ids = torch.stack(phrase_input_ids)
                phrase_input_ids = phrase_input_ids.view(-1, phrase_input_ids.shape[-1])
                phrase_embeds = self.text_encoder(phrase_input_ids.to(self.device)).pooler_output
                grounding_kwargs = {"boxes": batch_boxes, "phrase_embeds": phrase_embeds, "drop_grounding_tokens": drop_grounding_tokens}
            else:
                grounding_kwargs = None
                
            boxes = torch.repeat_interleave(boxes, repeats=num_images_per_prompt, dim=0)
            uncond_boxes = torch.zeros_like(boxes)
            boxes = torch.cat([uncond_boxes, boxes], dim=0)
            cross_attention_kwargs = {"boxes": boxes}

        if phrase_idxes is not None:
            phrase_idxes = torch.tensor(phrase_idxes).to(self.device, torch.int)
            phrase_idxes = torch.repeat_interleave(phrase_idxes, repeats=num_images_per_prompt, dim=0)
            uncond_phrase_idxes = torch.zeros_like(phrase_idxes)
            phrase_idxes = torch.cat([uncond_phrase_idxes, phrase_idxes], dim=0)
            cross_attention_kwargs["phrase_idxes"] = phrase_idxes

        concept_images = concept_images.to(device, prompt_embeds.dtype)

        intermediate_feats = self.image_encoder(
            concept_images,
            grounding_kwargs=grounding_kwargs,
            return_intermediate=True
        )
        first_gcn_cls = intermediate_feats["first_gcn_updated_cls"]
        first_gcn_all_tokens = intermediate_feats["first_gcn_updated_all_tokens"]
        num_subjects = intermediate_feats["num_subjects"]

        text_prompt_embeds = prompt_embeds.clone()
        text_negative_prompt_embeds = negative_prompt_embeds.clone()

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                batch_size = latents.shape[0]

                updated_cond_latents, updated_subject_nodes, _ = self.image_encoder.noise_cls_gcn_forward(
                    latents,
                    first_gcn_all_tokens.repeat(batch_size // first_gcn_all_tokens.shape[0], 1, 1)
                )

                fused_subject_nodes, _ = self.image_encoder.subject_fusion_net(
                    first_gcn_all_tokens.repeat(batch_size // first_gcn_all_tokens.shape[0], 1, 1),
                    updated_subject_nodes
                )

                image_prompt_embeds = self.image_encoder.resample_with_updated_cls(
                    fused_subject_nodes,
                    num_subjects,
                    grounding_kwargs
                ).to(self.dtype)
                image_prompt_embeds = image_prompt_embeds.view(batch_size, -1, image_prompt_embeds.shape[-1])

                if self.do_classifier_free_guidance:
                    cond_prompt_embeds = torch.cat([
                        text_prompt_embeds.repeat(batch_size // text_prompt_embeds.shape[0], 1, 1),
                        image_prompt_embeds
                    ], dim=1)

                    uncond_prompt_embeds = torch.cat([
                        text_negative_prompt_embeds.repeat(batch_size // text_negative_prompt_embeds.shape[0], 1, 1),
                        torch.zeros_like(image_prompt_embeds)
                    ], dim=1)

                    prompt_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0)

                    latent_model_input = torch.cat([latents, updated_cond_latents], dim=0)
                else:
                    prompt_embeds = torch.cat([
                        text_prompt_embeds.repeat(batch_size // text_prompt_embeds.shape[0], 1, 1),
                        image_prompt_embeds
                    ], dim=1)
                    latent_model_input = updated_cond_latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:

                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                scheduler_latents = updated_cond_latents if use_updated_noise_for_scheduler else latents

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, scheduler_latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    self.vae = self.vae.to(latents.dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        
        return StableDiffusionXLPipelineOutput(images=image)