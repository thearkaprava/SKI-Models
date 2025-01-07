from typing import List, Optional, Tuple, Union
import torch
import pickle
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from ..constants import * # contains the default tokens for the modality prefixes

# DEFAULT_VIDEO_TOKEN = "<video>"
# DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
# DEFAULT_VID_START_TOKEN = "<vid_start>"
# DEFAULT_VID_END_TOKEN = "<vid_end>"

# DEFAULT_POSE_TOKEN = "<human_pose>"
# DEFAULT_POSE_PATCH_TOKEN = "<pose_patch>"
# DEFAULT_POSE_START_TOKEN = "<pose_start>"
# DEFAULT_POSE_END_TOKEN = "<pose_end>"

# DEFAULT_OBJECT_TOKEN= "<object>"
# DEFAULT_OBJECT_PATCH_TOKEN = "<object_patch>"
# DEFAULT_OBJECT_START_TOKEN = "<object_start>"
# DEFAULT_OBJECT_END_TOKEN = "<object_end>"



class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024 # the shape of the features from the vision encoder
        self.hidden_size_pose = 216 # the shape of the features from the vision encoder
        self.hidden_size_object= 512
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None
        self.object_start_token = None
        self.object_end_token = None
        self.object_patch_token = None
        self.pose_start_token = None
        self.pose_end_token = None
        self.pose_patch_token = None

class SKILVLMConfig(LlamaConfig):
    model_type = "SKILVLM"


class SKILVLMLlamaModel(LlamaModel):
    config_class = SKILVLMConfig

    def __init__(self, config: LlamaConfig, modality_args, mm_vision_tower=None, mm_hidden_size=None):  # TODO: Remove unused params
        super(SKILVLMLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        # initialize the projectors
        if hasattr(config, "use_mm_proj") and modality_args is not None:
            if modality_args['video']:
                self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

            if modality_args['object']:
                self.mm_projector_forobject = nn.Linear(self.vision_config.hidden_size_object, self.config.hidden_size)

            if modality_args['pose']:
                self.mm_projector_forpose = nn.Linear(self.vision_config.hidden_size_pose, self.config.hidden_size)
        else: # at inference
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, modalities_to_use, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True

        # Load the pretrained weights for the multimodal projector
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            # intialize the multimodal projectors and load weights if passed
            if modalities_to_use['video']:
                if pretrain_mm_mlp_adapter is not None:
                    self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if ('pose' not in k and 'object' not in k)})

            if modalities_to_use['object']:
                if pretrain_mm_mlp_adapter is not None:
                    self.mm_projector_forobject.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if 'object' in k})

            if modalities_to_use['pose']:
                if pretrain_mm_mlp_adapter is not None:
                    self.mm_projector_forpose.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if 'pose' in k})

        return dict(
            video_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            pose_features: Optional[torch.FloatTensor] = None,
            object_features:Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print('embed_tokens.weight.requires_grad = ', self.model.embed_tokens.weight.requires_grad)
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (input_ids.shape[1] != 1 or self.training):
            if video_spatio_temporal_features is not None:
                video_features = self.mm_projector(video_spatio_temporal_features)

            if object_features is not None:
                object_features_projected = self.mm_projector_forobject(object_features)

            if pose_features is not None:
                pose_features_projected = self.mm_projector_forpose(pose_features)
        
            new_input_embeds = []
            cur_video_idx = 0

            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                '''
                Initially, the input_embeds are the embeddings of the tokens in the input_ids. This means that the special video and pose tokens (e.g., <vid_patch> and <pose_start>) were processed like words (and the embeddings are the representations from the model).

                Below, we replace the embeddings of the video and pose tokens with the actual video and pose features. We do this by finding the positions of the video and pose tokens in the input_ids, and then replacing the embeddings of the tokens at those positions with the video and pose features.
                '''
                # if self.vision_config.use_vid_start_end: # Are we using modality token prefixes?
                start_token_positions, end_token_positions = [], []
                features_to_append = [] # will store the features to append to the input embeddings

                '''
                Add video features
                '''
                if video_spatio_temporal_features is not None:
                    cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                    
                    if self.vision_config.use_vid_start_end: # append start+end tokens and features
                        video_start_token_pos = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0].item()
                        video_end_token_pos = torch.where(cur_input_ids == self.vision_config.vid_end_token)[0].item()

                        features_to_append.append(cur_input_embeds[video_start_token_pos:video_start_token_pos + 1])
                        features_to_append.append(cur_video_features)
                        features_to_append.append(cur_input_embeds[video_end_token_pos:video_end_token_pos + 1])
                    else: # only append features
                        video_start_token_pos = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0][0].item()
                        video_end_token_pos = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0][-1].item()
                        features_to_append.append(cur_video_features)

                    start_token_positions.append(video_start_token_pos)
                    end_token_positions.append(video_end_token_pos)

                '''
                Add object features
                '''
                if object_features is not None:
                    cur_object_features = object_features_projected[cur_video_idx].to(device=cur_input_embeds.device)
                    # cant rely on the first dimension to get number of objects, it could be padded
                    num_object_patches = (cur_input_ids == self.vision_config.object_patch_token).sum().item()
                    num_objects_for_sample = num_object_patches // 8

                    if self.vision_config.use_vid_start_end:
                        object_start_token_idx = torch.where(cur_input_ids == self.vision_config.object_start_token)[0].item()
                        object_end_token_idx = torch.where(cur_input_ids == self.vision_config.object_end_token)[0].item()
                    else:
                        object_start_token_idx = torch.where(cur_input_ids == self.vision_config.object_patch_token)[0][0].item()
                        object_end_token_idx = torch.where(cur_input_ids == self.vision_config.object_patch_token)[0][-1].item()

                    # incase there are tokens between the end of the last modality and the start of the current modality
                    # this happens when use_modality_string_prefix is True
                    if len(start_token_positions) > 0 and object_start_token_idx - end_token_positions[-1] > 1:
                        features_to_append.append(cur_input_embeds[end_token_positions[-1]+1:object_start_token_idx])

                    start_token_positions.append(object_start_token_idx)
                    end_token_positions.append(object_end_token_idx)

                    # build object feature tensor
                    object_embed = torch.empty(0, 4096)
                    
                    n = 0
                    while n < num_objects_for_sample:
                        if object_embed.nelement() == 0:
                            if self.vision_config.use_vid_start_end: # add start token and features for 1st object
                                object_embed = torch.cat((
                                    cur_input_embeds[object_start_token_idx : object_start_token_idx + 1],
                                    cur_object_features[n*8:(n+1)*8]
                                ), dim=0)
                            else: # add features for 1st object
                                object_embed = cur_object_features[n*8:(n+1)*8]
                        # add the rest of the object features
                        else:
                            object_embed = torch.cat((
                                object_embed,
                                cur_object_features[n*8:(n+1)*8]
                            ), dim=0)

                        n = n + 1

                    if self.vision_config.use_vid_start_end: # add object end token
                        object_embed = torch.cat((
                            object_embed,
                            cur_input_embeds[object_end_token_idx: object_end_token_idx + 1]
                        ), dim=0)

                    object_embed = object_embed.to(cur_input_embeds.device)

                    features_to_append.append(object_embed)
                # breakpoint()
                '''
                Add pose features
                '''
                if pose_features is not None:
                    cur_pose_features = pose_features_projected[cur_video_idx].to(device=cur_input_embeds.device)

                    if self.vision_config.use_vid_start_end:
                        pose_start_token_idx = torch.where(cur_input_ids == self.vision_config.pose_start_token)[0].item()
                        pose_end_token_idx = torch.where(cur_input_ids == self.vision_config.pose_end_token)[0].item()
                    else:
                        pose_start_token_idx = torch.where(cur_input_ids == self.vision_config.pose_patch_token)[0][0].item()
                        pose_end_token_idx = torch.where(cur_input_ids == self.vision_config.pose_patch_token)[0][-1].item()

                    # incase there are tokens between the end of the last modality and the start of the current modality
                    # this happens when use_modality_string_prefix is True
                    if len(start_token_positions) > 0 and pose_start_token_idx - end_token_positions[-1] > 1:
                        features_to_append.append(cur_input_embeds[end_token_positions[-1]+1:pose_start_token_idx])

                    start_token_positions.append(pose_start_token_idx)
                    end_token_positions.append(pose_end_token_idx)

                    # add start token
                    if self.vision_config.use_vid_start_end:
                        features_to_append.append(cur_input_embeds[pose_start_token_idx:pose_start_token_idx + 1])

                    features_to_append.append(cur_pose_features)

                    # add end token
                    if self.vision_config.use_vid_start_end:
                        features_to_append.append(cur_input_embeds[pose_end_token_idx:pose_end_token_idx + 1])

                # error checking
                assert len(start_token_positions) != 0 and len(end_token_positions) != 0, "There should be at least one modality token. Cant train with only text"
                assert len(start_token_positions) == len(end_token_positions), "The number of start and end tokens should be the same."

                earliest_start_token_pos = min(start_token_positions)
                latest_end_token_pos = max(end_token_positions)

                features_to_append = torch.cat(features_to_append, dim=0)

                if orig_embeds_params is not None:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:earliest_start_token_pos].detach(), # everything before first modality token
                                                        features_to_append, # the features
                                                        cur_input_embeds[latest_end_token_pos + 1:].detach()), # everything after last modality token
                                                        dim=0)
                else:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:earliest_start_token_pos], # everything before first modality token
                                                        features_to_append, # the features
                                                        cur_input_embeds[latest_end_token_pos + 1:]), # everything after last modality token
                                                        dim=0)
                assert cur_new_input_embeds.shape[0] == input_ids.shape[1], f"Shapes dont match: {cur_new_input_embeds.shape[0]} != {input_ids.shape[1]}"
                
                cur_video_idx += 1

                new_input_embeds.append(cur_new_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        
        return super(SKILVLMLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class SKILVLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = SKILVLMConfig

    def __init__(self, config, model_args=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SKILVLMLlamaModel(config, modality_args=model_args)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            object_features: Optional[torch.FloatTensor] = None,
            pose_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features,
            object_features=object_features,
            pose_features=pose_features
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ## Padding or Clipping shift_logits
            # if shift_logits.shape[1] > shift_labels.shape[1]:
            #     shift_logits = shift_logits[:, :shift_labels.shape[1], :]
            # elif shift_logits.shape[1] < shift_labels.shape[1]:
            #     pad_size = shift_labels.shape[1] - shift_logits.shape[1]
            #     shift_logits = torch.nn.functional.pad(shift_logits, (0, 0, 0, pad_size), value=0)


            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
                "object_features": kwargs.get("object_features", None),
                "pose_features": kwargs.get("pose_features",None)
            }
        )

        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end

        # we dont care about the representation of these tokens since they are replaced with the actual features
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_OBJECT_PATCH_TOKEN, DEFAULT_POSE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer)) # 32003 -> 32006

        if mm_use_vid_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_OBJECT_START_TOKEN, DEFAULT_OBJECT_END_TOKEN, DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN], special_tokens=True)
            
            self.resize_token_embeddings(len(tokenizer)) # 32006 -> 32012

            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])
            vision_config.object_start_token, vision_config.object_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_OBJECT_START_TOKEN, DEFAULT_OBJECT_END_TOKEN])
            vision_config.pose_start_token, vision_config.pose_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data # self.embed_tokens.weight
                output_embeddings = self.get_output_embeddings().weight.data # self.lm_head.weight

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                
                # intialize new token embeddings with avg of existing embeddings
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # if finetuning the adapters
            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters(): # train self.embed_tokens
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters(): # dont train self.lm_head
                    p.requires_grad = False

            # if initializing the adapters with pretrained weights
            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                # input_embeddings = embed_tokens_weight # this is fine if the model we are loading from did not update any of the non-new token embeddings
                assert num_new_tokens == 6, f"Unexpected number of new tokens: {num_new_tokens}. All models using token modality prefixes should have 6 new tokens, i.e., start and end for video, object, and pose."
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        else:
            # if finetuning the adapters
            if tune_mm_mlp_adapter:
                if pretrain_mm_mlp_adapter:
                    input_embeddings = self.get_input_embeddings().weight.data # self.embed_tokens.weight
                    output_embeddings = self.get_output_embeddings().weight.data # self.lm_head.weight

                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                    embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

                    assert input_embeddings.shape == embed_tokens_weight.shape, f"Error loading weights from path passed in tune_mm_mlp_adapter. Unexpected embed_tokens_weight shape. From tune_mm_mlp_adapter: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}."
                    input_embeddings = embed_tokens_weight

                for p in self.get_input_embeddings().parameters(): # dont train self.embed_tokens
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters(): # dont train self.lm_head
                    p.requires_grad = False

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
        vision_config.object_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_OBJECT_PATCH_TOKEN])[0]
        vision_config.pose_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_POSE_PATCH_TOKEN])[0]

        # print the ids of all new tokens
        # print(tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_OBJECT_PATCH_TOKEN, DEFAULT_POSE_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_OBJECT_START_TOKEN, DEFAULT_OBJECT_END_TOKEN, DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN]))


AutoConfig.register("SKILVLM", SKILVLMConfig)
AutoModelForCausalLM.register(SKILVLMConfig, SKILVLMLlamaForCausalLM)