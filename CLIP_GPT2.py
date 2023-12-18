
import torch
import torch.nn as nn

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import GPT2Model
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPGPT2Classification(nn.Module):
    def __init__(self, num_class=18, model_subver='v0', vis_pos_emb=None):
        super(CLIPGPT2Classification, self).__init__()

        self.sub_ver = model_subver
        self.vis_pos_emb = vis_pos_emb

        # prepare CLIP encoders (visual and text)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = model.config  # 获取模型的配置信息, output_attentions, output_hidden_states
        self.text_model = model.text_model  # 获取文本编码模型
        self.vision_model = model.vision_model  # 获取图像编码模型
        self.visual_projection = model.visual_projection  # 获取视觉嵌入的投影层
        self.text_projection = model.text_projection  # 获取文本嵌入的投影层
        self.logit_scale = model.logit_scale  # 获取缩放因子，用于调整 logits

        # prepare GPT2 decoder
        # GPT2 visual_cotext_aware_decoder
        self.VCA_decoder = GPT2Model.from_pretrained('gpt2')

        # intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self,
        input_ids=None,  # 非空, 文本id的数组
        pixel_values=None,  # 非空, 图像像素值的数组
        attention_mask=None,  # 非空, 注意力掩码
        patch_num=None,
        position_ids=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)

        # use CLIP visual and text models to process data
        vision_outputs = self.vision_model(  # 使用视觉模型处理图像数据
            pixel_values=pixel_values,  # 图像的像素值数组
            output_attentions=output_attentions,  # True/False, 用来控制是否输出模型的注意力机制的细节
            output_hidden_states=output_hidden_states,  # True/False, 用来控制是否输出模型中间层的隐藏状态
            return_dict=return_dict,  # True/False, 被设置为True时, 模型的输出将被封装在一个字典中
        )
        text_outputs = self.text_model(  # 使用文本模型处理文本数据
            input_ids=input_ids,  # 文本ID的数组; 这个数组是模型输入文本的主要形式
            attention_mask=attention_mask,  # 0/1数组; 用于指示哪些部分的 input_ids 应该被模型考虑, 哪些部分是填充 (应该被模型忽略)
            position_ids=position_ids,  # 通常是整数数组, 用于表示输入中每个 token 的位置信息; 如果不提供, 模型通常会自动生成一个默认的位置编码
            output_attentions=output_attentions,  # True/False
            output_hidden_states=output_hidden_states,  # True/False
            return_dict=return_dict,  # True/False
        )

        # get visual and text embeddings
        image_embeds = vision_outputs[1]  # why vision_outputs[1] but vision_outputs[0]?
        image_embeds = self.visual_projection(image_embeds)  # 对图像嵌入进行投影处理
        # print(image_embeds.shape)  # torch.Size([1, 512])
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        # print(text_embeds.shape)  # torch.Size([1, 512])

        # get text and visual attention mask
        text_attention_mask = attention_mask
        # print(text_attention_mask.shape)  # torch.Size([1, 7])
        batch_size = text_embeds.shape[0]
        visual_attention_mask = torch.ones((batch_size, patch_num), dtype=torch.float)
        # print('image_embeds: ', image_embeds.shape)  # image_embeds:  torch.Size([1, 512])
        attention_mask = torch.cat((text_attention_mask, visual_attention_mask), dim=1)  # torch.Size([1, 56])

        # concatenate text and visual embeddings (text first)
        inputs_embeds = torch.cat((text_embeds, image_embeds), dim=1)  # torch.Size([1, 1024])
        # need 768 instead of 1024, how to do that?

        # decode
        decoder_output = self.VCA_decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        decoder_output = decoder_output.last_hidden_state.swapaxes(1, 2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output, 1)
        decoder_output = decoder_output.swapaxes(1, 2).squeeze(1)

        # intermediate layers
        out = self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        return out


model = CLIPGPT2Classification()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('./cat.jpg')

inputs = processor(text=[" a photo of a cat"], images=image, return_tensors="pt", padding=True)
# print(inputs['input_ids'].shape)  # torch.Size([1, 7])
# print(inputs['pixel_values'].shape)  # torch.Size([1, 3, 224, 224])
# print(inputs['attention_mask'].shape)  # torch.Size([1, 7])
# print(inputs['attention_mask'])  # tensor([[1, 1, 1, 1, 1, 1, 1]])
visual_patch_num = int((224*224) / (32*32))
output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
               pixel_values=inputs['pixel_values'], patch_num=visual_patch_num)

