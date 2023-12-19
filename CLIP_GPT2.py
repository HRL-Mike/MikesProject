
import torch
import torch.nn as nn

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers import GPT2Model
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPGPT2Classification(nn.Module):
    def __init__(self, num_class=18, model_subver='v0', vis_pos_emb=None):
        super(CLIPGPT2Classification, self).__init__()

        self.sub_ver = model_subver
        self.vis_pos_emb = vis_pos_emb

        # prepare CLIP encoders (visual and text)
        config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32", projection_dim=768)
        model = CLIPModel(config)

        self.config = model.config  # 获取模型的配置信息, output_attentions, output_hidden_states
        self.text_model = model.text_model  # 获取文本编码模型
        self.vision_model = model.vision_model  # 获取图像编码模型
        self.visual_projection = nn.Linear(model.visual_projection.in_features, 768)
        self.text_projection = nn.Linear(model.text_projection.in_features, 768)
        # self.visual_projection = model.visual_projection  # 获取视觉嵌入的投影层
        # self.text_projection = model.text_projection  # 获取文本嵌入的投影层
        self.logit_scale = model.logit_scale  # 获取缩放因子，用于调整 logits

        # prepare GPT2 decoder
        # GPT2 visual_cotext_aware_decoder
        self.VCA_decoder = GPT2Model.from_pretrained('gpt2')

        # intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)
        # self.LayerNorm = nn.BatchNorm1d(512)
        # Batch normalization requires at least two data points in each batch to calculate the batch mean and variance,
        # which are used for normalization.
        self.LayerNorm = nn.LayerNorm(512)  # use this one if only one data point in each batch
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self,
        input_ids=None,  # 非空, 文本id的数组
        pixel_values=None,  # 非空, 图像像素值的数组
        attention_mask=None,  # 非空, 注意力掩码
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
        image_embeds = vision_outputs[0]  # why vision_outputs[1] but vision_outputs[0]?
        # print(vision_outputs[0].shape)  # torch.Size([1, 50, 768]) = [batch_size, sequence_length, hidden_size]
        # the last hidden state of the vision model;
        # the final layer's output for each patch of the image, including the [CLS] token. 50 = 49 patches + 1 [CLS]
        # print(vision_outputs[1].shape)  # torch.Size([1, 768])
        # the pooled output; used as a comprehensive representation of the entire image.
        # the pooled output is generally derived from the [CLS] token's embedding but may undergo additional processing,
        # such as a layer normalization or a linear transformation, depending on the model's design
        # 所以 vision_outputs[0][:, 0, :] 和 vision_outputs[1] 可能是不一样的, 取决于 vision_outputs[1] 有没有进行额外处理
        image_embeds = self.visual_projection(image_embeds)  # 对图像嵌入进行投影处理
        # print('image_embeds: ', image_embeds.shape)
        # torch.Size([1, 512]) --> torch.Size([1, 768]) --> torch.Size([1, 50, 768])
        text_embeds = text_outputs[0]
        # print(text_outputs[0].shape)  # torch.Size([1, 7, 512])
        # print(text_outputs[1].shape)  # torch.Size([1, 512])
        text_embeds = self.text_projection(text_embeds)
        # print('text_embeds: ', text_embeds.shape)
        # torch.Size([1, 512]) --> torch.Size([1, 768]) --> torch.Size([1, 7, 768])

        batch_size = image_embeds.shape[0]  # 1
        visual_seq_len = image_embeds.shape[1]  # 50
        text_seq_len = text_embeds.shape[1]  # 7

        # get text and visual attention mask
        text_attention_mask = attention_mask
        # print(text_attention_mask.shape)  # torch.Size([1, 7])
        visual_attention_mask = torch.ones((batch_size, visual_seq_len), dtype=torch.float)
        # print(visual_attention_mask.shape)  # torch.Size([1, 50])
        inputs_attention_mask = torch.cat((text_attention_mask, visual_attention_mask), dim=1)
        # print(inputs_attention_mask.shape)  # torch.Size([1, 57])

        # concatenate text and visual embeddings (text first)
        inputs_embeds = torch.cat((text_embeds, image_embeds), dim=1)
        # print(inputs_embeds.shape)  # torch.Size([1, 57, 768])
        # in surgicalGPT, # torch.Size([40, 25, 768]) + torch.Size([40, 49, 768]) = torch.Size([40, 74, 768])

        # decode
        decoder_output = self.VCA_decoder(inputs_embeds=inputs_embeds, attention_mask=inputs_attention_mask)

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
output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
               pixel_values=inputs['pixel_values'])
print(output)
# tensor([[ 0.5306, -0.1376, -0.5758,  0.6965,  0.0882,  0.0663, -0.5692,  0.1317,
#           0.0770, -1.3769,  0.2197, -0.0885,  0.0443,  1.2386, -0.7299, -0.7493,
#           0.2236,  0.0422]], grad_fn=<AddmmBackward0>)
print(output.shape)  # torch.Size([1, 18])
# we can then use labels and gradient descent to optimize the model

# shape of some key variables in SurgicalGPT_v2
# question_embeds:  torch.Size([40, 25, 768])
# visual_embeds:  torch.Size([40, 49, 768])
# inputs_embeds:  torch.Size([40, 74, 768])

# question_attention_mask:  torch.Size([40, 25])
# visual_attention_mask:  torch.Size([40, 49])
# attention_mask:  torch.Size([40, 74])  # 这里的74和前面inputs_embeds的74要保持一致

# question_id_type:  torch.Size([40, 25])
# visual_id_type:  torch.Size([40, 49])
# token_type_ids:  torch.Size([40, 74])

# question_position_id:  torch.Size([40, 25])
# visual_position_id:  torch.Size([40, 49])
# position_ids:  torch.Size([40, 74])
