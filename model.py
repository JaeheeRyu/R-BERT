import torch
import torch.nn as nn
from transformers import BertModel,AutoConfig, AutoModel, BertPreTrainedModel


### model architecture
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT_RobertaForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_classes, dr_rate=0.1):
        super(RBERT_RobertaForSequenceClassification, self).__init__()

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = config.hidden_size
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.num_classes = num_classes
        self.dropout_rate = dr_rate

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, self.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, self.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size * 3, self.num_classes, self.dropout_rate,
                                        use_activation=False)

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask):
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)

        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector