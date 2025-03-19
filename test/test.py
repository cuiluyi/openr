from transformers import DistilBertConfig, AutoModel

default_config = DistilBertConfig()
default_pretrained_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased")
my_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased", activation="relu", attention_dropout=0.4)


print(default_config)
print(default_pretrained_config)
print(my_config)