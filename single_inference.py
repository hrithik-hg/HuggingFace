!pip install transformers

!pip install  SentencePiece

"""Use of generate() is advised during inference. This approach takes care of encoding the input, feeding the encoded hidden states to the decoder through cross-attention layers, and producing the decoder output auto-regressively."""

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
