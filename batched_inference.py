!pip install transformers

!pip install  SentencePiece

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

task_prefix = "translate English to German: "
# use different length sentences to test batching
sentences = ["The house is wonderful.", "I do not like to work in NYC."]

inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

print("English to German translation of ['''The house is wonderful.''', '''I do not like to work in NYC.''' ]")
print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

"""Because T5 has been trained with the span-mask denoising objective, it can be used to predict the sentinel (masked-out) tokens during inference. The predicted tokens will then be placed between the sentinel tokens."""

