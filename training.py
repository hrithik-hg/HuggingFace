pip install transformers;

pip install  SentencePiece

"""#Training

T5 is an encoder-decoder model that converts all natural language processing (NLP) difficulties into text-to-text format. It is trained by instructor coercion. This means that we always require an input sequence and a corresponding target sequence for training. The model is supplied the input sequence through input ids. The target sequence is moved to the right, i.e., a start-sequence token is appended, and provided to the decoder using the decoder input ids. The target sequence is then added by the EOS token and corresponds to the labels in teacher-forcing manner. The PAD token is utilised as the start-sequence token in this case. T5 may be trained and fine-tuned both supervised and unsupervised.

#Unsupervised denoising training

In this setup, spans of the input sequence are masked by so-called sentinel tokens (a.k.a unique mask tokens) and the output sequence is formed as a concatenation of the same sentinel tokens and the real masked tokens. Each sentinel token represents a unique mask token.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
print("By doing unsupervised training the loss is:-",loss.item())

"""#Supervised Training

In this setup, the input sequence and output sequence are a standard sequence-to-sequence input-output mapping.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
print("By doing supervised training the loss is:-",loss.item())

"""However, the above example only presents one training example. In reality, deep learning models are trained in batches. As a result, we must pad/truncate instances to the same length. In encoder-decoder models, the max source length and max target length parameters set the maximum length of the input and output sequences, respectively (otherwise they are truncated). Depending on the work, they should be properly set."""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 128

# Suppose we have the following 2 training examples:
input_sequence_1 = "Welcome to NYC"
output_sequence_1 = "Bienvenue à NYC"

input_sequence_2 = "HuggingFace is a company"
output_sequence_2 = "HuggingFace est une entreprise"

# encode the inputs
task_prefix = "translate English to French: "
input_sequences = [input_sequence_1, input_sequence_2]

encoding = tokenizer(
    [task_prefix + sequence for sequence in input_sequences],
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# encode the targets
target_encoding = tokenizer(
    [output_sequence_1, output_sequence_2], padding="longest", max_length=max_target_length, truncation=True
)
labels = target_encoding.input_ids

# replace padding token id's of the labels by -100 so it's ignored by the loss
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
print("Loss by using two hyperparameters:-",loss.item()
