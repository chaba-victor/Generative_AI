from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
from peft import PeftModel
model = PeftModel.from_pretrained(original_model, "prompt")
import streamlit as st


st.title('Summarize')
txt=st.text_area('Text to analyze', '''''')

# Load saved model
def get_results(abs_text):
  dialogue = abs_text
  

  prompt = f"""
  Summarize the following conversation.

  {dialogue}

  Summary: """

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  input_ids = input_ids.to(next(model.parameters()).device)

  peft_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
  peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)


  print(dash_line)
  print(f'PEFT MODEL: {peft_model_text_output}')
  


print("\n\nAI formatted abstract is given below:\n")
if st.button('summarize'):
  get_results(txt)
