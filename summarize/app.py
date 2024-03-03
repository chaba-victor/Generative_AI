from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
from peft import PeftModel
model = PeftModel.from_pretrained(original_model, "prompt")
import streamlit as st


st.title('Skimlit')
txt=st.text_area('Text to analyze', '''Wikis are enabled by wiki software, otherwise known as wiki engines. A wiki engine, being a form of a content management system, differs from other web-based systems such as blog software, in that the content is created without any defined owner or leader, and wikis have little inherent structure, allowing structure to emerge according to the needs of the users.[1] Wiki engines usually allow content to be written using a simplified markup language and sometimes edited with the help of a rich-text editor.[2] There are dozens of different wiki engines in use, both standalone and part of other software, such as bug tracking systems. Some wiki engines are free and open-source, whereas others are proprietary. Some permit control over different functions (levels of access); for example, editing rights may permit changing, adding, or removing material. Others may permit access without enforcing access control. Further rules may be imposed to organize content.''')

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
if st.button('skimmed'):
  get_results(txt)