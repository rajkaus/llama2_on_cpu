from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers    #for quantization models, need to import CTransformers  instead of Transformer
from src.helper import *

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


instruction = "Convert the following text from English to Hindi: \n\n {text}"


SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS    #if you do not wish to get extra output messages then use CUSTOM_PROMPT defined in helper.py
template = B_INST + SYSTEM_PROMPT + instruction + E_INST


prompt = PromptTemplate(template=template, input_variables=["text"])


llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )


LLM_Chain=LLMChain(prompt=prompt, llm=llm)

print(LLM_Chain.run("How are you?"))