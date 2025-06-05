from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Any, Dict
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# https://python.langchain.com/v0.1/docs/modules/callbacks/
class CustomCallback(BaseCallbackHandler):

    def __init__(self):
        self.messages = {}
    
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        self.messages["on_llm_start_prompts"] = prompts    
        self.messages["on_llm_start_kwargs"] = kwargs    

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.messages["on_llm_end_response"] = response    
        self.messages["on_llm_end_kwargs"] = kwargs    


callback = CustomCallback()

endpoint_url = "http://mds-gpu-medinym.et.uni-magdeburg.de:9000"
llm =  HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=[callback]
)

prompt = PromptTemplate.from_template("Say {foo}")

chain =  prompt | llm | StrOutputParser()

chain.invoke({"foo": "hello"})

for key, value in callback.messages.items():
    print(key)

