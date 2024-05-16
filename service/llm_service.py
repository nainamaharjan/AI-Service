"""
   Created by: Naina Maharjan
   Created on: 2024-01-15
"""
import guidance
from guidance import models, system, user,assistant, gen, instruction
from service.prompt_collection import AiPrompts
from entities.ai_entities import Prompt, TaskType
import regex as re
import torch
import gc
from settings import MODEL


class LlmService():
    def __init__(self, device: str = "auto", **kwargs):
        self.device = device
        self.llm = models.TransformersChat(model=MODEL)

    def inference_stream(self, input_text: str = None, clear_cache: bool = True):
        if clear_cache:
            torch.cuda.empty_cache()
            gc.collect()
        reply_text = ""
        for output in self.llm.stream() + get_lm_for_ai_assistant(input_text=input_text):
            output = str(output)
            if "<|im_content|>" in output:
                output = output.split("<|im_content|>")[-1]
                output = output.strip("</s>").rstrip("}").lstrip()
                output = output.lstrip("<|im_end|>")
            if reply_text.strip() == "":
                reply_text = output
                print(reply_text, end='', flush=True)
                yield output

            response = output.replace(reply_text, "")
            print(output.replace(reply_text, ""), flush=True, end='')
            reply_text = output
            yield response



    @torch.inference_mode()
    def inference(self, input_text: str = None,
                  task_type: TaskType = None) -> str:

        print("Performing %s via Guidance for input %s", task_type, input_text)
        response = self.get_lm(task_type=task_type, input_text=input_text)
        response = self._post_process_response(str(response))
        print("The response is: %s", response)
        gc.collect()
        return response

    def get_lm_for_raw(self, prompt, input_text):
        lm = self.llm + prompt.replace("{{input_sentence}}", input_text)
        return lm


    def get_lm(self, input_text, task_type:TaskType):
        prompt = Prompt()
        if task_type == TaskType.CASUAL:
            prompt = AiPrompts.casual_prompt
        elif task_type == TaskType.SHORTEN:
            prompt = AiPrompts.shorten_prompt
        elif task_type == TaskType.ELABORATE:
            prompt = AiPrompts.elaborate_prompt
        elif task_type == TaskType.PROFESSIONAL:
            prompt = AiPrompts.professional_prompt
        elif task_type == TaskType.GRAMMAR_CHECK:
            prompt = AiPrompts.grammar_prompt
        with system():
            lm = self.llm + prompt.system_instruction
        with instruction():
            lm += prompt.system_instruction
        with user():
            lm += input_text
        with assistant():
            lm += gen(name=prompt.response)
        return lm[prompt.response]


    def _pre_process(self, input_text: str) -> str:
        clean_text = re.sub('<.*?>', '', input_text)
        return clean_text

    def _post_process_response(self, response:str) -> str:
        response = response.split("assistant")[-1]
        response = response.rstrip("<|im_end|>")
        return response

@guidance
def get_lm_for_ai_assistant(lm, input_text):
    with system():
        lm = lm + "You are a helpful ai assistant that answers user query in friendly and polite manner."
    with user():
        lm += input_text
    with assistant():
        lm += gen("response")
    return lm


