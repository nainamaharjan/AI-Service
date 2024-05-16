from fastapi import FastAPI, HTTPException
from entities.ai_entities import TaskType
from service.llm_service import LlmService
app = FastAPI()
service = LlmService()

def check_grammar(text):
    response = service.inference(task_type=TaskType.GRAMMAR_CHECK, input_text=text)
    return response


def elaborate(text):
    response = service.inference(task_type=TaskType.ELABORATE, input_text=text)
    return response


def shorten(text):
    response = service.inference(task_type=TaskType.SHORTEN, input_text=text)
    return response

def professional(text):
    response = service.inference(task_type=TaskType.PROFESSIONAL, input_text=text)
    return response

def casual(text):
    response = service.inference(task_type=TaskType.CASUAL, input_text=text)
    return response


def get_response_from_llm(input_text):
    response = service.inference_stream(input_text=input_text)
    reply = ""
    for res in response:
        reply = reply + res
    return reply

@app.get("/check-grammar")
def read_root(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")

    # Replace 'temp_file_path' with the actual path where the audio file is stored
    response = check_grammar(text)
    return {"response": response}


@app.get("/convert-to-professional")
def read_root(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Audio file path is required")
    response = professional(text)
    return {"response": response}


@app.get("/convert-to-casual")
def read_root(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    response = casual(text)
    return {"response": response}


@app.get("/response-from-llm")
def read_root(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    llm_response = get_response_from_llm(input_text=text)
    return {"response": llm_response}


@app.get("/elaborate-text")
def read_root(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    response = elaborate(text)
    return {"response": response}


@app.get("/shorten-text")
def read_root(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    summarized_text = shorten(text)
    return {"response": summarized_text}
