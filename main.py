import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import webbrowser
from urllib.parse import quote
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from RealtimeSTT import AudioToTextRecorder
import pyttsx3
import threading

# extra dependences:
# RealtimeSTT
# "RealtimeSTT[faster-whisper]"
# silero-vad
# packaging

# LOAD WEBSITES PRECONFIGURED URLS FROM JSON FILE

diretorio_atual = Path(__file__).resolve().parent
caminho_json = diretorio_atual / 'websites.json'

with open(caminho_json, "r", encoding="utf-8") as file:
    websites = json.load(file)


# FUNCTIONS

# Function to open a known website based on the keyword
def openlink(keyword):
    keyword = keyword.lower().strip()
    url = websites.get(keyword)

    if url:
        webbrowser.open(url)
        return json.dumps({
            "status": "success",
            "action": "open_website",
            "website": keyword,
            "url": url
        })

    return json.dumps({
        "status": "error",
        "message": f"I don't know the website '{keyword}'."
    })


# Function to perform a Google search based on the keyword
def onlysearch(keyword):
    url = f"https://www.google.com/search?q={quote(keyword)}"
    webbrowser.open(url)

    return json.dumps({
        "status": "success",
        "action": "search",
        "query": keyword
    })

# Function mapping for dynamic execution
available_functions = {
    "onlysearch": onlysearch,
    "openlink": openlink,
}


# GROQ CONFIGURATION & TOOLS

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

tools = [
    {
        "type": "function",
        "function": {
            "name": "onlysearch",
            "description": "Search for information on Google when the user wants to search for something on the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The exact words or phrase that should be searched on Google."
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openlink",
            "description": "Open a known website directly when the user explicitly wants to visit it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The name of the website to open, for example: youtube, spotify, github, google, or instagram."
                    }
                },
                "required": ["keyword"],
            },
        },
    }
]

system_instruction = (
    "You are 'kronos' an automation assistant. You have access to tools to search Google or open websites. "
    "Always use 'openlink' when the user explicitly asks to open a site, and 'onlysearch' when they want to search for something."
    "You should provide short responses and avoid unnecessary explanations, no long searches or explanations unless the user ask. If you don't know the answer, use 'onlysearch' to find it. "
    "provide short texts, dont say '*' or'_'or any other markdown formatting, just plain text. for a natural response"
)

# Conversation message history
messages = [
    {"role": "system", "content": system_instruction}
]

# Groq model with tool support
MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# FUNCTIONS WITH AUDIO SCOPE FIXED
# ============================================================





def ouvir_microfone_ultra_rapido(recorder_instance):
    print("\n[Kronos] Ouvindo... (Pode falar nativamente)")
    texto_transcrito = recorder_instance.text()
    return texto_transcrito.strip()


# Função auxiliar para falar sem travar os subprocessos do RealtimeSTT
def falar_texto_assincrono(texto):
    def worker(t):
        try:
            # Inicializa e encerra o motor dentro da própria thread dedicada
            fala_engine = pyttsx3.init()
            fala_engine.say(t)
            fala_engine.runAndWait()
            del fala_engine # Libera explicitamente o canal SAPI5 do Windows
        except Exception as e:
            print(f"Erro na thread de voz: {e}")

    # Cria e inicia a thread para liberar o loop do chatbot imediatamente
    thread_fala = threading.Thread(target=worker, args=(texto,), daemon=True)
    thread_fala.start()


# ============================================================
# MAIN PROGRAM ENTRANCE (SAFE FOR WINDOWS SPAWN)
# ============================================================

if __name__ == '__main__':
    print("Starting the chat with Groq API...")
    print("Carregando modelo de voz ultra rápido...")

    # Inicializa o gravador com segurança dentro do bloco main
    recorder = AudioToTextRecorder(
        model="tiny",              
        language="pt",             
        input_device_index=None,   
        spinner=False              
    )

    print("Sistema Kronos AI pronto!")
    falar_texto_assincrono("Sistema pronto.")

    while True:
        user_message = ouvir_microfone_ultra_rapido(recorder)

        if user_message:
            print(f"Você disse: {user_message}")

            if user_message.lower() == "exit" or "sair" in user_message.lower():
                print("Goodbye!")
                falar_texto_assincrono("Até logo!")
                break

            messages.append({"role": "user", "content": user_message})

            try:
                # Send the request to Groq
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=4096
                )

                response_message = response.choices[0].message
                messages.append(response_message)

                # 1. If Groq decides to call a tool (Function Calling)
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        function_to_call = available_functions[function_name]
                        function_response = function_to_call(keyword=function_args.get("keyword"))

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        })

                    second_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages
                    )
                    final_text = second_response.choices[0].message.content
                    
                    if final_text:
                        print(f"[AI]: {final_text}")
                        falar_texto_assincrono(final_text) # Chamada assíncrona corrigida

                # 2. If it's just a text response
                elif response_message.content:
                    print(f"[AI]: {response_message.content}")
                    falar_texto_assincrono(response_message.content) # Chamada assíncrona corrigida

            except Exception as error:
                print("\nError:")
                print(error)
                falar_texto_assincrono("Ocorreu um erro no processamento.")