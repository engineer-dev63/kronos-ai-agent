import json
import webbrowser
from urllib.parse import quote
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os
import pyttsx3
import speech_recognition as sr


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
    "You are 'kronos' an automation assistant created by 'The Master Ailan'. You have access to tools to search Google or open websites. "
    "Always use 'openlink' when the user explicitly asks to open a site, and 'onlysearch' when they want to search for something."
    "You should provide short responses and avoid unnecessary explanations, no long searches or explanations unless the user ask. If you don't know the answer, use 'onlysearch' to find it. "
    'You should speak portuguese br, unless the user want other language.'
    "There is a secret password, it is '0006' 'The Master Ailan', your creator. If the user says this password, you should respond with 'Hello, Master Ailan! How can I assist you today?'"
)

# Conversation message history
messages = [
    {"role": "system", "content": system_instruction}
]

# Groq model with tool support
MODEL_NAME = "openai/gpt-oss-120b"

# ============================================================
# MAIN LOOP
# ============================================================

print("Starting the chat with Groq API...")

# start the text-to-speech engine
engine = pyttsx3.init()






# Function to listen to the microphone and convert speech to text
def ouvir_microfone():
    reconhecedor = sr.Recognizer()
    with sr.Microphone() as fonte:
        # Ajusta o ruído de fundo automaticamente por 1 segundo
        reconhecedor.adjust_for_ambient_noise(fonte, duration=1)
        print("Pode falar... Estou ouvindo:")
        
        try:
            # Captura o áudio direto na memória RAM (bloqueia até terminar de falar)
            audio_em_memoria = reconhecedor.listen(fonte, timeout=5, phrase_time_limit=10)
            print("Processando fala...")
            
            # Converte os bytes de áudio em texto usando a API do Google (Gratuito)
            texto = reconhecedor.recognize_whisper(audio_em_memoria, language='portuguese')
            print(texto)
            
        except sr.UnknownValueError:
            print("Não consegui entender o que você disse.")
        except sr.RequestError:
            print("Erro de conexão com o serviço de transcrição.")
        except sr.WaitTimeoutError:
            print("Tempo de espera esgotado. Ninguém falou nada.")







while True:

    user_message = ouvir_microfone()

    # print(f"\n[YOU]: {user_message}")


    # Add the user message to the history
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

                # print(f"\n[TOOL CALL]: {function_name}")
                # print(f"[ARGUMENTS]: {function_args}")

                # Execute the function
                function_to_call = available_functions[function_name]
                function_response = function_to_call(keyword=function_args.get("keyword"))

                # Send the function result back to the model
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })

            # Generate the final response with the tool's result
            second_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            final_text = second_response.choices[0].message.content
            
            if final_text:
                print(f"[AI]: {final_text}")
                engine.say(final_text)
                engine.runAndWait()


        # 2. If it's just a text response
        elif response_message.content:
            print(f"[AI]: {response_message.content}")
            engine.say(response_message.content)
            engine.runAndWait()


    except Exception as error:
        print("\nError:")
        print(error)
