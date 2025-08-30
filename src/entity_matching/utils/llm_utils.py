from openai import OpenAI, AsyncOpenAI
import openai
import logging
import os
import json
import asyncio

async def _ask(client, messages, model, temperature, max_tokens, response_format=None):
    response = None
    retry_delay = 5  # seconds
    response = None

    while response is None:
        try:
            response = await client(
                model=model,
                messages=messages,
                temperature=temperature,
                # max_tokens=max_tokens,
                response_format=response_format
            )
            break

        except openai.RateLimitError:
            print("Rate limit exceeded. Retrying after a short delay...")
            await asyncio.sleep(retry_delay)

        except openai.BadRequestError as e:
            # if '"Unsupported value: \'temperature\'' in str(e):
            #     print("Temperature parameter is not supported.")

            try:
                response = await client(
                    model=model,
                    messages=messages,
                    response_format=response_format
                )
                break
            except Exception as e:
                print(f'{e=}')
                print(messages)
                await asyncio.sleep(retry_delay)

        except Exception as e:
            print(f'{e=}')
            print(messages)
            await asyncio.sleep(retry_delay)

        finally:
            await asyncio.sleep(1)  # brief pause to avoid hitting rate limits
    
    return response.choices[0].message

async def openai_chat_completion(model, system_prompt, history=[], temperature=0, max_tokens=512, return_type=None):
    base_url = None
    api_key = None

    with open('.env', 'r') as f: 
        local_env_setting = json.load(f)

    if model.startswith('gpt'):
        base_url = None
        api_key = os.environ.get('OPENAI_API_KEY', None) or local_env_setting.get('OPENAI_API_KEY', None)
    elif model.startswith('deepseek'):
        base_url = 'https://api.deepseek.com'
        api_key = os.environ.get('DEEPSEEK_API_KEY', None) or local_env_setting.get('DEEPSEEK_API_KEY', None)
    else:
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = os.environ.get('DASHSCOPE_API_KEY', None) or local_env_setting.get('DASHSCOPE_API_KEY', None)

    if api_key is None:
        print("""API is missing. You might add your api key in .env file. For example,
              {
                "OPENAI_API_KEY": "...",
                "DEEPSEEK_API_KEY": "...",
                "DASHSCOPE_API_KEY": "..."
              }
              """)
        
        exit()
    
    aclient = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60,
    )

    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        if return_type:
            response = await _ask(aclient.chat.completions.parse, messages, model, temperature, max_tokens, return_type)
            logging.debug(f"Model: {model}\nPrompt:\n {messages}\n openaiResult: {response}")
            if response.refusal:
                return None
            
            return response.parsed
        else:
            response = await _ask(aclient.chat.completions.create, messages, model, temperature, max_tokens)
            response = response.content
            logging.debug(f"Model: {model}\nPrompt:\n {messages}\n openaiResult: {response}")
            
            return str(response)
    

def fast_openai_chat_completion(model, system_prompt, history=[], temperature=0, max_tokens=None):
    base_url = None
    api_key = None

    with open('.env', 'r') as f: 
        local_env_setting = json.load(f)

    if model.startswith('gpt'):
        base_url = None
        api_key = os.environ.get('OPENAI_API_KEY', None) or local_env_setting.get('OPENAI_API_KEY', None)
    elif model.startswith('deepseek'):
        base_url = 'https://api.deepseek.com'
        api_key = os.environ.get('DEEPSEEK_API_KEY', None) or local_env_setting.get('DEEPSEEK_API_KEY', None)
    else:
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = os.environ.get('DASHSCOPE_API_KEY', None) or local_env_setting.get('DASHSCOPE_API_KEY', None)

    if api_key is None:
        print("""API is missing. You might add your api key in .env file. For example,
              {
                "OPENAI_API_KEY": "...",
                "DEEPSEEK_API_KEY": "...",
                "DASHSCOPE_API_KEY": "..."
              }
              """)
        
        exit()
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60,
    )

    response = None
    messages = [{
        'role': 'system',
        'content': system_prompt
    }] + history

    while response is None:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break

        except openai.RateLimitError:
            print("Rate limit exceeded. Retrying after a short delay...")

        except openai.BadRequestError as e:
            # if '"Unsupported value: \'temperature\'' in str(e):
            #     print("Temperature parameter is not supported.")

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                break
            except Exception as e:
                print(f'{e=}')
                print(messages)

        except Exception as e:
            print(f'{e=}')
            print(messages)

    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n openaiResult: {response.choices[0].message}")
    
