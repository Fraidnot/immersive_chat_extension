import time
import io
import base64
import binascii
import json
from pathlib import Path

import gradio as gr
import torch
import requests

from extensions.silero_tts import tts_preprocessor
from modules import chat, shared
from modules.html_generator import chat_html_wrapper
from PIL import Image


torch._C._jit_set_profiling_mode(False)


params = {
    'activate': True,
    'speaker': 'en_56',
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'show_text': True,
    'autoplay': True,
    'voice_pitch': 'medium',
    'voice_speed': 'medium',
    'local_cache_path': '',  # User can override the default cache path to something other via settings.json
    'address': 'http://127.0.0.1:7860',
    'mode': 0,  # modes of operation: 0 (Manual only), 1 (Immersive/Interactive - looks for words to trigger), 2 (Picturebook Adventure - Always on)
    'manage_VRAM': False,
    'save_img': False,
    'SD_model': 'NeverEndingDream',  # not used right now
    'prompt_prefix': '(Masterpiece:1.1), detailed, intricate, colorful',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'create_image': 'Describe what would be on a picture taken at the current moment of this story.',
    'picture_response': True,
    'width': 512,
    'height': 512,
    'denoising_strength': 0.61,
    'restore_faces': False,
    'enable_hr': False,
    'hr_upscaler': 'ESRGAN_4x',
    'hr_scale': '1.0',
    'seed': -1,
    'sampler_name': 'DDIM',
    'steps': 32,
    'cfg_scale': 7
}
current_params = params.copy()
voices_by_gender = ['en_99', 'en_45', 'en_18', 'en_117', 'en_49', 'en_51', 'en_68', 'en_0', 'en_26', 'en_56', 'en_74', 'en_5', 'en_38', 'en_53', 'en_21', 'en_37', 'en_107', 'en_10', 'en_82', 'en_16', 'en_41', 'en_12', 'en_67', 'en_61', 'en_14', 'en_11', 'en_39', 'en_52', 'en_24', 'en_97', 'en_28', 'en_72', 'en_94', 'en_36', 'en_4', 'en_43', 'en_88', 'en_25', 'en_65', 'en_6', 'en_44', 'en_75', 'en_91', 'en_60', 'en_109', 'en_85', 'en_101', 'en_108', 'en_50', 'en_96', 'en_64', 'en_92', 'en_76', 'en_33', 'en_116', 'en_48', 'en_98', 'en_86', 'en_62', 'en_54', 'en_95', 'en_55', 'en_111', 'en_3', 'en_83', 'en_8', 'en_47', 'en_59', 'en_1', 'en_2', 'en_7', 'en_9', 'en_13', 'en_15', 'en_17', 'en_19', 'en_20', 'en_22', 'en_23', 'en_27', 'en_29', 'en_30', 'en_31', 'en_32', 'en_34', 'en_35', 'en_40', 'en_42', 'en_46', 'en_57', 'en_58', 'en_63', 'en_66', 'en_69', 'en_70', 'en_71', 'en_73', 'en_77', 'en_78', 'en_79', 'en_80', 'en_81', 'en_84', 'en_87', 'en_89', 'en_90', 'en_93', 'en_100', 'en_102', 'en_103', 'en_104', 'en_105', 'en_106', 'en_110', 'en_112', 'en_113', 'en_114', 'en_115']
voice_pitches = ['x-low', 'low', 'medium', 'high', 'x-high']
voice_speeds = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']
streaming_state = shared.args.no_stream  # remember if chat streaming was enabled

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})
def give_VRAM_priority(actor):
    global shared, params

    if actor == 'SD':
        unload_model()
        print("Requesting Auto1111 to re-load last checkpoint used...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()

    elif actor == 'LLM':
        print("Requesting Auto1111 to vacate VRAM...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()
        reload_model()

    elif actor == 'set':
        print("VRAM mangement activated -- requesting Auto1111 to vacate VRAM...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()

    elif actor == 'reset':
        print("VRAM mangement deactivated -- requesting Auto1111 to reload checkpoint")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()

    else:
        raise RuntimeError(f'Managing VRAM: "{actor}" is not a known state!')

    response.raise_for_status()
    del response


if params['manage_VRAM']:
    give_VRAM_priority('set')

samplers = ['DDIM', 'DPM++ 2M Karras']  # TODO: get the availible samplers with http://{address}}/sdapi/v1/samplers
SD_models = ['NeverEndingDream']  # TODO: get with http://{address}}/sdapi/v1/sd-models and allow user to select

streaming_state = shared.args.no_stream  # remember if chat streaming was enabled
picture_response = True  # specifies if the next model response should appear as a picture


def xmlesc(txt):
    return txt.translate(table)

def load_model():
    torch_cache_path = torch.hub.get_dir() if params['local_cache_path'] == '' else params['local_cache_path']
    model_path = torch_cache_path + "/snakers4_silero-models_master/src/silero/model/" + params['model_id'] + ".pt"
    if Path(model_path).is_file():
        print(f'\nUsing Silero TTS cached checkpoint found at {torch_cache_path}')
        model, example_text = torch.hub.load(repo_or_dir=torch_cache_path + '/snakers4_silero-models_master/', model='silero_tts', language=params['language'], speaker=params['model_id'], source='local', path=model_path, force_reload=True)
    else:
        print(f'\nSilero TTS cache not found at {torch_cache_path}. Attempting to download...')
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=params['language'], speaker=params['model_id'])
    model.to(params['device'])
    return model


def remove_tts_from_history(name1, name2, mode):
    for i, entry in enumerate(shared.history['internal']):
        shared.history['visible'][i] = [shared.history['visible'][i][0], entry[1]]
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def toggle_text_in_history(name1, name2, mode):
    for i, entry in enumerate(shared.history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = shared.history['internal'][i][1]
                shared.history['visible'][i] = [shared.history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                shared.history['visible'][i] = [shared.history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)

input_string =""

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    # Remove autoplay from the last reply
    if shared.is_chat() and len(shared.history['internal']) > 0:
        shared.history['visible'][-1] = [shared.history['visible'][-1][0], shared.history['visible'][-1][1].replace('controls autoplay>', 'controls>')]

    shared.processing_message = "*Is recording a voice message...*"
    shared.args.no_stream = True  # Disable streaming cause otherwise the audio output will stutter and begin anew every time the message is being updated
    input_string = string

    return string


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global model, current_params, streaming_state, picture_response, params

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break


    original_string = string

    if params['activate']:
        tts_processed_string = tts_preprocessor.preprocess(string)

        if tts_processed_string != '':
            output_file = Path(f'extensions/silero_tts/outputs/{shared.character}_{int(time.time())}.wav')
            prosody = '<prosody rate="{}" pitch="{}">'.format(params['voice_speed'], params['voice_pitch'])
            silero_input = f'<speak>{prosody}{xmlesc(string)}</prosody></speak>'
            model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))

            autoplay = 'autoplay' if params['autoplay'] else ''
            string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
            #if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    #shared.args.no_stream = streaming_state  # restore the streaming option to the previous value
    
    if not params['picture_response']:
        return string

    #renderstring = generate_impage_prompt(original_string, "Create an image prompt depicting the current moment of the story.")
    renderstring = generate_impage_prompt(original_string, params['create_image'])
    image_prompt_string = get_Image_Prompt(renderstring)
    if image_prompt_string and image_prompt_string != "" and image_prompt_string != " ":
        string = get_SD_pictures(image_prompt_string) + "\n" + string

    return string

def generate_impage_prompt(response_string, image_creation_command):
    rows = []
    min_rows = 3

    # Finding the maximum prompt size
    max_length = shared.settings['truncation_length_max'] - shared.settings['max_new_tokens']


    # Building the prompt start with context

    i = len(shared.history['internal']) - 1
    while i >= 0 and len('\n'.join(rows)) < max_length:
        rows.insert(1, shared.history['internal'][i][0].strip())
        rows.insert(1, shared.history['internal'][i][1].strip())
        i -= 1

    
    # Adding the user message
    if len(input_string) > 0:
        rows.append(response_string)
    if len(response_string) > 0:
        rows.append(response_string)

    # Adding the Character prefix
    rows.append("\n" + image_creation_command)

    while len(rows) > min_rows and (len('\n'.join(rows)) + len(shared.settings['context'].strip())) >= max_length:
        rows.pop(1)

    rows.insert(0, shared.settings['context'].strip())

    prompt = '\n'.join(rows)
    return prompt

# Get a prompt for image generation
def get_Image_Prompt(image_promt):
    HOST = 'localhost:5000'
    URI = f'http://{HOST}/api/v1/generate'
    print("Generateing image_promt using " + URI)
    request = {
        'prompt': json.dumps(image_promt),
        'max_new_tokens': 250,
        'do_sample': True,
        'temperature': 1.3,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
        'singleline': True,
        'use_story': False,
        'use_memory': False,
        'use_authors_note': False,
        'use_world_info': False,
    }

    response = requests.post(URI, json=request, headers={'Content-type': 'application/json'}, timeout=(5*60*1000))
    response.raise_for_status()

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        print("Image prompt text: "+ result)
        return result

# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description):

    global params

    if params['manage_VRAM']:
        give_VRAM_priority('SD')

    payload = {
        "prompt": params['prompt_prefix'] + description,
        "seed": params['seed'],
        "sampler_name": params['sampler_name'],
        "enable_hr": params['enable_hr'],
        "hr_scale": params['hr_scale'],
        "hr_upscaler": params['hr_upscaler'],
        "denoising_strength": params['denoising_strength'],
        "steps": params['steps'],
        "cfg_scale": params['cfg_scale'],
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "override_settings_restore_afterwards": True,
        "negative_prompt": params['negative_prompt']
    }

    print(f'Prompting the image generator via the API on {params["address"]}...')
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
    r = response.json()

    visible_result = ""
    for img_str in r['images']:
        if params['save_img']:
            img_data = base64.b64decode(img_str)

            variadic = f'{date.today().strftime("%Y_%m_%d")}/{shared.character}_{int(time.time())}'
            output_file = Path(f'extensions/sd_api_pictures/outputs/{variadic}.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file.as_posix(), 'wb') as f:
                f.write(img_data)

            visible_result = visible_result + f'<img src="/file/extensions/sd_api_pictures/outputs/{variadic}.png" alt="that" style="max-width: unset; max-height: unset;">\n'
        else:
            image = Image.open(io.BytesIO(base64.b64decode(img_str.split(",", 1)[0])))
            # lower the resolution of received images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
            image.thumbnail((300, 300))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            buffered.seek(0)
            image_bytes = buffered.getvalue()
            img_str = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
            visible_result = visible_result + f'<img src="{img_str}" alt="that">\n'

    if params['manage_VRAM']:
        give_VRAM_priority('LLM')

    return visible_result

def SD_api_address_update(address):

    global params

    msg = "✔️ SD API is found on:"
    address = filter_address(address)
    params.update({"address": address})
    try:
        response = requests.get(url=f'{params["address"]}/sdapi/v1/sd-models')
        response.raise_for_status()
        # r = response.json()
    except:
        msg = "❌ No SD API endpoint on:"

    return gr.Textbox.update(label=msg)

def toggle_generation(*args):
    global picture_response, shared, streaming_state

    if not args:
        picture_response = not picture_response
    else:
        picture_response = args[0]

    shared.args.no_stream = True if picture_response else streaming_state  # Disable streaming cause otherwise the SD-generated picture would return as a dud
    shared.processing_message = "*Is sending a picture...*" if picture_response else "*Is typing...*"

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string

def setup():
    global model
    model = load_model()


def ui():
    # Gradio elements
    with gr.Accordion("Immersive"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        voice = gr.Dropdown(value=params['speaker'], choices=voices_by_gender, label='TTS voice')
        with gr.Row():
            v_pitch = gr.Dropdown(value=params['voice_pitch'], choices=voice_pitches, label='Voice pitch')
            v_speed = gr.Dropdown(value=params['voice_speed'], choices=voice_speeds, label='Voice speed')
        with gr.Row():
            address = gr.Textbox(placeholder=params['address'], value=params['address'], label='Auto1111\'s WebUI address')
            picture_response = gr.Checkbox(value=params['picture_response'], label="Generate Image")
            with gr.Column(scale=1, min_width=300):
                manage_VRAM = gr.Checkbox(value=params['manage_VRAM'], label='Manage VRAM')
                save_img = gr.Checkbox(value=params['save_img'], label='Keep original images and use them in chat')

        with gr.Accordion("Generation parameters", open=False):
            prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
            negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
            create_image_prompt = gr.Textbox(placeholder=params['create_image'], value=params['create_image'], label='Create Image Prompt')
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(256, 768, value=params['width'], step=64, label='Width')
                    height = gr.Slider(256, 768, value=params['height'], step=64, label='Height')
                with gr.Column():
                    sampler_name = gr.Textbox(placeholder=params['sampler_name'], value=params['sampler_name'], label='Sampling method', elem_id="sampler_box")
                    steps = gr.Slider(1, 150, value=params['steps'], step=1, label="Sampling steps")
            with gr.Row():
                seed = gr.Number(label="Seed", value=params['seed'], elem_id="seed_box")
                cfg_scale = gr.Number(label="CFG Scale", value=params['cfg_scale'], elem_id="cfg_box")
                with gr.Column() as hr_options:
                    restore_faces = gr.Checkbox(value=params['restore_faces'], label='Restore faces')
                    enable_hr = gr.Checkbox(value=params['enable_hr'], label='Hires. fix')                    
            with gr.Row(visible=params['enable_hr'], elem_classes="hires_opts") as hr_options:
                    hr_scale = gr.Slider(1, 4, value=params['hr_scale'], step=0.1, label='Upscale by')
                    denoising_strength = gr.Slider(0, 1, value=params['denoising_strength'], step=0.01, label='Denoising strength')
                    hr_upscaler = gr.Textbox(placeholder=params['hr_upscaler'], value=params['hr_upscaler'], label='Upscaler')                    

    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None)
    show_text.change(toggle_text_in_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']], shared.gradio['display'])
    show_text.change(lambda: chat.save_history('instruct',timestamp=False), [], [], show_progress=False)

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    voice.change(lambda x: params.update({"speaker": x}), voice, None)
    v_pitch.change(lambda x: params.update({"voice_pitch": x}), v_pitch, None)
    v_speed.change(lambda x: params.update({"voice_speed": x}), v_speed, None)
    # Gradio elements
    # gr.Markdown('### Stable Diffusion API Pictures') # Currently the name of extension is shown as the title
    # Event functions to update the parameters in the backend
    address.change(lambda x: params.update({"address": filter_address(x)}), address, None)
    picture_response.change(lambda x: params.update({"picture_response": x}), picture_response, None)
    manage_VRAM.change(lambda x: params.update({"manage_VRAM": x}), manage_VRAM, None)
    manage_VRAM.change(lambda x: give_VRAM_priority('set' if x else 'reset'), inputs=manage_VRAM, outputs=None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)

    address.submit(fn=SD_api_address_update, inputs=address, outputs=address)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    create_image_prompt.change(lambda x: params.update({"create_image": x}), create_image_prompt, None)
    width.change(lambda x: params.update({"width": x}), width, None)
    height.change(lambda x: params.update({"height": x}), height, None)
    hr_scale.change(lambda x: params.update({"hr_scale": x}), hr_scale, None)
    denoising_strength.change(lambda x: params.update({"denoising_strength": x}), denoising_strength, None)
    restore_faces.change(lambda x: params.update({"restore_faces": x}), restore_faces, None)
    hr_upscaler.change(lambda x: params.update({"hr_upscaler": x}), hr_upscaler, None)
    enable_hr.change(lambda x: params.update({"enable_hr": x}), enable_hr, None)
    enable_hr.change(lambda x: hr_options.update(visible=params["enable_hr"]), enable_hr, hr_options)

    sampler_name.change(lambda x: params.update({"sampler_name": x}), sampler_name, None)
    steps.change(lambda x: params.update({"steps": x}), steps, None)
    seed.change(lambda x: params.update({"seed": x}), seed, None)
    cfg_scale.change(lambda x: params.update({"cfg_scale": x}), cfg_scale, None)