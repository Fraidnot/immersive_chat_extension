# immersive_chat_extension
Extension for oobabooga/text-generation-webui that lets you use text to speech and automatically generates images with Stable diffusion.

Requirements: You need to have oobabooga/text-generation-webui and stable diffusion for image generation

Installation: Drop the folder immersive_talk in the text-generation-webui\extensions folder

Important: text-generation-webui needs to be run with the api mode and the extension needs to be turned on to do this go to the 'interface mode' tab in the webui or modify webui.py to run the server with the parameter '--api' something like this: 
    run_cmd("python server.py --api --extension immersive_talk")

This is work that combines work from the two extensions sd_api_pictures and silero_tts all credit belongs to those who I stole code from, but the real kicker about this is that uses the local api to have AI to generate an image prompt. By default this is done using the command:
  "Describe what would be on a picture taken at the current moment of this story." 
This command can be chaned in the ui,
  "Create an image prompt depicting the current scene" sometimes works well
By defualt it uses the api at 127.0.0.1:5000 to generate images 

ALSO IMPORTANT: only vicuna-13b was tested to work with this, other models apparently aren't smart enough to understand the prompt and normally do not generate an image prompt (it's not clear as to why this is the case).

One problem is this is not fast... but it works... sometimes... still a work in progress
