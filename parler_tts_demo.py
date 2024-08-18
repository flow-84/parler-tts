import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Gerät auswählen (GPU oder CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Modell und Tokenizer laden
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Eingabeprompt und Beschreibung
prompt = "Hey, how are you doing today?"
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

# Tokenize Eingabe und Beschreibung
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Audio generieren
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

# Audio in eine Datei schreiben
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

print("Audio erfolgreich generiert: parler_tts_out.wav")
