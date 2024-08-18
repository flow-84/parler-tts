import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from tqdm import tqdm

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

# Fortschrittsanzeige und Generierung des Audios
with tqdm(total=1, desc="Generating Audio") as pbar:
    outputs = model.generate(input_ids, prompt_input_ids, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
    pbar.update(1)

# Audio speichern
audio = outputs[0].cpu().numpy()
sf.write("output.wav", audio, 22050)

print("Audio erfolgreich generiert: output.wav")
