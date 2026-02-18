import whisper

model = whisper.load_model("base")
result = model.transcribe("talk.mp3",  fp16=False)
print(result["text"])