# import whisper
# from jiwer import wer

# # Load Whisper model
# model = whisper.load_model("base")

# # Paths to your accent audio files
# accent_files = {
#     "American": "data/american.wav",
#     "British": "data/british.wav",
#     "Indian": "data/indian.wav",
# }

# # Ground truth transcripts (for WER comparison)
# ground_truths = {
#     "American": "Hello, my name is Bes.",
#     "British": "Hello, my name is Bes.",
#     "Indian": "Hello, my name is Bes.",
# }

# # Output file
# with open("results/transcripts.txt", "w") as f_out:
#     for accent, path in accent_files.items():
#         print(f"\n--- {accent} Accent ---")
#         result = model.transcribe(path)
#         transcript = result["text"]
#         reference = ground_truths[accent]

#         print("Transcript:", transcript)
#         print("Reference: ", reference)
#         print("WER:       ", wer(reference, transcript))

#         f_out.write(f"{accent} Accent:\n")
#         f_out.write(f"Transcript: {transcript}\n")
#         f_out.write(f"Reference:  {reference}\n")
#         f_out.write(f"WER:        {wer(reference, transcript)}\n\n")
