from datasets import load_dataset
import whisper
import soundfile as sf
import os
from jiwer import wer
import csv
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
 
# ✅ Load Whisper model
model = whisper.load_model("base")
print("✅ Whisper is working!")

# ✅ Load a small portion of the dataset
dataset = load_dataset("DTU54DL/common-accent", split="train[:10%]")

# ✅ Create temp directory to save audio
os.makedirs("temp_audio", exist_ok=True)

# ✅ Prepare to store results
results = []

# ✅ Process each sample
for i, sample in enumerate(dataset):
    accent = sample['accent']
    sentence = sample['sentence']
    audio_array = sample['audio']['array']
    sr = sample['audio']['sampling_rate']

    # Save to temp WAV file
    temp_path = f"temp_audio/sample_{i}.wav"
    sf.write(temp_path, audio_array, sr)

    # Transcribe
    result = model.transcribe(temp_path)
    predicted = result['text']

    # Calculate WER
    error = wer(sentence, predicted)

    # Print results
    print(f"\n--- Sample {i+1} ---")
    print("Accent:      ", accent)
    print("Actual:      ", sentence)
    print("Transcribed: ", predicted)
    print("WER:         ", round(error, 2))

    # Save result
    results.append({
        "Sample": i + 1,
        "Accent": accent,
        "Reference": sentence,
        "Prediction": predicted,
        "WER": round(error, 2)
    })

# ✅ Save to CSV
os.makedirs("results", exist_ok=True)
with open("results/whisper_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ✅ Compute and print average WER
average_wer = sum(r["WER"] for r in results) / len(results)
print(f"\n🔍 Average WER across all samples: {round(average_wer, 2)}")

# ✅ Create visualization
df = pd.DataFrame(results)
plt.figure(figsize=(10, 5))
plt.bar(df["Sample"], df["WER"])
plt.xlabel("Sample")
plt.ylabel("Word Error Rate (WER)")
plt.title("WER per Sample (Whisper on Accented Speech)")
plt.tight_layout()
plt.savefig("results/wer_plot.png")
plt.show()
print("✅ Results saved to results/whisper_results.csv")
print("✅ Plot saved to results/wer_plot.png")
