import torch
import gradio as gr
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
import traceback
import numpy as np
from scipy.signal import stft
from sklearn.calibration import calibration_curve

# Load the pre-trained model and processor for audio classification
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-960h-lv60")

def get_manipulation_nature(waveform, sample_rate):
    manipulations = []
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Compute spectrogram
    _, _, Zxx = stft(waveform.numpy().squeeze(), fs=sample_rate, nperseg=1024)
    spectrogram = np.abs(Zxx)
    
    # Check for pitch manipulation
    pitch_std = np.std(np.diff(np.argmax(spectrogram, axis=0)))
    if pitch_std > 10:  # Threshold determined empirically
        manipulations.append("Pitch")
    
    # Check for speed manipulation
    duration = waveform.shape[1] / sample_rate
    if duration < 0.5 or duration > 10:  # Assuming normal speech duration
        manipulations.append("Speed")
    
    # Check for background noise
    noise_level = np.mean(spectrogram[spectrogram < np.percentile(spectrogram, 10)])
    if noise_level > 0.1:  # Threshold determined empirically
        manipulations.append("Background Noise")
    
    # Check for echo
    auto_corr = np.correlate(waveform.numpy().squeeze(), waveform.numpy().squeeze(), mode='full')
    peaks = np.argpartition(auto_corr, -5)[-5:]  # Get indices of top 5 peaks
    if np.any(peaks > len(auto_corr) // 2):
        manipulations.append("Echo")
    
    # Check for tone manipulation
    tone_variation = np.std(np.sum(spectrogram, axis=0))
    if tone_variation > 100:  # Threshold determined empirically
        manipulations.append("Tone")
    
    # Check for volume manipulation
    if np.max(np.abs(waveform.numpy())) > 0.9:  # Close to clipping
        manipulations.append("Volume")
    
    if not manipulations:
        return "No manipulation detected"
    else:
        return f"Potential manipulations detected: {', '.join(manipulations)}"

def calculate_confidence(logits):
    probs = F.softmax(logits, dim=-1)
    max_prob = torch.max(probs).item()
    
    # Calculate entropy of the probability distribution
    entropy = -torch.sum(probs * torch.log(probs))
    max_entropy = -torch.log(torch.tensor(1.0 / probs.shape[-1]))
    normalized_entropy = entropy / max_entropy
    
    # Calculate the margin between the top two probabilities
    top2_probs, _ = torch.topk(probs, 2)
    margin = (top2_probs[0] - top2_probs[1]).item()
    
    # Combine max probability, normalized entropy, and margin for a more nuanced confidence score
    confidence = (max_prob * (1 - normalized_entropy) * (1 + margin)) / 2
    
    return confidence * 100  # Convert to percentage

def calibrate_confidence(confidence, calibration_data):
    # This function would be used to calibrate the confidence scores
    # based on historical data. For demonstration, we'll use a simple scaling.
    return min(confidence * 1.1, 100)  # Scale up slightly, but cap at 100%

def process_audio(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform).squeeze()
        inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        with tqdm(total=1, desc="Processing audio", unit="file") as pbar:
            with torch.no_grad():
                outputs = model(**inputs)
            pbar.update(1)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        confidence = calculate_confidence(logits)
        
        # In a real-world scenario, you would have a separate calibration dataset
        # Here, we're using a placeholder for demonstration
        calibration_data = None
        calibrated_confidence = calibrate_confidence(confidence, calibration_data)
        
        manipulation_nature = get_manipulation_nature(waveform, sample_rate)
        result = "AI-GENERATED" if prediction == 1 else "Real"
        return result, calibrated_confidence, manipulation_nature
    except Exception as e:
        error_message = f"Error processing audio: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return "Error", None, error_message

def classify_audio(audio_file):
    try:
        if audio_file is None:
            return "No file uploaded", "N/A", "Please upload an audio file to analyze"

        if isinstance(audio_file, tuple) and audio_file[1] == 'microphone':
            waveform, sample_rate = torchaudio.load(audio_file[0])
            manipulation_nature = get_manipulation_nature(waveform, sample_rate)
            return "Real", "N/A", manipulation_nature

        audio_path = audio_file.name if isinstance(audio_file, gr.File) else audio_file
        result, confidence, manipulation = process_audio(audio_path)
        return result, f"{confidence:.2f}%" if confidence else "N/A", manipulation
    except Exception as e:
        error_message = f"Error in classify_audio: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return "Error", "N/A", error_message

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI-Generated Voice Detection")
    gr.Markdown("Upload an audio file or use the microphone to check if the voice is AI-generated or real using AI.")

    with gr.Row():
        input_audio = gr.Audio(type="filepath", label="Upload Audio or Record")
        analyze_btn = gr.Button("Analyze")

    with gr.Row():
        result = gr.Textbox(label="Classification Result")
        confidence = gr.Textbox(label="Confidence Score")
        manipulation = gr.Textbox(label="Nature of Manipulation")

    analyze_btn.click(
        fn=classify_audio,
        inputs=[input_audio],
        outputs=[result, confidence, manipulation]
    )
    
    
if __name__ == "__main__":
    demo.launch(share=True)
