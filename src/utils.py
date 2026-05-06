import torch
import torch.nn.functional as F
from typing import Union, Optional
import librosa
import numpy as np
import matplotlib.pyplot as plt

def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400 # fast fourier transform window size is 25ms, so there are 400 samples in 25ms
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_MELS = 80
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def format_data(sample):
    """
    Convert a training sample into OpenAI conversation format.
    Each sample contains audio data and its corresponding transcription.
    """
    # Extract the transcription text
    transcription = sample['text']
    
    # Create the conversation format
    conversation = [
        {
            "content":[{
                            "text": "You are an ASR model that transcribes speech to text. Avoid additional explanation unless absolutely necessary.",
                            "type": "text"
                        }],
            "role":"system"
        },
        {
            "content": [{
                    "audio": sample['wav']['bytes'],  # Raw audio bytes
                    "type": "audio"
                },
                {
                    "text": "Transcribe this speech into text.",
                    "type": "text",
                }
            ],
            "role": "user"
        },
        {
            "content": [
                {
                    "text": transcription,
                    "type": "text"
                }
            ],
            "role": "assistant",
        }
    ]
    
    return conversation




def pad_and_split(
    sr: int,
    audio_np: np.ndarray,
    device: Optional[Union[str, torch.device]] = 'cpu',
    chunk_length_s: int = 30,
) -> torch.Tensor:
    """
    Split 1D audio array into fixed-length chunks of target_length where target_length = sr * chunk_length_s.
    Right-pad the last chunk with zeros. Returns (B, T) float32 on device.
    """

    # samples within chunk (30 sec * 16kHz = 480000 samples)
    target_length = int(sr * chunk_length_s)
    # convert to tensor and send to device
    x = torch.from_numpy(audio_np).to(device=device, dtype=torch.float32)
    # calculate the chunk count
    # do ceiling division to include partial chunk
    # 5.1M samples with target_length = 500k -> 11 chunks
    chunks = (x.shape[0] + target_length - 1) // target_length
    padding = chunks * target_length - x.shape[0]
    if padding:
        x = F.pad(x,(0,padding))

    return x.view(chunks, target_length)


def convert_np_to_log_mel_spectrogram(sr: int,
                                      n_fft: int,
                                      hop_length: int,
                                      n_mels: int,
                                      n_frames: int,
                                      audio_np: np.ndarray,
                                      device: Optional[Union[str, torch.device]]):
    """
    Convert a 1D audio numpy array to a log mel spectrogram.

    Steps:
    1. Check the audio size
        1.1. If < 30 seconds, pad with zeros
        1.2. If > 30 seconds, truncate to 30 seconds and create another sample with the remaining audio
    2. Compute the mel spectrogram using librosa (FFT)
    3. Convert to log scale (dB)
    4. Normalize to [-1, 1]
    5. Ensure the spectrogram has the correct shape (80,3000)
    """

    # Step 1: Check audio size, create new samples if needed
    audios = pad_and_split(sr, audio_np, device).to(device) # (B,T)

    # Step 2: Compute the log mel for each chunk

    # window is to taper off the edges of each chunk to have smoother transitions across chunks
    # torch.hann_window(8) returns tensor([0.0000, 0.1464, 0.5000, 0.8536, 1.0000, 0.8536, 0.5000, 0.1464])
    # like a bell curve, high in middle, low at the edges
    window = torch.hann_window(n_fft).to(device)
    # window.shape torch.Size([400])

    # transform with Fourier (B,T) -> (B,F+1,T+1)

    stft = torch.stft(audios, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    # stft.shape torch.Size([2, 201, 3001])

    # batch size of 2

    # 201 freq bins → number of unique frequency bins along the y-axis
    # Computed as N_FFT // 2 + 1
    # Why +1? Because:
    # - An N_FFT-point FFT produces N_FFT bins (0 … N_FFT-1), covering both + and – frequencies
    # - For real audio, the spectrum is Hermitian: negative frequencies are redundant
    # - So we keep only the non-redundant half: from 0 Hz (DC) up to Nyquist (sr/2)
    #   That gives N_FFT//2 + 1 bins
    # Example: N_FFT = 400 → 400//2 + 1 = 201 bins

    # 3001 time frames -> count of frames on the time axis (x) calculated as T // HOP_LENGTH + 1
    # Why +1? Because: stft is center = True by default meaning
    # if stride is 12, seq len is 100 and window size is 25
    # first window is centered at 0, so it covers -12 to +12
    # last window is centered at 96, so it covers 84 to 100
    # so we have 9 windows centered at 0,12,24,36,48,60,72,84,96
    # formula: T // HOP_LENGTH + 1 which makes 480000 // 160 + 1 = 3001

    magnitude = (stft.abs() ** 2)[..., :-1]
    # magnitude.shape torch.Size([2, 201, 3000])
    # we remove the last frame to make it 3000 frames

    mel_np = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr/2.0).astype(np.float32)
    mel = torch.tensor(mel_np, dtype=magnitude.dtype, device=device)
    # mel.shape torch.Size([80, 201])
    # mel represents the energy stored in that frequency bin for a specific time frame

    mel_spec = torch.einsum('mf,bft->bmt', mel, magnitude)
    # with batched, M,F @ B,F,T -> B,M,T
    # mel_spec.shape torch.Size([2, 80, 3000])

    log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    max_ele = log_mel_spec.amax(dim=(-2, -1), keepdim=True) # to get the max element in each spectrogram

    # since the human ear can only capture 80 dB range, we clip the log mel spectrogram to max-8 as lower bound
    # 8 and not 80 because we are in log10 scale
    # this makes the log mel spectrogram values to be in the range of [max-8, max] where max ~ 0
    log_mel_spec = torch.maximum(log_mel_spec, max_ele - 8.0) 
    # then we normalize to [-1, 1] by dividing by 4
    log_mel_spec = (log_mel_spec + 4.0) / 4.0

    # Step 5: Ensure the spectrogram has the correct shape (80,3000) and not (80,3001)
    # if < 3000 frames, pad with zeros
    # if > 3000 frames, truncate to 3000 frames
    T = log_mel_spec.shape[-1]
    if T < n_frames:
        log_mel_spec = F.pad(log_mel_spec, (0, n_frames - T))
    elif T > n_frames:
        log_mel_spec = log_mel_spec[..., :n_frames]
    
    return log_mel_spec


def show_logmel(logmel: np.ndarray, sr=16000, hop_length=160, title="Log-Mel (Whisper style)"):
    assert logmel.ndim == 2, "expected (n_mels, n_frames)"
    n_mels, n_frames = logmel.shape
    dur_s = n_frames * hop_length / sr  # seconds
    extent = [0, dur_s, 0, n_mels]      # x: seconds, y: mel-bin index

    plt.figure(figsize=(10, 4))
    plt.imshow(logmel, aspect="auto", origin="lower", extent=extent)
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bin")
    plt.title(title)
    plt.colorbar(label="scaled log10 power")
    plt.tight_layout()
    plt.show()

def show_logmel_torch(logmel_t: torch.Tensor, idx=0, sr=16000, hop_length=160, title=None):
    lm = logmel_t[idx].detach().cpu().numpy()
    show_logmel(lm, sr=sr, hop_length=hop_length, title=title or f"Log-Mel sample {idx}")


def format_data(sample):
    """
    Convert a training sample into OpenAI conversation format.
    Each sample contains audio data and its corresponding transcription.
    """
    # Extract the transcription text
    transcription = sample['text']
    
    # Create the conversation format
    conversation = [
        {
            "content":[{
                            "text": "You are an ASR model that transcribes speech to text. Avoid additional explanation unless absolutely necessary.",
                            "type": "text"
                        }],
            "role":"system"
        },
        {
            "content": [{
                    "audio": sample['wav']['bytes'],  # Raw audio bytes
                    "type": "audio"
                },
                {
                    "text": "Transcribe this speech into text.",
                    "type": "text",
                }
            ],
            "role": "user"
        },
        {
            "content": [
                {
                    "text": transcription,
                    "type": "text"
                }
            ],
            "role": "assistant",
        }
    ]
    
    return conversation