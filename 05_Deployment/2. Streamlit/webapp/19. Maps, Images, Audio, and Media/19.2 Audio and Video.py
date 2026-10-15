import streamlit as st
import io
import wave
import struct
import math
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="19.2 Audio and Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.html(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    </style>
    """
)

st.markdown('<div class="main-title">19.2 Audio and Video in Streamlit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production architectural guide to st.audio(), st.video(), in-memory PCM waveform synthesis, multimedia streaming, upload pipelines, codec support, and bandwidth optimization.</div>',
    unsafe_allow_html=True
)

# Helper function to generate clean in-memory PCM WAV audio without external dependencies
def generate_synthetic_pcm_wav(frequency=440.0, duration_sec=1.5, sample_rate=16000, wave_type="Sine"):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    if wave_type == "Sine":
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    elif wave_type == "Chirp (Frequency Sweep)":
        waveform = 0.5 * np.sin(2 * np.pi * (frequency + 200 * t) * t)
    else:  # Square wave
        waveform = 0.4 * np.sign(np.sin(2 * np.pi * frequency * t))
        
    # Convert float array to 16-bit PCM integers
    audio_pcm = (waveform * 32767).astype(np.int16)
    
    # Pack into standard RIFF WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm.tobytes())
    buffer.seek(0)
    return buffer.getvalue(), waveform, sample_rate

# -----------------------------------------------------------------------------
# TOPIC 1: ST.AUDIO() FUNDAMENTALS & PARAMETER SIGNATURE
# -----------------------------------------------------------------------------
st.header("1. st.audio() Fundamentals & Parameter Signature", divider="blue")

st.markdown(
    r"""
`st.audio()` renders an HTML5 audio player widget supporting raw bytes, numpy arrays, file paths, and web URLs.
It natively handles streaming playback, loop options, and precise start/end seek timestamps.

```python
st.audio(
    data,
    format="audio/wav",
    start_time=0,
    *,
    sample_rate=None,
    end_time=None,
    loop=False,
    autoplay=False
)
```
"""
)

st.markdown(
    r"""
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `bytes`, `str`, `np.ndarray`, `io.BytesIO` | *Required* | Raw audio bytes, file path, remote HTTPS URL, or 1D/2D numpy array. |
| `format` | `str` | `"audio/wav"` | MIME type: `"audio/wav"`, `"audio/mp3"`, `"audio/ogg"`, `"audio/aac"`, or `"audio/flac"`. |
| `start_time` | `int` or `float` | `0` | Initial playback seek position in seconds. |
| `sample_rate` | `int` | `None` | Required when passing raw numpy array buffers directly to infer playback speed. |
| `end_time` | `int` or `float` | `None` | Playback cutoff timestamp in seconds. |
| `loop` | `bool` | `False` | Continuously loops the audio playback once reached the end. |
| `autoplay` | `bool` | `False` | Automatically begins playback on mount (subject to browser security policies). |
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import numpy as np

# 1. Direct NumPy array rendering (16kHz sine wave)
sample_rate = 16000
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))
signal = np.sin(2 * np.pi * 440 * t) # A4 pitch

# Render audio directly from NumPy array
st.audio(
    signal,
    sample_rate=sample_rate,
    start_time=0,
    loop=False,
    format="audio/wav"
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Native st.audio() Player Demonstration")
        
        wav_bytes, _, _ = generate_synthetic_pcm_wav(frequency=440.0, duration_sec=2.0)
        st.audio(wav_bytes, format="audio/wav")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Format", "audio/wav")
        c2.metric("Sample Rate", "16,000 Hz")
        c3.metric("Bit Depth", "16-bit PCM")

# -----------------------------------------------------------------------------
# TOPIC 2: ST.VIDEO() FUNDAMENTALS & PARAMETER SIGNATURE
# -----------------------------------------------------------------------------
st.header("2. st.video() Fundamentals & Parameter Signature", divider="blue")

st.markdown(
    r"""
`st.video()` embeds an HTML5 video player capable of decoding local MP4/WebM files, byte buffers, 
and remote video platforms (YouTube, Vimeo, CDN streams). It supports closed-caption subtitle tracks via WebVTT.

```python
st.video(
    data,
    format="video/mp4",
    start_time=0,
    *,
    subtitles=None,
    end_time=None,
    loop=False,
    autoplay=False,
    muted=False
)
```
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

# Embed CDN or public video stream
video_url = "https://www.w3schools.com/html/mov_bbb.mp4"

st.video(
    video_url,
    format="video/mp4",
    start_time=2,      # Start 2 seconds in
    end_time=8,        # Stop after 8 seconds
    loop=True,
    muted=True,
    subtitles="captions.vtt" # Optional WebVTT track
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Native st.video() Player Demonstration")
        
        # Public standards-compliant test video
        sample_video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        
        seek_time = st.slider("Initial Seek Timestamp (seconds)", 0, 10, 0, key="video_seek_slider")
        is_muted = st.checkbox("Mute Audio Track", value=True, key="video_mute_toggle")
        
        st.video(
            sample_video_url,
            format="video/mp4",
            start_time=seek_time,
            muted=is_muted
        )
        st.info("💡 Supports direct YouTube URLs: `st.video('https://youtu.be/...')`")

# -----------------------------------------------------------------------------
# TOPIC 3: UPLOADING AUDIO
# -----------------------------------------------------------------------------
st.header("3. Uploading Audio Files & Waveform Inspection", divider="blue")

st.markdown(
    r"""
Audio ingestion workflows in Speech-to-Text, Voice Cloning, and Acoustic Telemetry use `st.file_uploader()` 
with MIME filtering (`type=["wav", "mp3", "ogg", "flac"]`). The in-memory buffer can be visualized as an 
interactive time-domain amplitude waveform with `st.line_chart()`.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import wave
import numpy as np

uploaded_audio = st.file_uploader("Upload Acoustic Sample", type=["wav", "mp3"])

if uploaded_audio:
    st.audio(uploaded_audio)
    
    # Read raw PCM frames for waveform analysis
    with wave.open(uploaded_audio, "rb") as wf:
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        signal = np.frombuffer(frames, dtype=np.int16)
        
    # Plot amplitude envelope
    st.line_chart(signal[::100], height=150)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Audio Ingestion & Time-Domain Waveform Visualizer")
        
        audio_src = st.radio(
            "Audio Ingestion Mode",
            ["Simulate Synthetic Upload (Chirp 800Hz)", "Live File Uploader"],
            horizontal=True,
            key="audio_up_mode"
        )
        
        if audio_src == "Live File Uploader":
            usr_audio = st.file_uploader("Upload .wav or .mp3 file", type=["wav", "mp3"], key="usr_audio_up")
            if usr_audio:
                st.audio(usr_audio)
            else:
                st.info("Awaiting audio file upload...")
        else:
            sim_wav, sim_wave_arr, sim_sr = generate_synthetic_pcm_wav(
                frequency=600.0,
                duration_sec=1.0,
                sample_rate=16000,
                wave_type="Chirp (Frequency Sweep)"
            )
            st.audio(sim_wav, format="audio/wav")
            
            st.caption("Time-Domain Amplitude Envelope (Downsampled 1:50)")
            st.line_chart(sim_wave_arr[::50], height=140)
            st.caption(f"Total Audio Samples: {len(sim_wave_arr):,} | Duration: 1.00s | Nyquist Freq: 8,000 Hz")

# -----------------------------------------------------------------------------
# TOPIC 4: UPLOADING VIDEO
# -----------------------------------------------------------------------------
st.header("4. Uploading Video Files & Frame Diagnostics", divider="blue")

st.markdown(
    r"""
Video uploads in Streamlit serve object tracking, action recognition, and automated video captioning.
Because video files are typically large, configure `maxUploadSize` in `.streamlit/config.toml` (default is 200MB).
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

uploaded_video = st.file_uploader(
    "Upload Surveillance Video",
    type=["mp4", "mov", "avi", "webm"]
)

if uploaded_video:
    # 1. Playback video preview
    st.video(uploaded_video)
    
    # 2. Extract file telemetry
    file_size_mb = uploaded_video.size / (1024 * 1024)
    st.metric("File Size", f"{file_size_mb:.2f} MB")
    st.write(f"MIME Type: {uploaded_video.type}")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Video Pipeline Ingestion Simulator")
        
        col_up_info, col_up_meta = st.columns([1.5, 1])
        with col_up_info:
            st.info("📹 **Simulated File:** `security_feed_cam02_clip.mp4` (H.264 / AAC)")
            sample_mp4 = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4"
            st.video(sample_mp4, format="video/mp4", start_time=0)
        with col_up_meta:
            st.markdown("**File Telemetry:**")
            st.metric("Container Size", "4.82 MB")
            st.metric("Resolution", "1080p (1920x1080)")
            st.metric("Frame Rate", "29.97 FPS")
            st.caption("MPEG-4 Part 14 Container (QuickTime/AVC compliant)")

# -----------------------------------------------------------------------------
# TOPIC 5: PLAYING MEDIA FILES FROM DISK & CACHED BUFFERS
# -----------------------------------------------------------------------------
st.header("5. Playing Media Files from Disk & Cached Buffers", divider="blue")

st.markdown(
    r"""
Reading media files from disk on every Streamlit script rerun introduces severe I/O bottlenecks.
Use `@st.cache_data` to cache the raw bytes in memory so repeated page reruns consume zero disk I/O.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

# Cache raw bytes in memory to prevent repeated disk reads
@st.cache_data(show_spinner="Loading audio buffer...")
def load_media_asset(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()

# Zero-overhead playback on subsequent reruns
audio_bytes = load_media_asset("soundtrack.mp3")
st.audio(audio_bytes, format="audio/mp3")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Zero-I/O In-Memory Cache Simulator")
        
        clip_sel = st.selectbox(
            "Select Audio Asset from Cached Soundbank",
            ["Radar Ping (440Hz)", "Low Alert Beacon (220Hz)", "Sonar Pulse (880Hz)"],
            key="soundbank_sel"
        )
        
        freq_map = {"Radar Ping (440Hz)": 440.0, "Low Alert Beacon (220Hz)": 220.0, "Sonar Pulse (880Hz)": 880.0}
        
        cached_wav, _, _ = generate_synthetic_pcm_wav(frequency=freq_map[clip_sel], duration_sec=0.8)
        st.audio(cached_wav, format="audio/wav")
        st.success("⚡ Buffer served directly from `@st.cache_data` memory cache (0.2 ms latency).")

# -----------------------------------------------------------------------------
# TOPIC 6: DISPLAYING GENERATED AUDIO (AI TTS & SYNTHESIS)
# -----------------------------------------------------------------------------
st.header("6. Displaying Generated Audio (AI TTS & Synthesis)", divider="blue")

st.markdown(
    r"""
AI Text-to-Speech (TTS) models (e.g., ElevenLabs, OpenAI Audio TTS, Bark, Coqui TTS) return 
raw audio bytes or base64 strings. In Streamlit, pipe the returned bytes directly into `st.audio()` 
without saving temporary files to disk.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
from openai import OpenAI

client = OpenAI()

def synthesize_ai_speech(text_prompt: str, voice: str = "alloy") -> bytes:
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text_prompt
    )
    # Streamlit receives raw binary bytes directly
    return response.content

tts_input = st.text_input("Text for TTS:", "Welcome to Streamlit 1.65.")
if st.button("Generate AI Voice"):
    audio_data = synthesize_ai_speech(tts_input)
    st.audio(audio_data, format="audio/mp3")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("AI Text-to-Speech Synthesizer Simulator")
        
        tts_text = st.text_input(
            "Enter Script to Synthesize",
            value="All neural inference pipelines are healthy. Model throughput is operating at optimal capacity.",
            key="tts_text_input"
        )
        
        c_voice, c_pitch = st.columns(2)
        with c_voice:
            voice_style = st.selectbox("TTS Voice Profile", ["Alloy (Neutral)", "Echo (Warm)", "Nova (Energetic)"], key="tts_voice")
        with c_pitch:
            pitch_freq = st.slider("Base Frequency (Hz)", 300, 700, 480, key="tts_pitch")
            
        if st.button("Synthesize Speech Buffer", key="tts_synth_btn", width="stretch"):
            with st.spinner(f"Synthesizing speech via {voice_style.split()[0]} neural acoustic model..."):
                tts_wav, _, _ = generate_synthetic_pcm_wav(frequency=float(pitch_freq), duration_sec=2.2, wave_type="Sine")
                st.audio(tts_wav, format="audio/wav")
                st.success("Audio synthesis completed. Buffer size: 70.4 KB (16 kHz, 16-bit Mono).")

# -----------------------------------------------------------------------------
# TOPIC 7: FILE FORMAT HANDLING & CODEC COMPATIBILITY
# -----------------------------------------------------------------------------
st.header("7. File Format Handling & Codec Compatibility", divider="blue")

st.markdown(
    r"""
Different browsers support different container formats and underlying codecs.
Serving an unsupported codec (such as H.265/HEVC or uncompressed AVI) can result in a blank player or audio-only playback.
"""
)

st.markdown(
    r"""
| Format Category | Supported Container | Underlying Codec | Browser Support | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Audio** | `.wav` | Linear PCM | 100% (Universal) | Uncompressed, largest file size |
| **Audio** | `.mp3` | MPEG-1 Audio Layer III | 100% (Universal) | Industry standard lossy compression |
| **Audio** | `.ogg` | Ogg Vorbis / Opus | >95% (Modern browsers) | Excellent open-source codec |
| **Audio** | `.flac` | Free Lossless Audio | >90% | Lossless audio compression |
| **Video** | `.mp4` | H.264 (AVC) + AAC | 100% (Universal) | **Most recommended** for production web apps |
| **Video** | `.webm` | VP8 / VP9 / AV1 | Chrome, Firefox, Edge | Modern, royalty-free web video |
| **Video (Caution)** | `.avi` / `.mov` | Apple ProRes / Cinepak | Spotty / Unsupported | Must be transcoded to H.264 first |
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

# Format mapping dictionary for dynamic MIME header assignment
MIME_MAP = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "ogg": "audio/ogg"
}

file_ext = uploaded_file.name.split(".")[-1].lower()
mime_type = MIME_MAP.get(file_ext, "application/octet-stream")

if "video" in mime_type:
    st.video(uploaded_file, format=mime_type)
elif "audio" in mime_type:
    st.audio(uploaded_file, format=mime_type)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("MIME Header Resolver & Codec Verifier")
        
        test_filename = st.selectbox(
            "Select Test Media Asset",
            ["interview_recording.wav", "lecture_presentation.mp4", "ambient_soundtrack.ogg", "screen_capture.webm"],
            key="test_file_sel"
        )
        
        ext = test_filename.split(".")[-1]
        mime_map = {
            "wav": ("audio/wav", "PCM Uncompressed Audio", "✅ Universal"),
            "mp4": ("video/mp4", "AVC / H.264 Video", "✅ Universal"),
            "ogg": ("audio/ogg", "Ogg Vorbis Audio", "✅ Modern Browsers"),
            "webm": ("video/webm", "Google VP9 Web Video", "✅ Modern Browsers")
        }
        
        mime, desc, comp = mime_map[ext]
        st.write(f"**Resolved MIME Type:** `{mime}`")
        st.write(f"**Codec Family:** {desc}")
        st.write(f"**Compatibility:** {comp}")

# -----------------------------------------------------------------------------
# TOPIC 8: MEDIA SIZE CONSIDERATIONS & BANDWIDTH OPTIMIZATION
# -----------------------------------------------------------------------------
st.header("8. Media Size Considerations & Bandwidth Optimization", divider="blue")

st.markdown(
    r"""
Media files are typically the largest assets in a Streamlit application. Handling multi-gigabyte video or uncompressed 
audio can exhaust server RAM and exceed upload ceilings.

### Key Performance Strategies:
1. **Configure Max Upload Ceilings**:
   In `.streamlit/config.toml`:
   ```toml
   [server]
   maxUploadSize = 500  # Size in MB (default is 200)
   ```
2. **Stream Remote URLs Rather Than Local Buffers**: Passing an HTTPS URL (`st.video('https://...')`) offloads video streaming directly to the client browser without consuming server memory.
3. **Downsample Audio Sample Rates**: Downsample speech from 44.1 kHz to 16 kHz before passing to ASR models, reducing byte payload by $64\%$.
4. **Avoid Writing Temporary Files to Disk**: Process audio and video in-memory using `io.BytesIO`.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''# Downsample 44.1kHz stereo to 16kHz mono in Python
import numpy as np

def downsample_audio(audio_44k: np.ndarray, orig_sr: int = 44100, target_sr: int = 16000):
    step = orig_sr / target_sr
    indices = np.arange(0, len(audio_44k), step).astype(int)
    downsampled = audio_44k[indices]
    return downsampled

# Yields 64% reduction in memory footprint
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Bandwidth & Memory Footprint Calculator")
        
        duration_min = st.slider("Audio Duration (Minutes)", 1, 60, 10, key="duration_slider_calc")
        
        # Audio calculation
        wav_44k_mb = (duration_min * 60 * 44100 * 2 * 2) / (1024 * 1024) # 44.1kHz 16-bit stereo
        wav_16k_mb = (duration_min * 60 * 16000 * 2 * 1) / (1024 * 1024) # 16kHz 16-bit mono
        mp3_128k_mb = (duration_min * 60 * 128 * 1000 / 8) / (1024 * 1024) # 128kbps MP3
        
        c_w44, c_w16, c_mp3 = st.columns(3)
        c_w44.metric("Uncompressed 44.1k Stereo", f"{wav_44k_mb:.1f} MB")
        c_w16.metric("Optimized 16k Mono", f"{wav_16k_mb:.1f} MB", delta=f"-{(1 - wav_16k_mb/wav_44k_mb)*100:.0f}% RAM")
        c_mp3.metric("Compressed 128kbps MP3", f"{mp3_128k_mb:.1f} MB", delta=f"-{(1 - mp3_128k_mb/wav_44k_mb)*100:.0f}%")
        
        st.info("💡 For production LLM/Whisper pipelines, converting to 16kHz Mono reduces upload latency by ~64%.")
