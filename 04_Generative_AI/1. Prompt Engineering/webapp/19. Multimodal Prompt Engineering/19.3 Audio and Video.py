import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="19.3 Audio and Video | GenAI Roadmap",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .media-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🎬 19.3 Audio and Video — Temporal Multimodal Prompting & Frame Analysis")
st.markdown(
    """
Welcome to **Lesson 19.3: Audio and Video**.  
Static images capture a single frozen instant in time. Audio and video, however, introduce the **temporal dimension**: speech intonation, cadence, environmental background soundscapes, and moving visual sequences.

Prompting across audio and video requires mastering **timestamped frame extraction**, **scene-change detection**, **multimodal audio-visual alignment**, and **temporal event localization** (e.g. identifying exactly *when* an incident occurred).
"""
)

st.divider()

# ==============================================================================
# Section 1: How Frontier Models Ingest Audio and Video
# ==============================================================================
st.header("1. Audio & Video Modalities in Frontier Architectures", divider="rainbow")

col_av1, col_av2 = st.columns(2)

with col_av1:
    with st.container(border=True):
        st.subheader("🎙️ Audio Ingestion Paradigms")
        st.markdown(
            """
- **Two-Stage (ASR + LLM):** Audio is transcribed via Whisper to plaintext, then fed to an LLM. 
  - *Limitation:* Loses acoustic tone, sarcasm, shouting, crying, accents, and background acoustic events (gunshots, alarms).
- **Native Audio Tokens (GPT-4o / Gemini Native):** Raw audio waveforms are processed via spectrogram encoders into continuous audio tokens. 
  - *Advantage:* Model perceives speaker emotion, hesitation, pitch inflection, and background acoustics directly.
"""
        )

with col_av2:
    with st.container(border=True):
        st.subheader("📹 Video Ingestion Paradigms")
        st.markdown(
            """
- Video is computationally processed as an **ordered sequence of image frames** paired with synchronized audio.
- Models like Gemini 1.5 Pro natively ingest up to 1 hour of video (~3,600 frames at 1 fps) within a 1-million-token context window.
- The model computes cross-attention across both the visual frames and time-aligned audio timestamps, allowing precise temporal reasoning.
"""
        )

st.divider()

# ==============================================================================
# Section 2: Video Sampling Strategies: Uniform FPS vs. Keyframe Detection
# ==============================================================================
st.header("2. Video Frame Sampling Strategies & Token Economics", divider="rainbow")
st.markdown("Balancing visual temporal resolution with token budget constraints:")

sampling_df = pd.DataFrame({
    "Sampling Strategy": [
        "1. Uniform Subsampling (1 Frame/Sec)",
        "2. Coarse Subsampling (1 Frame every 5-10 Sec)",
        "3. Shot-Boundary / Keyframe Detection",
        "4. Motion-Triggered Dynamic Sampling"
    ],
    "Token Footprint (10-Min Video)": [
        "600 frames * ~258 tokens = ~154,800 tokens",
        "60 frames * ~258 tokens = ~15,480 tokens",
        "80-120 keyframes = ~20,000 to ~30,000 tokens",
        "Variable: 50 frames during quiet scenes; 300 during action"
    ],
    "Temporal Resolution": [
        "High (Captures short actions and fast gestures)",
        "Low (May miss actions occurring between intervals)",
        "Optimal (Captures every distinct camera angle and slide change)",
        "High Efficiency (Allocates compute only when scene changes)"
    ],
    "Best Production Use Case": [
        "High-stakes security footage, sports motion analysis, industrial robotics",
        "Long podcast videos, lecture summaries, movie plot synopses",
        "Keynote slide presentations, webinar breakdown, document tutorials",
        "CCTV warehouse surveillance, smart doorbell event logging"
    ]
})
st.dataframe(sampling_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Temporal Multimodal Video Pipeline", divider="rainbow")
st.markdown("How raw video streams are decomposed into timestamped frames and ingested for temporal reasoning:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        video [label="Raw Video File (.mp4)\n(1080p @ 30 FPS)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        sampler [label="Frame Sampler\n(OpenCV 1 FPS Subsampler)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        stamps [label="Timestamped Frames\n[00:01, 00:02, ...]", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        vlm [label="Multimodal Transformer\n(Temporal Cross-Attention)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        event_log [label="Temporal Action Log\n(Events with Timestamps)", fillcolor="#10B981", fontcolor="#FFFFFF"]

        video -> sampler -> stamps -> vlm -> event_log
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Video & Audio Reasoning Simulator
# ==============================================================================
st.header("4. Interactive Temporal Reasoning Simulator", divider="rainbow")
st.markdown("Simulate how a multimodal model pinpoints timestamps and transcribes acoustic events:")

media_scenario = st.selectbox(
    "Select Video / Audio Scenario",
    [
        "Warehouse CCTV Safety Violation Audit (Forklift Near-Miss)",
        "Customer Service Escalation Call (Acoustic Distress & Sarcasm)",
        "Software Engineering Screen Recording (Bug Reproduction Trace)"
    ]
)

col_ms1, col_ms2 = st.columns([1, 1])

with col_ms1:
    st.subheader("⏱️ Chronological Frame & Audio Stream")
    if "CCTV" in media_scenario:
        st.markdown(
            """
```text
[00:12] Frame 12: Forklift enters aisle 4 at standard speed.
[00:14] Frame 14: Pedestrian worker walks past racking without high-vis vest.
[00:16] Frame 16: Forklift turns blind corner; driver hits brakes abruptly.
[00:17] Audio: Loud metal screeching and forklift horn blast.
[00:18] Frame 18: Forklift halts 1.2 meters from pedestrian; no impact.
```
"""
        )
    elif "Customer Service" in media_scenario:
        st.markdown(
            """
```text
[00:05] User: "My internet has been down for 3 days." (Calm tone)
[00:15] Agent: "Our next technician visit is Thursday."
[00:18] User: "Thursday?! Oh, absolutely brilliant service, thank you SO much!" 
        Audio Features: Pitch spikes +24dB, heavy ironic cadence, acoustic sarcasm.
[00:24] User: Slams phone receiver / abrupt disconnect.
```
"""
        )
    else:
        st.markdown(
            """
```text
[00:02] Frame 02: User clicks 'Checkout with Apple Pay' button.
[00:04] Frame 04: Modal spinner appears; network tab shows 504 GATEWAY_TIMEOUT.
[00:06] Frame 06: Red banner displays: 'Payment provider unreachable'.
[00:08] Frame 08: User rage-clicks button 4 times in 1.5 seconds.
```
"""
        )

with col_ms2:
    st.subheader("📤 Multimodal Temporal Reasoning Output")
    if "CCTV" in media_scenario:
        event_report = {
            "incident_detected": True,
            "incident_type": "NEAR_MISS_PEDESTRIAN_FORKLIFT",
            "critical_timestamp": "00:16 - 00:18",
            "safety_violations": [
                "Pedestrian operating in active forklift zone without high-visibility vest (00:14)",
                "Forklift rounded blind corner without prior horn tap (00:16)"
            ],
            "acoustic_evidence": "Horn blast and brake screech confirmed abrupt emergency deceleration (00:17)",
            "severity_tier": "OSHA_FLAG_TIER_2"
        }
        st.json(event_report)
        st.warning("⚠️ Near-miss incident logged with exact second-level timestamps for OSHA review.")

    elif "Customer Service" in media_scenario:
        event_report = {
            "call_outcome": "CUSTOMER_CHURN_HIGH_RISK",
            "sentiment_classification": "EXTREMELY_FRUSTRATED_SARCASTIC",
            "acoustic_markers": {
                "perceived_emotion": "Rage / Disgust masked by sarcasm",
                "disconnection_timestamp": "00:24",
                "irony_flag": True
            },
            "root_cause": "Unacceptable 3-day SLA delay for critical internet outage",
            "automated_action": "TRIGGER_SUPERVISOR_CALLBACK_STAT"
        }
        st.json(event_report)
        st.error("🚨 Native audio analysis correctly detected acoustic sarcasm rather than naive positive text.")

    else:
        event_report = {
            "ux_defect_detected": True,
            "error_timestamp": "00:04",
            "failure_trigger": "504 Gateway Timeout from Apple Pay relay endpoint",
            "user_behavior": "Rage clicking (4 clicks within 1.5s at 00:08)",
            "engineering_remediation": "Disable checkout button immediately upon first click to prevent duplicate transaction submissions."
        }
        st.json(event_report)
        st.info("ℹ️ User friction and technical bug localized from screen recording.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: OpenCV Keyframe Sampler", divider="rainbow")
st.markdown("Python script to extract 1 frame per second from video and package timestamped multimodal payloads:")

st.code(
    '''import cv2
import base64
from typing import List, Dict, Any

def extract_timestamped_frames(video_path: str, fps_sample_rate: float = 1.0) -> List[Dict[str, Any]]:
    """
    Subsamples video at a fixed rate (e.g. 1 frame every second),
    extracts base64 images, and labels each frame with an exact MM:SS timestamp.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_sample_rate)
    
    frames_payload = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            timestamp_sec = int(frame_idx / video_fps)
            minutes = timestamp_sec // 60
            seconds = timestamp_sec % 60
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            
            # Encode frame to JPEG in memory
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            frames_payload.append({
                "timestamp": timestamp_str,
                "base64_image": b64_str
            })
            
        frame_idx += 1
        
    cap.release()
    return frames_payload

VIDEO_AUDIT_SYSTEM_PROMPT = """<system>
You are an expert Video and Temporal Activity Analyst.
Analyze the sequence of timestamped frames.
Requirements:
1. Identify all notable actions, state transitions, or safety violations.
2. For every detected event, cite the exact timestamp in [MM:SS] format.
3. Output a structured JSON timeline of events.
</system>
"""

print("Video Frame Extractor ready for temporal multimodal analysis.")
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Why does native audio token processing detect sarcasm and emotion better than Whisper transcription + LLM?", expanded=True):
        st.markdown(
            """
- **Text-Only Loss**: When a user says *"Oh, wonderful service, thank you!"* with heavy sarcasm and anger, an ASR transcriber emits those exact literal words. A downstream text LLM reads the words and misinterprets the message as glowing positive praise.
- **Native Audio Alignment**: Native audio models (GPT-4o, Gemini) process acoustic spectrogram features (pitch elevation, vocal tension, elongation of vowels, prolonged pauses). The model perceives the **acoustic emotional valence** directly, correctly classifying the utterance as severe dissatisfaction.
"""
        )
    with st.expander("Q2: How do you manage token budgets when passing a 30-minute video into an LLM context window?"):
        st.markdown(
            """
At 1 frame per second, 30 minutes of video generates 1,800 frames ($\approx 460,000$ tokens), which is expensive and slow.
**Production Optimization**:
1. **Scene-Change Sampling**: Use OpenCV or FFmpeg (`select='gt(scene,0.3)'`) to sample only when the visual scene shifts (dropping duplicate frames where the camera is still).
2. **Audio-Guided Keyframing**: Run audio transcription first; extract video frames *only* around timestamps where spoken keywords or acoustic anomalies occur.
3. **Thumbnail Contact Sheets**: Tile 9 frames (3x3 grid) into a single image to reduce API image calls.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we feed full 60 FPS video directly to an LLM API to get maximum accuracy?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, 60 FPS gives the model every pixel of motion."*
- **Winning Response**: *"Never. 60 FPS video generates 3,600 frames per minute, consuming nearly 1 million tokens in 60 seconds. This exceeds context limits, costs hundreds of dollars per minute, and crashes inference engines. 99% of business tasks (surveillance, tutorials, defect inspection) require only **0.5 to 1.0 FPS** to capture complete semantic state changes."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Native Audio Ingestion**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Processing raw audio waveforms into tokens directly, preserving acoustic pitch, emotion, sarcasm, and background soundscapes.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Temporal Event Localization**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The capability of an LLM to pinpoint the exact timestamp [MM:SS] when a visual or acoustic action occurred in a video stream.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Scene-Change Subsampling**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Extracting video frames only when camera angles or scene transitions occur, saving up to 80% of video token costs.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 19.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Temporal Dimension**: Video introduces time-series frame sequences; audio introduces acoustic emotion.
2. **Native Audio Superiority**: Native audio tokens preserve pitch, sarcasm, and inflection lost in transcripts.
3. **Subsampling Economics**: Subsample video at 1.0 FPS or use scene detection to control token expenditure.
4. **Timestamped Grounding**: Always require models to cite exact [MM:SS] timestamps for verified actions.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know why native audio models classify emotional sarcasm more accurately than text transcripts.", value=True, key="chk_19_3_1")
    st.checkbox("I understand how to subsample video frames using OpenCV in Python.", value=True, key="chk_19_3_2")
    st.checkbox("I can calculate token budgets for video ingestion pipelines.", value=True, key="chk_19_3_3")
    st.checkbox("I know how to prompt for temporal event localization with second-level timestamps.", value=True, key="chk_19_3_4")
