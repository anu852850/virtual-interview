import streamlit as st
import streamlit.components.v1 as components
import base64
import time
import uuid
import os
import html
import logging
import shutil

from interview_logic import (
    ask_interview_question,
    speak_question,
    get_candidate_answer,
    evaluate_answer,
    evaluate_answer_prompt,
    setup_llm,
    Evaluate,
    llm,
    cleanup_old_vectorstores,
    ResumeProcessingError,
    LLMCallError,
    TranscriptionError,
    TTSError,
)

logger = logging.getLogger("app")


# ============================================================
# CONFIG
# ============================================================

NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "5"))
MAX_RESUME_MB = int(os.getenv("MAX_RESUME_MB", "10"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
AUDIO_DIR = os.getenv("AUDIO_DIR", "Question_Audios")
ANSWER_WAIT_SECONDS = int(os.getenv("ANSWER_WAIT_SECONDS", "5"))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Virtual Interview",
    layout="centered"
)

st.title("🤖 AI Virtual Interview Platform")


# ============================================================
# CLEANUP HELPERS
# ============================================================

def cleanup_question_audios(session_id):
    """Interview khatam hone pe (ya crash/interrupt hone pe) us session ki
    saari question-audio files delete karta hai."""
    if os.path.exists(AUDIO_DIR):
        for f in os.listdir(AUDIO_DIR):
            if f.startswith(session_id):
                try:
                    os.remove(os.path.join(AUDIO_DIR, f))
                except OSError:
                    logger.exception("Failed to remove audio file %s", f)


def cleanup_resume(session_id):
    """Uploaded resume PDF ko delete karta hai — PII hai, session khatam
    hone ke baad disk pe nahi rehna chahiye."""
    resume_path = st.session_state.get("resume_path")
    if resume_path and os.path.exists(resume_path):
        try:
            os.remove(resume_path)
        except OSError:
            logger.exception("Failed to remove resume file %s", resume_path)


def cleanup_session(session_id):
    cleanup_question_audios(session_id)
    cleanup_resume(session_id)


# ============================================================
# AVATAR + AUDIO WAVEFORM
# ============================================================

def render_avatar_with_waveform(audio_path, height=380):

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    html_code = f"""
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        background:#0e1117;
        border-radius:16px;
        padding:24px;
    ">

        <div style="
            position:relative;
            width:220px;
            height:220px;
            display:flex;
            align-items:center;
            justify-content:center;
        ">

            <div id="wave-container"
                 style="
                    position:absolute;
                    width:100%;
                    height:100%;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    gap:6px;
                    z-index:1;
                 ">

                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>

            </div>

            <div style="
                position:relative;
                z-index:2;
                width:170px;
                height:170px;
                border-radius:50%;
                background:#1c1f2a;
                box-shadow:0 0 25px rgba(0,200,255,0.35);
                display:flex;
                align-items:center;
                justify-content:center;
                overflow:hidden;
            ">

                <svg
                    width="90"
                    height="90"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <circle
                        cx="12"
                        cy="8"
                        r="4"
                        fill="#6b7684"
                    />

                    <path
                        d="M4 20c0-4.4 3.6-8 8-8s8 3.6 8 8"
                        fill="#6b7684"
                    />
                </svg>

            </div>

        </div>

        <p id="status-text"
           style="
              color:#8a93a3;
              font-family:sans-serif;
              margin-top:14px;
              font-size:14px;
           ">
            Speaking...
        </p>

        <audio id="q-audio">
            <source
                src="data:audio/mp3;base64,{audio_b64}"
                type="audio/mp3"
            >
        </audio>

    </div>

    <style>

        .bar {{
            width:6px;
            height:20px;
            background:linear-gradient(
                180deg,
                #4fd1ff,
                #1e6fa8
            );
            border-radius:4px;
            transition:height 0.08s ease;
        }}

    </style>

    <script>

        const audio =
            document.getElementById("q-audio");

        const bars =
            document.querySelectorAll(".bar");

        const statusText =
            document.getElementById("status-text");


        const AudioCtx =
            window.AudioContext ||
            window.webkitAudioContext;


        const audioCtx =
            new AudioCtx();


        const analyser =
            audioCtx.createAnalyser();

        analyser.fftSize = 64;


        const dataArray =
            new Uint8Array(
                analyser.frequencyBinCount
            );


        // ONLY ONE AUDIO OUTPUT PATH
        const source =
            audioCtx.createMediaElementSource(audio);

        source.connect(analyser);

        analyser.connect(audioCtx.destination);


        function animate() {{

            analyser.getByteFrequencyData(
                dataArray
            );


            bars.forEach((bar, i) => {{

                const value =
                    dataArray[i * 3] || 0;

                const height =
                    20 + (value / 255) * 90;

                bar.style.height =
                    height + "px";

            }});


            if (!audio.paused && !audio.ended) {{

                requestAnimationFrame(
                    animate
                );

            }}

        }}


        audio.addEventListener("play", async () => {{

            if (audioCtx.state === "suspended") {{

                await audioCtx.resume();

            }}

            statusText.innerText =
                "🎙️ Speaking...";

            animate();

        }});


        audio.addEventListener("ended", () => {{

            statusText.innerText =
                "✅ Done. Your turn now.";

            bars.forEach(bar => {{

                bar.style.height = "20px";

            }});

        }});


        // Start audio ONCE
        window.addEventListener("load", async () => {{

            try {{

                await audioCtx.resume();

                audio.currentTime = 0;

                await audio.play();

            }} catch (err) {{

                console.log(
                    "Autoplay blocked:",
                    err
                );

            }}

        }});

    </script>
    """

    components.html(
        html_code,
        height=height
    )


# ============================================================
# STARTUP HOUSEKEEPING (stale vectorstore cache cleanup)
# ============================================================

if "startup_cleanup_done" not in st.session_state:
    try:
        cleanup_old_vectorstores()
    except Exception:
        logger.exception("Vectorstore cleanup on startup failed (non-fatal)")
    st.session_state.startup_cleanup_done = True


# ============================================================
# RESUME UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None and uploaded_file.size > MAX_RESUME_MB * 1024 * 1024:
    st.error(f"File too big ({uploaded_file.size / 1_000_000:.1f} MB). Max allowed is {MAX_RESUME_MB} MB.")
    st.stop()


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "started" not in st.session_state:
    st.session_state.started = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if "round_num" not in st.session_state:
    st.session_state.round_num = 0

if "question" not in st.session_state:
    st.session_state.question = None

if "resume_section" not in st.session_state:
    st.session_state.resume_section = None

if "answer_processed" not in st.session_state:
    st.session_state.answer_processed = False

if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "resume_path" not in st.session_state:
    st.session_state.resume_path = None


# ============================================================
# START INTERVIEW
# ============================================================

if (
    uploaded_file is not None
    and not st.session_state.started
):

    st.session_state.started = True
    st.session_state.round_num = 0
    st.session_state.question = None
    st.session_state.resume_section = None
    st.session_state.answer_processed = False
    st.session_state.interview_finished = False

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        resume_path = os.path.join(
            UPLOAD_DIR,
            f"{st.session_state.session_id}.pdf"
        )

        with open(resume_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.resume_path = resume_path

    except OSError as e:
        logger.exception("Failed to save uploaded resume")
        st.error(f"Could not save your resume: {e}")
        st.session_state.started = False
        st.stop()

    st.session_state.conversation_history.clear()

    st.rerun()


# ============================================================
# INTERVIEW
# ============================================================

if (
    st.session_state.started
    and not st.session_state.interview_finished
):

    round_num = st.session_state.round_num


    # ========================================================
    # CHECK IF INTERVIEW FINISHED
    # ========================================================

    if round_num >= NUM_QUESTIONS:

        st.session_state.interview_finished = True

        cleanup_session(st.session_state.session_id)

        st.success("🎉 Interview khatam!")

        st.stop()


    # ========================================================
    # GENERATE QUESTION
    # ========================================================

    if st.session_state.question is None:

        with st.spinner("Question generate ho raha hai..."):
            try:
                question, resume_section = ask_interview_question(
                    st.session_state.resume_path,
                    st.session_state.conversation_history
                )
            except (ResumeProcessingError, LLMCallError) as e:
                st.error(f"⚠️ {e}")
                st.info("Dobara try karne ke liye page reload karo, ya thodi der baad aao.")
                st.stop()
            except Exception:
                logger.exception("Unexpected error while generating question")
                st.error("⚠️ Kuch unexpected problem hui question generate karte waqt.")
                st.stop()

        st.session_state.question = question
        st.session_state.resume_section = resume_section
        st.session_state.answer_processed = False

        st.rerun()

    question = st.session_state.question


    # ========================================================
    # GENERATE QUESTION AUDIO
    # ========================================================

    os.makedirs(AUDIO_DIR, exist_ok=True)

    audio_filename = os.path.join(
        AUDIO_DIR,
        f"{st.session_state.session_id}_question_{round_num}.mp3"
    )

    try:
        audio_path, word_timings = speak_question(
            question,
            filename=audio_filename
        )
    except TTSError as e:
        # Non-fatal: fall back to text-only question, skip audio/waveform.
        logger.error("TTS failed, falling back to text-only question: %s", e)
        st.warning("🔇 Audio generate nahi ho paaya, question text mein padho.")
        audio_path, word_timings = None, []


    # ========================================================
    # SHOW AVATAR (only if audio generated successfully)
    # ========================================================

    if audio_path:
        avatar_placeholder = st.empty()
        with avatar_placeholder.container():
            render_avatar_with_waveform(audio_path)


    # ========================================================
    # QUESTION TEXT
    # ========================================================

    st.markdown(f"### Question {round_num + 1}")

    placeholder = st.empty()


    # ========================================================
    # WORD HIGHLIGHT (text escaped before HTML injection)
    # ========================================================

    prev_offset = 0

    if word_timings:
        for i, timing in enumerate(word_timings):

            sentence = ""

            for j, w in enumerate(word_timings):

                safe_word = html.escape(w['word'])

                if j < i:
                    sentence += f"<span style='color:green'>{safe_word}</span> "
                elif j == i:
                    sentence += (
                        f"<span style='color:red;font-size:28px;"
                        f"font-weight:bold'>{safe_word}</span> "
                    )
                else:
                    sentence += f"<span style='color:gray'>{safe_word}</span> "

            placeholder.markdown(f"<h3>{sentence}</h3>", unsafe_allow_html=True)

            gap = timing["offset"] - prev_offset
            time.sleep(max(gap, 0))
            prev_offset = timing["offset"]
    else:
        # No TTS audio/timings available — just show the plain question text.
        placeholder.markdown(f"<h3>{html.escape(question)}</h3>", unsafe_allow_html=True)


    # ========================================================
    # WAIT BEFORE ANSWER
    # ========================================================

    st.info(f"⏳ Ab jawaab dene ke liye {ANSWER_WAIT_SECONDS} second wait...")
    time.sleep(ANSWER_WAIT_SECONDS)


    # ========================================================
    # BROWSER MICROPHONE
    # ========================================================

    st.warning(
        "🎙️ Ab microphone icon par click karke "
        "apna answer bolna shuru karo."
    )

    audio_value = st.audio_input(
        "Candidate microphone",
        
        key=f"candidate_audio_{round_num}",
        label_visibility="collapsed"
    )


    # ========================================================
    # WAIT FOR RECORDING
    # ========================================================

    if audio_value is None:
        st.info("🎤 Microphone button dabao aur apna answer record karo.")
        st.stop()


    # ========================================================
    # PROCESS ANSWER ONLY ONCE
    # ========================================================

    if not st.session_state.answer_processed:

        st.session_state.answer_processed = True

        with st.spinner("🎧 Answer transcribe ho raha hai..."):
            try:
                answer_text = get_candidate_answer(audio_value)
            except TranscriptionError as e:
                st.error(f"⚠️ {e}")
                st.session_state.answer_processed = False
                st.stop()

        if not answer_text:
            st.error("Answer record nahi hua. Dobara try karo.")
            st.session_state.answer_processed = False
            st.stop()

        st.success(f"🗣️ Aapka jawaab: {answer_text}")


        # ====================================================
        # EVALUATION (direct against resume context, no reference answer)
        # ====================================================

        eval_prompt = evaluate_answer_prompt()

        with st.spinner("📊 Answer evaluate ho raha hai..."):
            try:
                result = evaluate_answer(
                    llm,
                    eval_prompt,
                    question,
                    st.session_state.resume_section,
                    answer_text
                )
            except LLMCallError as e:
                st.error(f"⚠️ {e}")
                st.info("Evaluation skip karke agla question try karo.")
                result = None

        if result is not None:
            final_score = (
                result.TechnicalCorrectness
                + result.completeness
                + result.relevance
                + result.depth
                + result.communication
            ) / 5

            st.markdown("### 📊 Evaluation")

            st.write(f"Technical Correctness: {result.TechnicalCorrectness}/10")
            st.write(f"Completeness: {result.completeness}/10")
            st.write(f"Relevance: {result.relevance}/10")
            st.write(f"Depth: {result.depth}/10")
            st.write(f"Communication: {result.communication}/10")
            st.write(f"**Final Score: {final_score:.1f}/10**")

            st.write("**Strengths:**")
            for point in result.strengths:
                st.write(f"- {point}")

            st.write("**Areas for Improvement:**")
            for point in result.improvements:
                st.write(f"- {point}")


        # ====================================================
        # SAVE CONVERSATION
        # ====================================================

        st.session_state.conversation_history.append((question, answer_text))

        st.session_state.question = None
        st.session_state.resume_section = None
        st.session_state.answer_processed = False
        st.session_state.round_num += 1


        # ====================================================
        # NEXT QUESTION
        # ====================================================

        if st.session_state.round_num < NUM_QUESTIONS:
            time.sleep(2)
            st.rerun()
        else:
            st.session_state.interview_finished = True
            cleanup_session(st.session_state.session_id)
            st.success("🎉 Interview khatam!")


# ============================================================
# MANUAL RESET (lets a user abandon mid-interview and clean up
# instead of leaving orphaned resume/audio files on disk)
# ============================================================

if st.session_state.get("started") and not st.session_state.get("interview_finished"):
    if st.button("❌ Interview cancel karo"):
        cleanup_session(st.session_state.session_id)
        for key in [
            "started", "conversation_history", "round_num", "question",
            "resume_section", "answer_processed", "interview_finished", "resume_path"
        ]:
            st.session_state.pop(key, None)
        st.rerun()
