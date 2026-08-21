# 🤖 InterviewIQ — AI Technical Interview Assessment Platform

InterviewIQ is an AI-powered technical interview platform that conducts **resume-based technical interviews** using a conversational question-and-answer flow.

The candidate uploads a resume, and InterviewIQ analyzes it to identify relevant projects, technologies, and technical details. The system generates technical interview questions based on the resume and previous answers, asks them using AI-generated speech, records the candidate's spoken response through the browser microphone, transcribes it, and evaluates it across multiple technical and communication criteria — all grounded in the candidate's actual resume.

---

## ✨ Key Features

- **Resume-Based Interview** — Resume PDF is parsed with `PyPDFLoader`, split into chunks, and used to ground every question.
- **Context-Aware Question Generation** — Considers resume context, previous questions, and previous answers. Can generate follow-up questions that dig into a specific technical detail from the last answer, and stays focused on one project at a time.
- **AI Interviewer Voice** — Questions are converted to speech using **Microsoft Edge TTS**, with word-boundary timings used to sync a word-by-word highlighted display with the audio.
- **Interactive AI Interviewer UI** — Streamlit interface with an animated audio waveform avatar and speaking status.
- **Browser-Based Answer Recording** — Candidates record answers via `st.audio_input()` at 16 kHz.
- **Speech-to-Text** — Answers are transcribed using **Groq Whisper**.
- **Structured Answer Evaluation** — Each answer is scored against the same resume context used to generate the question, on 5 criteria (Technical Correctness, Completeness, Relevance, Depth, Communication), each out of 10, enforced via a Pydantic schema. Also returns strengths and areas for improvement.
- **Resume Vectorstore Caching** — Resume is hashed with SHA-256; if a FAISS vectorstore already exists for that hash, it's loaded instead of rebuilt.
- **Automatic Cleanup** — On interview end/cancel, uploaded resume and question audio files are deleted. Stale vectorstore caches older than a configurable age are also cleaned up.
- **Session-Specific Storage** — Every interview gets a unique UUID, used in resume and audio filenames to avoid collisions between sessions.
- **Environment-Based Configuration** — Key settings are controlled via `.env` rather than hardcoded (see below).

---

## 🏗️ Architecture / Flow

```text
Candidate uploads resume (PDF)
        ↓
Saved with session-specific ID
        ↓
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
        ↓
HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
        ↓
FAISS vectorstore (created, or loaded from SHA-256 cache)
        ↓
Top-5 relevant chunks retrieved as resume context
        ↓
Groq LLM (Llama 3.3 70B) generates a context-aware question
        ↓
Edge TTS converts question to speech + word timings
        ↓
AI Interviewer UI plays audio with synced highlighting
        ↓
Candidate records answer via browser mic (st.audio_input, 16kHz)
        ↓
Groq Whisper transcribes the answer
        ↓
Groq LLM evaluates answer against the same resume context
        ↓
Scores (Technical Correctness, Completeness, Relevance, Depth,
Communication) + Strengths + Improvements displayed
        ↓
Question + answer saved to conversation history
        ↓
Next contextual question generated → repeats for NUM_QUESTIONS rounds
        ↓
Resume + question audio files deleted, session state reset
```

---

## 🧩 Project Structure

```text
InterviewIQ/
│
├── stream_main.py       # Streamlit UI + interview loop
├── interview_logic.py      # LLM, RAG, STT, TTS, evaluation logic
│
├── uploads/
│   └── <session_id>.pdf
│
├── Question_Audios/
│   └── <session_id>_question_<round>.mp3
│
├── vectorstore/
│   └── <resume_hash>/
│
├── model_cache/
│
├── .env
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category               | Technology                     |
| ----------------------- | ------------------------------ |
| Frontend / UI            | Streamlit                      |
| Language                | Python                         |
| LLM                     | Groq ChatGroq — Llama 3.3 70B Versatile |
| Speech-to-Text          | Groq Whisper                   |
| Text-to-Speech          | Edge TTS                       |
| Embeddings              | BAAI/bge-small-en-v1.5         |
| Vector Database          | FAISS                          |
| PDF Processing          | PyPDFLoader                    |
| Text Splitting          | RecursiveCharacterTextSplitter |
| Structured Output        | Pydantic                       |
| Prompting               | LangChain                      |
| Environment Management   | python-dotenv                  |

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/InterviewIQ.git
cd InterviewIQ

# create + activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key

GROQ_LLM_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3
EMBED_MODEL=BAAI/bge-small-en-v1.5
TTS_VOICE=en-US-AriaNeural
LLM_TEMPERATURE=0.5

NUM_QUESTIONS=5
MAX_RESUME_MB=10
ANSWER_WAIT_SECONDS=5

UPLOAD_DIR=uploads
AUDIO_DIR=Question_Audios
VECTORSTORE_DIR=vectorstore
VECTORSTORE_MAX_AGE_HOURS=24
```

**Never commit `.env` or API keys to GitHub.** Add to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
uploads/
Question_Audios/
vectorstore/
model_cache/
*.pyc
```

---

## ▶️ Running the Application

```bash
streamlit run stream_main.py
```

1. Upload your resume PDF.
2. Listen to the AI-generated question.
3. Record your answer using the microphone.
4. Wait for transcription and evaluation.
5. Review your scores and feedback.
6. Continue to the next question — interview ends after `NUM_QUESTIONS` rounds.

---

## 📊 Answer Evaluation

```text
Final Score = (Technical Correctness + Completeness + Relevance + Depth + Communication) / 5
```

Each criterion is scored /10, and the system also returns specific strengths and areas for improvement.

---

## 🛡️ Error Handling

Custom exceptions let the app handle failures independently instead of treating everything as a generic error:

```python
ResumeProcessingError   # bad/unreadable resume, vectorstore build/load failure
LLMCallError             # question generation / evaluation failure
TranscriptionError       # Groq Whisper failure
TTSError                 # Edge TTS failure — falls back to text-only question
```

---

## 🚀 Production-Oriented Features

- Environment-based configuration
- Logging instead of `print`
- Custom exception handling per failure type
- Session-specific file naming (UUID)
- Resume size validation
- Vectorstore caching (SHA-256) + stale cache cleanup
- Temporary file cleanup on finish/cancel
- Browser-based microphone recording
- TTS failure fallback (text-only question)
- Structured LLM evaluation via Pydantic
- Conversation history for contextual follow-up questions

---

## 🔮 Future Improvements

- Persistent user accounts and authentication
- Interview history dashboard, database-backed results
- Job-description-specific interviews (resume ↔ JD matching)
- Difficulty adaptation based on candidate performance
- Final interview report generation + email delivery
- Docker-based deployment, cloud storage
- Real-time interview analytics

---

## 👨‍💻 Author

**Anmakshi Singh**
B.Tech CSE (AI/ML), GBPIET — Class of 2027

---

**InterviewIQ — Turn your resume into an interactive technical interview.**
