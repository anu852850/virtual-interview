import logging
import os
import sys
import hashlib
import tempfile
import asyncio
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# LOGGING (replaces print statements)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("interview_logic")


# ============================================================
# CONFIG (previously hardcoded values)
# ============================================================

GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
VECTORSTORE_MAX_AGE_HOURS = int(os.getenv("VECTORSTORE_MAX_AGE_HOURS", "24"))


# ============================================================
# CUSTOM EXCEPTIONS (so app.py can catch specific failures)
# ============================================================

class ResumeProcessingError(Exception):
    """Raised when resume loading / embedding / retrieval fails."""


class LLMCallError(Exception):
    """Raised when a Groq LLM call fails."""


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


class TTSError(Exception):
    """Raised when text-to-speech generation fails."""


# ============================================================
# RESUME LOADING
# ============================================================

def load_resume(path):

    if path is None:
        logger.error("load_resume called with path=None")
        raise ResumeProcessingError("Resume path was not provided.")

    try:
        path = os.path.abspath(path)
        loader = PyPDFLoader(path)
        pages = loader.load()
    except Exception as e:
        logger.exception("Failed to load resume PDF at %s", path)
        raise ResumeProcessingError(f"Could not read the resume PDF: {e}") from e

    if not pages:
        raise ResumeProcessingError("Resume PDF appears to be empty or unreadable.")

    return pages


# ============================================================
# HASHING (used as cache key for the vectorstore)
# ============================================================

BUF_SIZE = 65536


def hash_store(path):
    s25 = hashlib.sha256()
    try:
        with open(path, 'rb') as file:
            while True:
                data = file.read(BUF_SIZE)
                if not data:
                    break
                s25.update(data)
    except OSError as e:
        logger.exception("Failed to hash file at %s", path)
        raise ResumeProcessingError(f"Could not read uploaded file: {e}") from e

    return s25.hexdigest()


# ============================================================
# VECTORSTORE CACHE (existence check + creation)
# ============================================================

def check_exist(path):
    file_hash = hash_store(path)
    cache_path = os.path.join(VECTORSTORE_DIR, file_hash)

    try:
        if os.path.exists(cache_path):
            vectorstore = FAISS.load_local(
                cache_path,
                embed,
                allow_dangerous_deserialization=True
            )
        else:
            resume = load_resume(path)
            chunks = splitter(resume)
            vectorstore = create_faiss(chunks, embed)
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vectorstore.save_local(cache_path)
    except ResumeProcessingError:
        raise
    except Exception as e:
        logger.exception("Failed to build/load vectorstore for %s", path)
        raise ResumeProcessingError(f"Could not process resume into a vector index: {e}") from e

    return vectorstore


def cleanup_old_vectorstores(max_age_hours=VECTORSTORE_MAX_AGE_HOURS):
    """Deletes cached vectorstore folders older than max_age_hours.
    Call this periodically (e.g. on app startup or via a cron/scheduled task)
    so disk usage doesn't grow unbounded as new resumes get uploaded."""
    import shutil
    import time

    if not os.path.exists(VECTORSTORE_DIR):
        return

    cutoff = time.time() - (max_age_hours * 3600)

    for entry in os.listdir(VECTORSTORE_DIR):
        entry_path = os.path.join(VECTORSTORE_DIR, entry)
        try:
            if os.path.isdir(entry_path) and os.path.getmtime(entry_path) < cutoff:
                shutil.rmtree(entry_path)
                logger.info("Removed stale vectorstore cache: %s", entry_path)
        except OSError:
            logger.exception("Failed to remove stale vectorstore cache: %s", entry_path)


# ============================================================
# TEXT SPLITTER
# ============================================================

def splitter(page):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[" ", ":", "-", "_"]
    )
    return text_splitter.split_documents(page)


# ============================================================
# LLM
# ============================================================

def setup_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMCallError("GROQ_API_KEY is not set in the environment.")

    return ChatGroq(
        model=GROQ_LLM_MODEL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def set_embed():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder="./model_cache"
    )


# ============================================================
# FAISS
# ============================================================

def create_faiss(chunks, emb):
    return FAISS.from_documents(chunks, emb)


# ============================================================
# INTERVIEW QUESTION PROMPT
# ============================================================

def interview_question_prompt():

    template = """<|system|>

You are an experienced technical interviewer conducting a job interview.

1. If there is no previous conversation, ask a technical question based on the resume context.

2. If there is a previous question and answer, ask a follow-up question that digs deeper into something specific the candidate mentioned in their last answer.

3. Do not ask generic questions like "tell me about yourself".

4. Keep the question concise (1-2 sentences).

5. Use simple, plain, easy-to-understand English. Avoid complex vocabulary, jargon-heavy phrasing, or overly formal sentence structures. Write as if explaining to someone comfortable with everyday spoken English, not academic writing.

6. The Resume Context below may contain fragments from more than one project. Pick ONE project only and ask your question about that project alone. Do NOT combine tools, metrics, or concepts from two different projects into a single question.

7. Output only the question, no explanation.

</s>

<|user|>

Resume Context:
{resume_section}

Conversation so far:
{conversation_history}

</s>

<|assistant|>
"""

    return PromptTemplate(
        input_variables=["resume_section", "conversation_history"],
        template=template
    )


def interview_question_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)


# ============================================================
# ONE TIME SETUP
# ============================================================

try:
    llm = setup_llm()
    embed = set_embed()
    prompt = interview_question_prompt()
except Exception:
    logger.exception("Startup failed while initializing LLM/embeddings.")
    raise


# ============================================================
# GET RESUME CONTEXT (retrieval only, reused for question + eval)
# ============================================================

def get_resume_context(path):

    if path is None:
        raise ResumeProcessingError("Resume path was not provided.")

    vectorstore = check_exist(path)

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(
            "one specific project: its tools, what was built, and key metrics"
        )
    except Exception as e:
        logger.exception("Retrieval failed for %s", path)
        raise ResumeProcessingError(f"Could not retrieve resume context: {e}") from e

    return "\n".join(doc.page_content for doc in relevant_docs)


# ============================================================
# ASK INTERVIEW QUESTION
# ============================================================

def ask_interview_question(path, conversation_history):

    if path is None:
        raise ResumeProcessingError("Resume path was not provided.")

    resume_section = get_resume_context(path)

    history_text = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in conversation_history
    )

    chain = interview_question_chain(llm=llm, prompt=prompt)

    try:
        response = chain.invoke({
            "resume_section": resume_section,
            "conversation_history": history_text
        })
        question = response["text"]
    except Exception as e:
        logger.exception("LLM call failed while generating interview question")
        raise LLMCallError(f"Could not generate a question right now: {e}") from e

    return question, resume_section


# ============================================================
# BROWSER MICROPHONE -> GROQ WHISPER
# ============================================================

from groq import Groq


def transcribe_audio(audio_bytes, filename="answer.wav"):
    """Receives audio recorded by the candidate's browser (from st.audio_input())."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise TranscriptionError("GROQ_API_KEY is not set in the environment.")

    client = Groq(api_key=api_key)

    try:
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_WHISPER_MODEL,
            language="en"
        )
    except Exception as e:
        logger.exception("Transcription failed for %s", filename)
        raise TranscriptionError(f"Could not transcribe audio: {e}") from e

    return transcription.text


def get_candidate_answer(audio_value):

    if audio_value is None:
        return None

    audio_bytes = audio_value.getvalue()

    logger.info(
        "Received candidate audio: name=%s type=%s size=%d bytes",
        audio_value.name, audio_value.type, len(audio_bytes)
    )

    answer_text = transcribe_audio(audio_bytes, filename=audio_value.name)

    logger.info("Transcribed candidate answer (%d chars)", len(answer_text or ""))

    return answer_text


# ============================================================
# TEXT TO SPEECH
# ============================================================

import edge_tts


async def generate_speech_with_timings(text, filename):

    communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE)

    word_timings = []
    first_chunk = True

    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mode = "wb" if first_chunk else "ab"
                with open(filename, mode) as f:
                    f.write(chunk["data"])
                first_chunk = False
            elif chunk["type"] == "WordBoundary":
                word_timings.append({
                    "word": chunk["text"],
                    "offset": chunk["offset"] / 10_000_000
                })
    except Exception as e:
        logger.exception("TTS generation failed for filename=%s", filename)
        raise TTSError(f"Could not generate question audio: {e}") from e

    return filename, word_timings


def speak_question(text, filename="question.mp3"):
    try:
        return asyncio.run(generate_speech_with_timings(text, filename))
    except TTSError:
        raise
    except Exception as e:
        logger.exception("speak_question failed")
        raise TTSError(f"Could not generate question audio: {e}") from e


# ============================================================
# EVALUATION SCHEMA
# ============================================================

class Evaluate(BaseModel):
    TechnicalCorrectness: int
    relevance: int
    depth: int
    communication: int
    strengths: list[str]
    completeness: int
    improvements: list[str]


# ============================================================
# EVALUATION PROMPT (resume-grounded, no separate reference answer)
# ============================================================

def evaluate_answer_prompt():

    template = """<|system|>

You are a technical interviewer evaluating a candidate's spoken answer.

You are given the resume context (the actual project/skills the question was based on) and the candidate's actual answer.

1. Judge the candidate's answer against what the resume context supports, for technical correctness, relevance, depth, communication and completeness.

2. Give a score out of 10 for each of the five criteria above (technical correctness, completeness, relevance, depth, communication).
3. List 2-3 specific strengths and 2-3 specific areas for improvement.

</s>

<|user|>

Question:
{question}

Resume Context:
{resume_section}

Candidate Answer:
{candidate_answer}

</s>

<|assistant|>
"""

    return PromptTemplate(
        input_variables=["question", "resume_section", "candidate_answer"],
        template=template
    )


def evaluate_answer_chain(llm, prompt):
    structured_llm = llm.with_structured_output(Evaluate)
    return prompt | structured_llm


def evaluate_answer(llm, prompt, question, resume_section, candidate_answer):

    chain = evaluate_answer_chain(llm=llm, prompt=prompt)

    try:
        result = chain.invoke({
            "question": question,
            "resume_section": resume_section,
            "candidate_answer": candidate_answer
        })
    except Exception as e:
        logger.exception("LLM call failed while evaluating answer")
        raise LLMCallError(f"Could not evaluate the answer right now: {e}") from e

    logger.info("Evaluation result: %s", result)

    return result


# ============================================================
# JOB DESCRIPTION / BM25
# ============================================================

def load_jd(text, resume_chunks, faiss_index):
    """Matches JD requirement chunks against resume chunks via BM25 + FAISS ensemble.
    Uses a per-call temp file instead of a shared 'text.txt' to avoid races
    between concurrent users/sessions."""

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        loader = TextLoader(tmp_path)
        documents = loader.load()
        chunks_jd = splitter(documents)

        bm25_retriever = BM25Retriever.from_documents(resume_chunks)
        bm25_retriever.k = 3

        faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        results = {}
        for requirement in chunks_jd:
            matches = ensemble_retriever.invoke(requirement.page_content)
            results[requirement.page_content] = matches

    except Exception as e:
        logger.exception("load_jd failed")
        raise ResumeProcessingError(f"Could not match job description against resume: {e}") from e
    finally:
        try:
            os.remove(tmp_path)
        except (OSError, NameError):
            pass

    return results