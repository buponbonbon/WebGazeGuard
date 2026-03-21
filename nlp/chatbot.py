from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import tkinter as tk
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tkinter import font

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector.index"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

PDF_CANDIDATES = [
    BASE_DIR / "text.pdf",
    BASE_DIR / "pdf.pdf",
    BASE_DIR / "report.pdf",
]

TOP1_CONF_THRESHOLD = float(os.getenv("RAG_TOP1_CONF_THRESHOLD", "0.30"))
MEAN_CONF_THRESHOLD = float(os.getenv("RAG_MEAN_CONF_THRESHOLD", "0.22"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

load_dotenv(BASE_DIR / ".env")
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None
model_embed = SentenceTransformer("all-MiniLM-L6-v2")


def resolve_pdf_path() -> Path:
    env_pdf = os.getenv("EYE_HEALTH_PDF", "").strip()
    if env_pdf:
        p = Path(env_pdf)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        if p.exists():
            return p

    for p in PDF_CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Không tìm thấy file PDF. Hãy đặt text.pdf trong thư mục nlp/ "
        "hoặc khai báo EYE_HEALTH_PDF trong file .env."
    )


PDF_PATH = resolve_pdf_path()


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> np.ndarray:
    vector = model_embed.encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype="float32")


def build_index(pdf_path: Path):
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("PDF không trích xuất được nội dung.")

    embeddings = np.vstack([embed_text(c) for c in chunks]).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query: str, index, chunks: List[str], k: int = TOP_K):
    qv = np.array([embed_text(query)], dtype="float32")
    scores, indices = index.search(qv, min(k, len(chunks)))

    contexts = []
    scored_contexts = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        contexts.append(chunks[idx])
        scored_contexts.append((chunks[idx], float(score)))

    return contexts, scored_contexts


def confidence_filter(scored_contexts: List[Tuple[str, float]]):
    if not scored_contexts:
        return False, 0.0, 0.0

    scores = [score for _, score in scored_contexts]
    top1 = scores[0]
    mean_score = float(np.mean(scores))
    passed = (top1 >= TOP1_CONF_THRESHOLD) and (mean_score >= MEAN_CONF_THRESHOLD)
    return passed, float(top1), mean_score


def ask_rag(question: str, index, chunks: List[str]) -> str:
    if client is None:
        return (
            "⚠️ Chưa có GEMINI_API_KEY. "
            "Hãy tạo file .env trong thư mục nlp và thêm dòng:\n"
            "GEMINI_API_KEY=your_api_key"
        )

    contexts, scored_contexts = retrieve(question, index, chunks, k=TOP_K)
    passed, top1_conf, mean_conf = confidence_filter(scored_contexts)

    if not passed:
        return (
            "Tôi chưa đủ tự tin để trả lời dựa trên tài liệu hiện có. "
            "Bạn hãy hỏi cụ thể hơn về sức khỏe mắt, hoặc kiểm tra lại tài liệu nguồn.\n\n"
            f"(top1_conf={top1_conf:.2f}, mean_conf={mean_conf:.2f})"
        )

    context_text = "\n\n".join(contexts)
    prompt = f"""
Bạn là một chuyên gia tư vấn sức khỏe mắt. Dưới đây là thông tin chuyên môn (Context) bằng tiếng Anh.
Hãy dựa vào đó để trả lời câu hỏi của người dùng bằng TIẾNG VIỆT.

Yêu cầu:
1. Chỉ trả lời dựa trên Context được cung cấp.
2. Trả lời chính xác, dễ hiểu, thân thiện.
3. Nếu Context không đủ để trả lời, hãy nói: "Tôi không tìm thấy thông tin này trong tài liệu".
4. Nếu có thuật ngữ chuyên môn phức tạp, hãy ghi chú thuật ngữ tiếng Anh trong ngoặc đơn.
5. LUÔN LUÔN kèm theo lời khuyên: "Đây chỉ là thông tin tham khảo, bạn nên đến gặp bác sĩ chuyên khoa mắt để được thăm khám chính xác".

Retrieval confidence:
- top1_conf = {top1_conf:.2f}
- mean_conf = {mean_conf:.2f}

Context (English):
{context_text}

Question (Vietnamese):
{question}

Answer (Vietnamese):
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


BG = "#0f1117"
SIDEBAR = "#1a1d27"
BUBBLE_AI = "#1e2235"
BUBBLE_ME = "#2563eb"
ACCENT = "#3b82f6"
TEXT = "#e2e8f0"
MUTED = "#64748b"
INPUT_BG = "#1e2235"
BORDER = "#2d3148"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"


class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("👁 Eye Health Assistant")
        self.geometry("900x680")
        self.minsize(700, 500)
        self.configure(bg=BG)
        self.index = None
        self.chunks = None

        self._build_fonts()
        self._build_layout()
        self._init_rag()

    def _build_fonts(self):
        self.f_title = font.Font(family="Segoe UI", size=13, weight="bold")
        self.f_body = font.Font(family="Segoe UI", size=10)
        self.f_body_b = font.Font(family="Segoe UI", size=10, weight="bold")
        self.f_small = font.Font(family="Segoe UI", size=8)
        self.f_input = font.Font(family="Segoe UI", size=11)

    def _build_layout(self):
        sidebar = tk.Frame(self, bg=SIDEBAR, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="👁", bg=SIDEBAR, fg=ACCENT,
                 font=("Segoe UI", 32)).pack(pady=(30, 4))
        tk.Label(sidebar, text="Eye Health\nAssistant", bg=SIDEBAR, fg=TEXT,
                 font=self.f_title, justify="center").pack()
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20, pady=20)

        self.status_dot = tk.Label(sidebar, text="●", bg=SIDEBAR, fg=WARNING,
                                   font=("Segoe UI", 10))
        self.status_dot.pack()
        self.status_lbl = tk.Label(
            sidebar,
            text="Đang tải mô hình…",
            bg=SIDEBAR,
            fg=MUTED,
            font=self.f_small,
            wraplength=180,
            justify="center",
        )
        self.status_lbl.pack(pady=(2, 20))

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20)

        tk.Button(
            sidebar, text="🗑 Xóa lịch sử",
            bg=SIDEBAR, fg=MUTED, relief="flat",
            font=self.f_small, cursor="hand2",
            activebackground=BORDER, activeforeground=TEXT,
            command=self._clear_chat,
        ).pack(side="bottom", pady=20)

        main = tk.Frame(self, bg=BG)
        main.pack(side="right", fill="both", expand=True)

        hdr = tk.Frame(main, bg=SIDEBAR, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Chat", bg=SIDEBAR, fg=TEXT,
                 font=self.f_title).pack(side="left", padx=20, pady=14)

        canvas_frame = tk.Frame(main, bg=BG)
        canvas_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self.canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.chat_frame = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")

        self.chat_frame.bind("<Configure>", self._on_frame_config)
        self.canvas.bind("<Configure>", self._on_canvas_config)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        bar = tk.Frame(main, bg=SIDEBAR, pady=12, padx=16)
        bar.pack(fill="x")

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(
            bar,
            textvariable=self.input_var,
            font=self.f_input,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=8,
        )
        self.entry.pack(side="left", fill="both", expand=True, ipady=6)
        self.entry.bind("<Return>", self._on_send)
        self.entry.insert(0, "Hỏi về sức khỏe mắt…")
        self.entry.config(fg=MUTED)
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<FocusOut>", self._restore_placeholder)

        self.send_btn = tk.Button(
            bar,
            text="Gửi ➤",
            font=self.f_body_b,
            bg=ACCENT,
            fg="white",
            relief="flat",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            activebackground="#1d4ed8",
            activeforeground="white",
            command=self._on_send,
            state="disabled",
        )
        self.send_btn.pack(side="left", padx=(10, 0))

    def _init_rag(self):
        def _load():
            try:
                if INDEX_PATH.exists() and CHUNKS_PATH.exists():
                    self.index, self.chunks = load_index()
                else:
                    self.index, self.chunks = build_index(PDF_PATH)
                self.after(0, self._on_ready)
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_ready(self):
        self.status_dot.config(fg=SUCCESS)
        self.status_lbl.config(text=f"Sẵn sàng! PDF: {PDF_PATH.name}")
        self.send_btn.config(state="normal")
        self._add_bubble(
            "assistant",
            "Xin chào! Tôi là trợ lý tư vấn sức khỏe mắt. "
            "Tôi sẽ trả lời dựa trên tài liệu trong thư mục nlp và có lọc độ tin cậy truy xuất.",
        )

    def _on_error(self, msg):
        self.status_dot.config(fg="#ef4444")
        self.status_lbl.config(text=f"Lỗi: {msg[:120]}")

    def _on_send(self, event=None):
        text = self.input_var.get().strip()
        if not text or text == "Hỏi về sức khỏe mắt…":
            return
        if self.index is None:
            return

        self.input_var.set("")
        self.send_btn.config(state="disabled")
        self._add_bubble("user", text)
        self._add_bubble("typing", "…")

        def _query():
            try:
                answer = ask_rag(text, self.index, self.chunks)
            except Exception as e:
                answer = f"⚠️ Lỗi: {e}"
            self.after(0, lambda: self._replace_typing(answer))

        threading.Thread(target=_query, daemon=True).start()

    def _add_bubble(self, role, text):
        wrapper = tk.Frame(self.chat_frame, bg=BG)
        wrapper.pack(fill="x", padx=16, pady=6)

        if role == "user":
            bubble = tk.Frame(wrapper, bg=BUBBLE_ME, padx=14, pady=10)
            bubble.pack(side="right")
            tk.Label(
                bubble, text=text, bg=BUBBLE_ME, fg="white",
                font=self.f_body, wraplength=480, justify="left", anchor="w",
            ).pack()
            self._typing_lbl = None

        elif role == "assistant":
            avatar = tk.Label(wrapper, text="👁", bg=BG, font=("Segoe UI", 14))
            avatar.pack(side="left", anchor="n", padx=(0, 8))
            bubble = tk.Frame(wrapper, bg=BUBBLE_AI, padx=14, pady=10)
            bubble.pack(side="left")
            tk.Label(
                bubble, text=text, bg=BUBBLE_AI, fg=TEXT,
                font=self.f_body, wraplength=520, justify="left", anchor="w",
            ).pack()
            self._typing_lbl = None

        elif role == "typing":
            avatar = tk.Label(wrapper, text="👁", bg=BG, font=("Segoe UI", 14))
            avatar.pack(side="left", anchor="n", padx=(0, 8))
            bubble = tk.Frame(wrapper, bg=BUBBLE_AI, padx=14, pady=10)
            bubble.pack(side="left")
            lbl = tk.Label(
                bubble, text=text, bg=BUBBLE_AI, fg=MUTED,
                font=self.f_body, wraplength=520,
            )
            lbl.pack()
            self._typing_wrapper = wrapper
            self._typing_lbl = lbl

        self._scroll_bottom()

    def _replace_typing(self, answer):
        if hasattr(self, "_typing_wrapper") and self._typing_wrapper:
            self._typing_wrapper.destroy()
            self._typing_wrapper = None
        self._add_bubble("assistant", answer)
        self.send_btn.config(state="normal")

    def _clear_chat(self):
        for w in self.chat_frame.winfo_children():
            w.destroy()

    def _on_frame_config(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_config(self, event=None):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_bottom(self):
        self.chat_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)

    def _clear_placeholder(self, event):
        if self.entry.get() == "Hỏi về sức khỏe mắt…":
            self.entry.delete(0, "end")
            self.entry.config(fg=TEXT)

    def _restore_placeholder(self, event):
        if not self.entry.get():
            self.entry.insert(0, "Hỏi về sức khỏe mắt…")
            self.entry.config(fg=MUTED)


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
