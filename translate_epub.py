from __future__ import annotations
import argparse, os, sys, time
from dataclasses import dataclass
from typing import Optional, List
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub, ITEM_DOCUMENT
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BLOCK_TAGS = {"p","div","li","blockquote","section","article","aside","header","footer","nav","main",
              "figure","figcaption","h1","h2","h3","h4","h5","h6","dd","dt","dl","table","thead","tbody",
              "tfoot","tr","td","th","caption","pre"}
CODE_LIKE_TAGS = {"code","pre","samp","kbd","var"}

SYSTEM_PROMPT = """You are an expert HTML translator.
Rules:
- Translate ONLY visible text from '{src}' to '{tgt}'.
- KEEP tags, attributes, href, src, ids, classes EXACTLY the same.
- DO NOT add/remove/reorder tags or attributes.
- Skip translation inside: code, pre, samp, kbd, var.
- Return HTML with SAME top-level tag and attributes as input.
"""

USER_PROMPT = """Translate the following HTML block, respecting rules:
@dataclass
class Settings:
    input_path: str
    output_path: str
    src_lang: str
    tgt_lang: str
    model: str
    temperature: float
    sleep_between_docs: float
    batch_size: int
    dry_run: bool

def _iter_blocks(body: Tag):
    for el in body.descendants:
        if isinstance(el, Tag) and el.name in BLOCK_TAGS:
            yield el

def _extract_outer(el: Tag) -> str:
    return str(el)

def _extract_inner(el: Tag) -> str:
    return "".join(str(c) for c in el.contents)

def _replace_inner(el: Tag, html: str):
    el.clear()
    parsed = BeautifulSoup(html, "lxml")
    for c in parsed.body.contents:
        el.append(c)

def _same_tag_and_attrs(orig: Tag, new: Tag) -> bool:
    return (orig.name == new.name) and (orig.attrs == new.attrs)

class Translator:
    def __init__(self, model: str, temp: float):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temp = temp

    @retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1,max=10),
           retry=retry_if_exception_type(Exception))
    def translate_html(self, html: str, src: str, tgt: str) -> str:
        if not html.strip(): return html
        sysmsg = SYSTEM_PROMPT.format(src=src, tgt=tgt)
        usrmsg = USER_PROMPT.format(html=html)
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            messages=[{"role":"system","content":sysmsg},
                      {"role":"user","content":usrmsg}]
        )
        out = r.choices[0].message.content.strip()
        if out.startswith("```"):
            out = "\n".join(out.splitlines()[1:-1]).strip()
        return out

def _translate_text_nodes(el: Tag, translator: Translator, src: str, tgt: str):
    for node in el.descendants:
        if isinstance(node, Tag) and node.name in CODE_LIKE_TAGS:
            return
    for node in list(el.descendants):
        if isinstance(node, NavigableString) and node.strip():
            frag = f"<span>{node}</span>"
            out = translator.translate_html(frag, src, tgt)
            parsed = BeautifulSoup(out, "lxml")
            span = parsed.find("span")
            node.replace_with(span.get_text() if span else parsed.get_text())

def translate_doc(html: bytes, tr: Translator, src: str, tgt: str, st: Settings, pb: Optional[tqdm]=None) -> bytes:
    soup = BeautifulSoup(html, "lxml")
    body = soup.body
    if not body: return html
    blocks = list(_iter_blocks(body))
    if pb: pb.reset(total=len(blocks))

    i = 0
    while i < len(blocks):
        batch = blocks[i:i+st.batch_size] if st.batch_size > 1 else [blocks[i]]
        if st.batch_size > 1:
            batch_html = "\n".join(_extract_outer(el) for el in batch)
            translated_batch = tr.translate_html(batch_html, src, tgt)
            translated_parts = translated_batch.split("\n")
            mapping = {j: translated_parts[j] if j < len(translated_parts) else None for j in range(len(batch))}
            for idx_in_batch, el in enumerate(batch):
                orig_html = _extract_outer(el)
                translated = mapping.get(idx_in_batch)
                if not translated:
                    _translate_text_nodes(el, tr, src, tgt)
                    if pb: pb.update(1)
                    continue
                parsed = BeautifulSoup(translated, "lxml")
                top = next((c for c in parsed.body.contents if isinstance(c, Tag)), None)
                if not top or not _same_tag_and_attrs(el, top):
                    print(f"[WARN] Block mismatch in batch at index {idx_in_batch}, retrying single mode...")
                    try:
                        out_html = tr.translate_html(orig_html, src, tgt)
                        parsed_retry = BeautifulSoup(out_html, "lxml")
                        top_retry = next((c for c in parsed_retry.body.contents if isinstance(c, Tag)), None)
                        if top_retry and _same_tag_and_attrs(el, top_retry):
                            _replace_inner(el, _extract_inner(top_retry))
                        else:
                            _translate_text_nodes(el, tr, src, tgt)
                    except Exception as e:
                        print(f"[ERROR] Fallback single translation failed: {e}")
                        _translate_text_nodes(el, tr, src, tgt)
                else:
                    _replace_inner(el, _extract_inner(top))
                if pb: pb.update(1)
            i += st.batch_size
        else:
            el = batch[0]
            orig_html = _extract_outer(el)
            out_html = tr.translate_html(orig_html, src, tgt)
            parsed = BeautifulSoup(out_html, "lxml")
            top = next((c for c in parsed.body.contents if isinstance(c, Tag)), None)
            if top and _same_tag_and_attrs(el, top):
                _replace_inner(el, _extract_inner(top))
            else:
                _translate_text_nodes(el, tr, src, tgt)
            if pb: pb.update(1)
            i += 1

    return str(soup).encode("utf-8")

def translate_epub(st: Settings):
    book = epub.read_epub(st.input_path)
    tr = Translator(st.model, st.temperature)
    docs = [it for it in book.get_items() if it.get_type() == ITEM_DOCUMENT]
    for idx, item in enumerate(docs, start=1):
        pb = tqdm(desc=f"Doc {idx}/{len(docs)}", leave=False)
        new_html = translate_doc(item.get_content(), tr, st.src_lang, st.tgt_lang, st, pb)
        pb.close()
        if not st.dry_run:
            item.set_content(new_html)
        if st.sleep_between_docs: time.sleep(st.sleep_between_docs)
    if not st.dry_run:
        epub.write_epub(st.output_path, book)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input",required=True)
    ap.add_argument("-o","--output",required=True)
    ap.add_argument("--source",default="en")
    ap.add_argument("--target",default="it")
    ap.add_argument("--model",default="gpt-4o-mini")
    ap.add_argument("--temp",type=float,default=0.2)
    ap.add_argument("--sleep-between-docs",type=float,default=0.0)
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--dry-run",action="store_true")
    a = ap.parse_args()
    return Settings(a.input,a.output,a.source,a.target,a.model,a.temp,a.sleep_between_docs,a.batch_size,a.dry_run)

if __name__ == "__main__":
    st = parse_args()
    if not os.path.exists(st.input_path):
        sys.exit(f"File not found: {st.input_path}")
    translate_epub(st)
    print(f"✅ Traduzione completata: {st.output_path}")
