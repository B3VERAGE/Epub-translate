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
