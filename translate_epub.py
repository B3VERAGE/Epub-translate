#!/usr/bin/env python3
import argparse
import os
import sys
import time
import openai
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List

class Settings:
    def __init__(self, input_path, output_path, source_lang, target_lang, 
                 model, temperature, sleep_time, batch_size, dry_run):
        self.input_path = input_path
        self.output_path = output_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        self.temperature = temperature
        self.sleep_between_docs = sleep_time
        self.batch_size = batch_size
        self.dry_run = dry_run

def translate_batch(texts: List[str], settings: Settings) -> List[str]:
    """Traduce un batch di testi"""
    try:
        response = openai.ChatCompletion.create(
            model=settings.model,
            messages=[{
                "role": "system",
                "content": f"Traduci testi dal {settings.source_lang} al {settings.target_lang}, mantieni markup HTML."
            }] + [{"role": "user", "content": text} for text in texts],
            temperature=settings.temperature
        )
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Errore traduzione: {e}")
        return texts

def process_html_content(html: str, settings: Settings) -> str:
    """Processa e traduce contenuto HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    texts_to_translate = []
    elements = []

    for element in soup.find_all(text=True):
        if element.parent.name not in ['script', 'style', 'meta', 'head'] and element.strip():
            texts_to_translate.append(element)
            elements.append(element)

    # Traduci in batch
    translated_texts = translate_batch([t for t in texts_to_translate], settings)

    for original, translated in zip(elements, translated_texts):
        original.replace_with(translated)

    return str(soup)

def translate_epub(settings: Settings):
    """Funzione principale"""
    try:
        book = epub.read_epub(settings.input_path)
        items = list(book.get_items_of_type(epub.ET_DOCUMENT))

        with tqdm(items, desc="Traduzione") as pb:
            for item in pb:
                content = item.get_content().decode('utf-8', errors='replace')
                new_content = process_html_content(content, settings)
                
                if not settings.dry_run:
                    item.set_content(new_content.encode('utf-8'))
                
                time.sleep(settings.sleep_between_docs)

        if not settings.dry_run:
            epub.write_epub(settings.output_path, book, {})

    except Exception as e:
        print(f"❌ Errore critico: {e}")
        sys.exit(1)

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        sys.exit("❌ Imposta OPENAI_API_KEY come variabile d'ambiente")

    args = parse_args()
    
    if not os.path.exists(args.input):
        sys.exit(f"❌ File non trovato: {args.input}")

    settings = Settings(
        args.input, args.output,
        args.source, args.target,
        args.model, args.temp,
        args.sleep_between_docs,
        args.batch_size, args.dry_run
    )

    translate_epub(settings)
    print(f"✅ Traduzione completata: {args.output}")

if __name__ == "__main__":
    main()
