#!/usr/bin/env python3
import argparse
import os
import sys
import time
import openai
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm

class EpubTranslator:
    def __init__(self):
        self.setup_openai()
        self.settings = self.parse_args()
        self.validate_input()

    def setup_openai(self):
        """Configura l'API di OpenAI"""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            sys.exit("❌ Errore: Imposta la variabile d'ambiente OPENAI_API_KEY")

    def parse_args(self):
        """Configura gli argomenti da riga di comando"""
        parser = argparse.ArgumentParser(description="Traduttore EPUB EN → IT")
        parser.add_argument("-i", "--input", required=True, help="EPUB da tradurre")
        parser.add_argument("-o", "--output", required=True, help="EPUB tradotto")
        parser.add_argument("--model", default="gpt-3.5-turbo", help="Modello OpenAI")
        parser.add_argument("--temp", type=float, default=0.3, help="Creatività traduzione (0-1)")
        parser.add_argument("--sleep", type=float, default=1.0, help="Pausa tra richieste API")
        return parser.parse_args()

    def validate_input(self):
        """Verifica che il file esista"""
        if not os.path.exists(self.settings.input):
            sys.exit(f"❌ File non trovato: {self.settings.input}")

    def translate_text(self, text: str) -> str:
        """Traduce un blocco di testo"""
        try:
            response = openai.ChatCompletion.create(
                model=self.settings.model,
                messages=[{
                    "role": "system",
                    "content": "Traduci letteralmente dall'inglese all'italiano mantenendo il formato esatto. Conserva tag HTML, numeri e codici."
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=self.settings.temp
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Errore traduzione: {e}")
            return text

    def process_epub(self):
        """Elabora il file EPUB"""
        try:
            book = epub.read_epub(self.settings.input)
            items = [item for item in book.get_items() if item.get_type() == epub.ET_DOCUMENT]

            print(f"\n📖 Inizio traduzione: {self.settings.input}")
            print(f"🔠 Modello: {self.settings.model} | Temp: {self.settings.temp}")
            print(f"⏳ Pausa tra richieste: {self.settings.sleep}s")
            print(f"📝 Capitoli da tradurre: {len(items)}\n")

            with tqdm(items, desc="Traduzione") as progress_bar:
                for item in progress_bar:
                    html = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(html, 'html.parser')

                    for element in soup.find_all(text=True):
                        if element.strip() and element.parent.name not in ['script', 'style']:
                            translated = self.translate_text(element)
                            element.replace_with(translated)

                    item.set_content(str(soup).encode('utf-8'))
                    time.sleep(self.settings.sleep)

            epub.write_epub(self.settings.output, book)
            print(f"\n✅ Traduzione completata: {self.settings.output}")

        except Exception as e:
            sys.exit(f"❌ Errore critico: {str(e)}")

if __name__ == "__main__":
    EpubTranslator().process_epub()
