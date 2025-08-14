#!/usr/bin/env python3
import argparse
import os
import sys
import time
import openai
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Dict

class EpubTranslatorPro:
    def __init__(self):
        self.setup_openai()
        self.settings = self.parse_args()
        self.validate_input()

    def setup_openai(self):
        """Configura l'API di OpenAI con controllo robusto"""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            sys.exit("""
❌ Errore: Imposta la variabile d'ambiente OPENAI_API_KEY
Guida:
1. Ottieni una chiave su: https://platform.openai.com/api-keys
2. Esportala con: export OPENAI_API_KEY='tua-chiave-here'
3. Verifica con: echo $OPENAI_API_KEY
""")

    def parse_args(self) -> argparse.Namespace:
        """Configurazione avanzata degli argomenti CLI"""
        parser = argparse.ArgumentParser(
            description="Traduttore Professionale EPUB EN → IT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("-i", "--input", required=True, 
                          help="Percorso del file EPUB da tradurre")
        parser.add_argument("-o", "--output", required=True,
                          help="Percorso del file EPUB tradotto")
        parser.add_argument("--model", default="gpt-4",
                          choices=["gpt-3.5-turbo", "gpt-4"],
                          help="Modello OpenAI da utilizzare")
        parser.add_argument("--temp", type=float, default=0.1,
                          help="Livello di creatività (0=letterale, 1=creativo)")
        parser.add_argument("--sleep", type=float, default=1.5,
                          help="Pausa tra richieste API (secondi)")
        parser.add_argument("--batch-size", type=int, default=3,
                          help="Blocchi di testo per richiesta API")
        parser.add_argument("--dry-run", action="store_true",
                          help="Analisi preliminare senza traduzione")
        parser.add_argument("--max-chars", type=int, default=1500,
                          help="Lunghezza massima per blocco di testo")
        return parser.parse_args()

    def validate_input(self):
        """Validazione avanzata dell'input"""
        if not os.path.exists(self.settings.input):
            sys.exit(f"❌ File non trovato: {self.settings.input}")
        
        if not self.settings.input.lower().endswith('.epub'):
            sys.exit("❌ Il file di input deve avere estensione .epub")

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Traduzione batch con gestione errori robusta"""
        try:
            response = openai.ChatCompletion.create(
                model=self.settings.model,
                messages=[{
                    "role": "system",
                    "content": """Traduci professionalmente dall'inglese all'italiano mantenendo:
1. Terminologia precisa
2. Struttura originale
3. Stile formale
4. Tag HTML intatti
5. Numerazioni e codici invariati"""
                }] + [{"role": "user", "content": text} for text in texts],
                temperature=self.settings.temp,
                request_timeout=30  # Timeout aumentato
            )
            return [choice.message.content for choice in response.choices]
        
        except openai.error.RateLimitError:
            print("\n⚠️ Rate limit raggiunto, attesa 20 secondi...")
            time.sleep(20)
            return self.translate_batch(texts)
        
        except Exception as e:
            print(f"\n⚠️ Errore batch: {str(e)}")
            return texts  # Fallback al testo originale

    def analyze_epub(self, book: epub.EpubBook) -> Dict:
        """Analisi dettagliata per dry-run"""
        items = [item for item in book.get_items() if item.get_type() == epub.ET_DOCUMENT]
        stats = {
            'total_chapters': len(items),
            'text_blocks': 0,
            'sample_texts': [],
            'estimated_time': 0,
            'estimated_cost': 0
        }

        print("\n🔍 ANALISI PRELIMINARE IN CORSO...\n")
        
        with tqdm(items, desc="Scansione capitoli") as pbar:
            for item in pbar:
                soup = BeautifulSoup(item.get_content().decode('utf-8', errors='replace'), 'html.parser')
                text_elements = [t for t in soup.find_all(text=True) 
                               if t.strip() and t.parent.name not in ['script', 'style']]
                
                stats['text_blocks'] += len(text_elements)
                
                # Campionatura testuale
                if len(stats['sample_texts']) < 3 and text_elements:
                    sample = text_elements[0].strip()[:200] + "..." if len(text_elements[0]) > 200 else text_elements[0]
                    stats['sample_texts'].append(sample)

        # Calcolo stime
        stats['estimated_time'] = (stats['text_blocks'] / self.settings.batch_size * self.settings.sleep) / 60
        stats['estimated_cost'] = (stats['text_blocks'] * 0.002) if self.settings.model == "gpt-4" else (stats['text_blocks'] * 0.0005)
        
        return stats

    def process_epub(self):
        """Elaborazione principale con modalità dry-run/effettiva"""
        try:
            book = epub.read_epub(self.settings.input)
            
            if self.settings.dry_run:
                stats = self.analyze_epub(book)
                
                print("\n📊 RISULTATI ANALISI PRELIMINARE:")
                print(f"- 📖 Capitoli totali: {stats['total_chapters']}")
                print(f"- 📝 Blocchi di testo: {stats['text_blocks']}")
                print(f"- ⏳ Tempo stimato: {stats['estimated_time']:.1f} minuti")
                print(f"- 💰 Costo stimato: ${stats['estimated_cost']:.2f}")
                
                print("\n🔠 CAMPIONI TESTO (prime righe):")
                for i, sample in enumerate(stats['sample_texts'], 1):
                    print(f"\n【Esempio {i}】\n{sample}")
                
                print("\n✅ DRY RUN COMPLETATO. Verificare i campioni prima della traduzione effettiva.")
                return
            
            # Modalità traduzione effettiva
            print("\n🚀 AVVIO TRADUZIONE EFFETTIVA\n")
            items = [item for item in book.get_items() if item.get_type() == epub.ET_DOCUMENT]
            
            with tqdm(items, desc="Traduzione") as pbar:
                for item in pbar:
                    soup = BeautifulSoup(item.get_content().decode('utf-8', errors='replace'), 'html.parser')
                    text_elements = [t for t in soup.find_all(text=True) 
                                  if t.strip() and t.parent.name not in ['script', 'style']]
                    
                    # Processamento a batch
                    for i in range(0, len(text_elements), self.settings.batch_size):
                        batch = text_elements[i:i + self.settings.batch_size]
                        translated = self.translate_batch([str(t) for t in batch])
                        
                        for orig, new in zip(batch, translated):
                            orig.replace_with(new)
                        
                        time.sleep(self.settings.sleep)
                    
                    item.set_content(str(soup).encode('utf-8'))

            epub.write_epub(self.settings.output, book)
            print(f"\n✅ TRADUZIONE COMPLETATA: {self.settings.output}")

        except Exception as e:
            sys.exit(f"\n❌ ERRORE CRITICO: {str(e)}")

if __name__ == "__main__":
    EpubTranslatorPro().process_epub()
