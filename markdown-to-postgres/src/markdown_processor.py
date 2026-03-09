import os
import glob
import markdown
from bs4 import BeautifulSoup # Optional: for better text extraction
import logging
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is available (run once: python -m nltk.downloader punkt)
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.warning("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
except LookupError:
     logging.warning("NLTK 'punkt' tokenizer not found. Downloading...")
     nltk.download('punkt', quiet=True)


def find_markdown_files(directory: str) -> list[str]:
    """Finds all markdown files recursively in a directory."""
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return []
    pattern = os.path.join(directory, '**', '*.md')
    files = glob.glob(pattern, recursive=True)
    logging.info(f"Found {len(files)} markdown files in '{directory}'.")
    return files

def extract_text_from_markdown(filepath: str) -> str | None:
    """Reads a markdown file and extracts plain text."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert Markdown to HTML
        html = markdown.markdown(md_content)

        # Extract text from HTML using BeautifulSoup for cleaner results
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator='\n', strip=True) # Use newline as separator

        # Basic fallback if BS4 fails or isn't used (less clean)
        # text = ''.join(BeautifulSoup(html, "html.parser").findAll(text=True))

        return text
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        return None

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunks text into smaller pieces based on sentences, respecting size and overlap."""
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        # If adding the sentence exceeds chunk size, finalize the current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap (take last few sentences/words of previous)
            # Simple overlap: take the last `chunk_overlap` chars of the previous chunk
            # More robust: re-evaluate sentences for overlap
            overlap_part = current_chunk[-chunk_overlap:]
            current_chunk = overlap_part + " " + sentence # Start new chunk with overlap + new sentence
            current_length = len(current_chunk)
            # Ensure the new chunk itself isn't immediately too large
            if current_length > chunk_size:
                 # If the single sentence + overlap is too big, just use the sentence
                 if sentence_length <= chunk_size:
                     chunks.append(sentence.strip()) # Add the long sentence as its own chunk if possible
                     current_chunk = "" # Reset
                     current_length = 0
                 else:
                     # Sentence is larger than chunk_size, split it crudely
                     # This is a fallback, ideally chunk_size is larger than typical sentences
                     for i in range(0, sentence_length, chunk_size - chunk_overlap):
                         sub_sentence = sentence[i:i + chunk_size]
                         chunks.append(sub_sentence.strip())
                     current_chunk = "" # Reset
                     current_length = 0

        # Otherwise, add the sentence to the current chunk
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_length += sentence_length + (1 if current_chunk else 0) # +1 for space

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out potentially empty chunks
    chunks = [chunk for chunk in chunks if chunk]
    logging.debug(f"Chunked text into {len(chunks)} chunks.")
    return chunks

