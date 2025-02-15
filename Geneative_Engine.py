import os
import re
import time
import nltk
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging  # For better error logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Setup ---
def setup_nltk():
    """Sets up NLTK resources."""
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    try:
        nltk.data.find("tokenizers/punkt/PY3/english.pickle")
    except LookupError:
        logging.info("Downloading NLTK punkt resource...")
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        logging.info(f"Downloaded punkt to: {nltk_data_path}")


# --- YouTube Data Fetching ---
def extract_video_id_from_link(youtube_link):
    """Extracts the video ID from a YouTube link."""
    try:
        video_id_match = re.search(r"(?P<id>([a-zA-Z0-9_-]{11}))", youtube_link)
        if video_id_match:
            return video_id_match.group("id")
        else:
            logging.warning(f"Invalid YouTube link format: {youtube_link}")
            return None
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return None


def get_transcript_text(video_id):
    """Fetches the transcript for a YouTube video and returns it as a single string."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ""
        for segment in transcript_list:
            full_transcript += segment['text'] + " "
        return full_transcript.strip()
    except NoTranscriptFound:
        logging.warning(f"No transcript found for video ID: {video_id}")
        return None
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return None


def get_video_details(video_id, api_key):
    """Retrieves video details (title, description, channel title) from YouTube."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        if response["items"]:
            video = response["items"][0]
            title = video["snippet"]["title"]
            description = video["snippet"]["description"]
            channel_title = video["snippet"]["channelTitle"]
            return title, description, channel_title
        else:
            logging.warning(f"Video with ID '{video_id}' not found.")
            return None, None, None

    except HttpError as e:
        if e.resp and hasattr(e.resp, 'status') and e.resp.status == 403:
            logging.error("Error: API quota exceeded or API key is invalid.")
        else:
            logging.exception(f"An HTTP error occurred: {e}")
        return None, None, None
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        return None, None, None


# --- Gemini Integration ---
def generate_notes_from_chunk(transcript_chunk, video_title, video_description, model):
    """Generates detailed notes from a transcript chunk using the Gemini model."""
    if model is None:
        logging.error("Gemini model is not initialized. Cannot generate notes.")
        return None

    prompt = f"""
    You are an expert note-taker. Based on the following transcript chunk from a YouTube video,
    create well-organized notes that explain the key topics.

    Follow these guidelines:

    1.  Identify the main topics discussed in the transcript chunk.
    2.  For each topic, write a paragraph-style explanation that summarizes the key information.
    3.  Use clear headings to introduce each topic.
    4.  Use combination of bullet points and paragraph.
    5.  Focus on providing a concise and understandable overview of each topic.

    Video Title: {video_title}

    Transcript Chunk:
    {transcript_chunk}

    Detailed Notes:
    """

    try:
        response = model.generate_content(prompt)
        return response.text  # Return the generated text
    except Exception as e:
        logging.exception(f"Error generating notes from chunk: {e}")
        return None  # Return None in case of error


def generate_summary(all_notes, video_title, model):
    """Generates a summary of the video based on all the notes."""
    if model is None:
        logging.error("Gemini model is not initialized. Cannot generate summary.")
        return None

    prompt = f"""
    You are an expert summarizer. Based on the following notes from a YouTube video,
    create a concise summary of the video in bullet points. Each bullet point should represent a key takeaway.
    Do not return any text like "Here's a concise summary of the video "What is Machine Learning?" in bullet points", only points.
    Video Title: {video_title}

    Notes:
    {all_notes}

    Summary:
    """

    try:
        response = model.generate_content(prompt)
        summary_text = response.text
        # Ensure that the response begins with a bullet point
        if not summary_text.startswith("*"):
            summary_text = "* " + summary_text
        return summary_text
    except Exception as e:
        logging.exception(f"Error generating summary: {e}")
        return None
# --- Chunking ---
def intelligent_chunking(text, max_tokens=500, lookahead_sentences=3):
    """
     Splits a text into chunks based on sentence boundaries and topic coherence,
     respecting a maximum token limit and using a lookahead strategy.

     Args:
         text (str): The text to chunk.
         max_tokens (int): The maximum number of tokens per chunk.  This is an approximate limit.
         lookahead_sentences (int): The number of sentences to look ahead to determine topic shift.

     Returns:
         list: A list of text chunks.
     """

    sentences = nltk.tokenize.sent_tokenize(text)  # Use nltk.tokenize
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence_token_count = len(sentence.split())  # Simple token count approximation

        if sentence_token_count > max_tokens:  # Handles single sentences that exceed max_tokens
            chunks.append(sentence)
            continue

        if current_token_count + sentence_token_count <= max_tokens:
            current_chunk += sentence + " "
            current_token_count = sentence_token_count
        else:
            # Check for topic shift using lookahead
            topic_shift = False
            if i + lookahead_sentences < len(sentences):
                # Calculate similarity between current chunk and lookahead sentences
                lookahead_text = " ".join(sentences[i:i + lookahead_sentences])
                similarity = calculate_similarity(current_chunk, lookahead_text)
                if similarity < 0.2:  # Adjust this threshold as needed
                    topic_shift = True

            if topic_shift:
                # End the current chunk and start a new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_token_count = sentence_token_count
            else:
                # Add the sentence to the current chunk even though it exceeds the limit
                # because it's likely related to the current topic.  We'll just have
                # slightly larger chunks sometimes.
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_token_count = sentence_token_count

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks



def calculate_similarity(text1, text2):
    """Calculates the cosine similarity between two texts using TF-IDF."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


# --- Main Function ---
def get_youtube_data(youtube_link, max_tokens=500, delay=3):
    """
    Fetches video title and yields notes from transcript chunks as they are generated.
    """
    try:
        video_id = extract_video_id_from_link(youtube_link)

        if not video_id:
            logging.warning(f"Invalid video ID for link: {youtube_link}")
            yield None, "Invalid YouTube link."
            return

        title, _, _ = get_video_details(video_id, API_KEY)

        if not title:
            logging.warning(f"Could not retrieve title for video ID: {video_id}")
            yield None, "Could not retrieve video title."
            return

        transcript = get_transcript_text(video_id)

        if not transcript:
            logging.warning(f"Could not retrieve transcript for video ID: {video_id}")
            yield None, "Could not retrieve video transcript."
            return

        transcript_chunks = intelligent_chunking(transcript, max_tokens=max_tokens)
        yield title, None  # Yield the title before the notes

        all_notes = ""  # Accumulate all notes for summary

        # Generate notes for each chunk and yield them
        for i, chunk in enumerate(transcript_chunks):
            notes = generate_notes_from_chunk(chunk, title, "", model)
            if notes:
                yield None, notes  # Yield only the notes
                all_notes += notes + "\n"  # Accumulate notes
                time.sleep(delay)  # Wait before processing the next chunk
            else:
                logging.warning(f"Failed to generate notes for chunk {i+1}")
                yield None, f"Failed to generate notes for chunk {i+1}"

        # Generate and yield the summary
        summary = generate_summary(all_notes, title, model)
        if summary:
            yield None, f"**Summary:**\n{summary}"  # Yield the summary
        else:
            logging.warning("Failed to generate summary.")
            yield None, "Failed to generate summary."

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        yield None, f"An unexpected error occurred: {e}"


# --- Initialization ---
setup_nltk()
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get Gemini API key from environment

# Initialize Gemini model outside the function to avoid re-initialization
try:
    genai.configure(api_key=GEMINI_API_KEY)  # Configure Gemini with the API key
    model = genai.GenerativeModel('gemini-2.0-flash-001')  # Select the model
    logging.info("Gemini model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Gemini model: {e}")
    model = None  # Set model to None if initialization fails


if __name__ == '__main__':
    # Example Usage (for testing the chunking and notes generation):
    example_youtube_link = "https://www.youtube.com/watch?v=HcqpanDadyQ"  # Replace with a real link

    print("Generated Notes:\n")
    for title, notes in get_youtube_data(example_youtube_link, max_tokens=300, delay=5):
        if title and notes:
            print(f"{title}")  # Printing the title
            print(f"{notes}")  # Printing the notes
        else:
            print(f"Error: {notes}")  # Notes contains the error