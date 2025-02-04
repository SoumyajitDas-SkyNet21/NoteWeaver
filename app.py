import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
import google.generativeai as genai
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual Gemini API key
GOOGLE_API_KEY = "AIzaSyA5qLZ1w6fsdCLrcRiTuwdYRj5srZ2ZOBo"
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = genai.types.GenerationConfig(
    temperature=0.1,
)
model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config
)

def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])

        cleaned_text = ""
        for item in transcript:
            cleaned_text += item['text'] + " "

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    except Exception as e:
        logging.error(f"An error occurred while fetching transcript: {e}")
        return None

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_notes_with_retry(prompt):
    try:
        response = model.generate_content(prompt, stream=True)
        full_text = ""
        for chunk in response:
            full_text += chunk.text
        return full_text
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}, Prompt: {prompt}")
        raise

def analyze_transcript_chunked(transcript, delay=2):
    if not transcript:
        st.error("No transcript found or an error occurred while fetching the transcript.")
        return

    chunks = chunk_text(transcript)
    previous_topics = ""
    previous_summary = ""
    full_text = ""
    notes_container = st.empty()

    for index, chunk in enumerate(chunks):
        logging.info(f"Analyzing chunk {index + 1} of {len(chunks)}")
        prompt = f"""
        Previous Summary:
        {previous_summary}

        Given the following part of the transcript:
        {chunk}

        Analyze the provided text to understand and extract key topics in the order they appear. Generate clear, structured notes for each topic, maintaining the natural sequence of ideas as presented in the video.Act as an expert in this field and present the information in a professional yet accessible manner. Format the notes using appropriate headings, subheadings, and bullet points for clarity. Avoid unnecessary details—focus only on essential insights and meaningful takeaways. Ensure conciseness while preserving the depth of the content.Summarize key points, infer meaningful insights where applicable, and avoid redundancy. Do not number the topics—only use their descriptive titles.

        """
        try:
            notes = generate_notes_with_retry(prompt)
            formatted_notes = f"<span style='font-size:4.2em'>{notes}</span>"
            full_text += formatted_notes
            full_text += "<br><br><br>"
            with notes_container.container():
                st.markdown(full_text, unsafe_allow_html=True)

            analysis_prompt = f"""
            Given the following text:
            {notes}

            Extract the topics that were discussed and create a summary of the text.

            return the result as a json object like this
            {{
            "topics": ["topic1", "topic2" ...],
            "summary": "your summary here"
            }}
            """
            try:
                analysis_response = model.generate_content(analysis_prompt)
                analysis_json = json.loads(analysis_response.text)
                previous_topics = ", ".join(analysis_json["topics"])
                previous_summary = analysis_json["summary"]
            except Exception as e:
                logging.error(f"Error while parsing json response: {e}, Response: {analysis_response.text}")
                previous_topics = ""
                previous_summary = ""

        except Exception as e:
            logging.error(f"Error during analysis of chunk {index + 1}: {e}")
            st.error(f"Failed to process chunk {index + 1}")
        time.sleep(delay)

def main():
    st.set_page_config(page_title="Note Weaver", layout="wide")
    st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 { 
        font-size: 2rem; /* Adjust the size for headings */
    }

    p, span, li { 
        font-size: 1.2rem; /* Adjust the size for normal text and bullet points */
    }
    
    /* Optional: Make text responsive for smaller screens */
    @media (max-width: 768px) {
        h1, h2, h3, h4, h5, h6 { 
            font-size: 1.5rem; 
        }
        p, span, li { 
            font-size: 1rem; 
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("<h1 style='text-align: center;'>Note Weaver</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Generate structured notes from a YouTube video</p>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .input-container { max-width: 800px; margin: 0 auto; }
        .button-container { margin: 0 auto;text-align: center;display: flex;justify-content: center;max-width: 300px; }
        .notes-container { max-width: 800px; margin: 20px auto; }
        .stButton button { width: 20%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    video_url = st.text_input("Enter a YouTube video URL:")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Generate Notes"):
        with st.spinner("Processing..."):
            if video_url:
                st.markdown('<div class="notes-container">', unsafe_allow_html=True)
                analyze_transcript_chunked(get_youtube_transcript(video_url))
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()