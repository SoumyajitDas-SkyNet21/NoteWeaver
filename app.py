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
        print("Fetching transcript...")
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])

        cleaned_text = ""
        for item in transcript:
            cleaned_text += item['text'] + " "

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        print("Transcript fetched successfully.")
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

stop = stop_after_attempt(5)
wait = wait_exponential(multiplier=1, min=2, max=10)

@retry(stop=stop, wait=wait)
def generate_notes_with_retry(prompt):
    try:
        print("Calling Gemini API with prompt: ", prompt)
        response = model.generate_content(prompt, stream=True)
        full_text = ""
        for chunk in response:
            full_text += chunk.text
        print("Gemini API call successfull")
        return full_text
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}, Prompt: {prompt}")
        raise

def analyze_transcript_chunked(transcript, delay=2):
    if not transcript:
        st.error("No transcript found or an error occurred while fetching the transcript.")
        return None, ""  # Returning None, "" if transcript not found
    print("Transcript received successfully")
    chunks = chunk_text(transcript)
    previous_topics = ""
    previous_summary = ""
    full_text = ""
    notes_container = st.empty()
    overall_notes = ""

    for index, chunk in enumerate(chunks):
        logging.info(f"Analyzing chunk {index + 1} of {len(chunks)}")
        # --- MODIFIED PROMPT ---
        prompt = f"""
        Previous Summary:
        {previous_summary}

        Given the following part of the transcript:
        {chunk}

        Analyze the provided text to understand and extract key topics in the order they appear. Generate clear, structured notes for each topic, maintaining the natural sequence of ideas as presented in the video. Act as an expert in this field and present the information in a professional yet accessible manner.
        When choosing topic names, be creative and specific to the content being discussed in this section. Prioritize clarity and uniqueness in the headings, so that each heading effectively summarizes the key focus of that particular section. Use different wording than previous topics, unless the topic is precisely the same. Choose the topic name wisely. Format the notes using appropriate headings, subheadings, and bullet points for clarity. Avoid unnecessary details—focus only on essential insights and meaningful takeaways. Ensure conciseness while preserving the depth of the content.Summarize key points, infer meaningful insights where applicable, and avoid redundancy. Do not number the topics—only use their descriptive titles.

        """
        # --- END MODIFIED PROMPT ---
        print("Chunk: ", chunk)
        try:
            print("Before notes generation for chunk", index + 1)
            notes = generate_notes_with_retry(prompt)

            # Apply styling to headings only using regex (removing font-size)
            formatted_notes = re.sub(r'(#+) (.*?)(?=\n|$)', r'<div style="text-align: left;"><span style="font-size: 3.8em;">\1 \2</span></div>', notes) #Increased heading size
            formatted_notes = f"<span style='font-size: 5.2em;'>{formatted_notes}</span>" #Increased paragraph size
            full_text += formatted_notes
            full_text += "<br><br><br>"
            with notes_container.container():
                st.markdown(full_text, unsafe_allow_html=True)
            print("After notes generated for chunk", index + 1)
            overall_notes += notes + "\n\n" # Appending to overall_notes

            analysis_prompt = f"""
            Given the following text:
            {notes}

            Extract the topics that were discussed and create a summary of the text, keep the font consistent throughtout the content.

            return the result as a json object like this
            {{
            "topics": ["topic1", "topic2" ...],
            "summary": "your summary here"
            }}
            """
            try:
                print("Before calling gemini for analysis for chunk", index + 1)
                analysis_response = model.generate_content(analysis_prompt)
                print("After calling gemini for analysis for chunk", index + 1)
                print("Analysis Response:", analysis_response.text)
                analysis_json = json.loads(analysis_response.text)
                print("Analysis Json: ", analysis_json)
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

    return full_text, overall_notes #returning overall notes too
def generate_overall_summary(overall_notes):
    prompt = f"""
    Given all the notes taken:
    {overall_notes}

    Generate an overall summary of the video. Focus on the main points and key takeaways.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error during overall summary Gemini API call: {e}")
        return "Failed to generate overall summary."

def main():
    st.set_page_config(page_title="Note Weaver", layout="wide")
    st.markdown(
    """
    <style>
    /* Target the markdown headings directly */
    div[data-baseweb="block-container"] h1, 
    div[data-baseweb="block-container"] h2,
    div[data-baseweb="block-container"] h3,
    div[data-baseweb="block-container"] h4,
    div[data-baseweb="block-container"] h5,
    div[data-baseweb="block-container"] h6 {
        font-size: 3.2em !important; /* Override any other styles */
    }

    /* Target the markdown paragraphs */
    div[data-baseweb="block-container"] p,
    div[data-baseweb="block-container"] span {
    font-size: 7.8em !important;
    }
    div[data-baseweb="block-container"] li {
        font-size: 1.8em !important;
    }
    

    /* Add margin to the Clear Notes button */
    .stButton > button {
        margin-top: 3em !important; /* Adjust this value as needed */
    }

    /* Reduce spacing between input and Generate button */
    .stTextInput, .stButton {
        margin-bottom: -1.5em !important; /* Adjust this value as needed */
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
        .button-container { max-width: 800px; margin: 0 auto;text-align: center;display: flex;justify-content: center;max-width: 300px; }
        .notes-container { max-width: 800px; margin: 20px auto; }
        .stButton button { width: 20%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use st.empty() to create a container for the entire app content
    app_container = st.empty()

    with app_container.container():  # Wrap all the previous code into a container
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        video_url = st.text_input("Enter a YouTube video URL:")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("Generate Notes"):
            with st.spinner("Processing..."):
                if video_url:
                    st.markdown('<div class="notes-container">', unsafe_allow_html=True)
                    full_text, overall_notes = analyze_transcript_chunked(get_youtube_transcript(video_url)) #returning overall notes

                    st.markdown('</div>', unsafe_allow_html=True)
                    if overall_notes:  # Check if there are any topics before generating the summary
                        overall_summary = generate_overall_summary(overall_notes)
                        st.subheader("Summary", divider = "rainbow") #Added "Summary" as a heading
                        st.markdown(f"""

                                        {overall_summary}

                                        """, unsafe_allow_html=True)

        # Add a reset button that clears the cache when clicked
        if st.button("Clear Notes"):
            app_container.empty()  # clear contents
            st.rerun()  # Rerun the entire script
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
