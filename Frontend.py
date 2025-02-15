import streamlit as st
from Geneative_Engine import get_youtube_data  # Import the get_youtube_data function

# For centering the Titles
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# For centering the buttons
st.markdown(
    """
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>NoteWeaver</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='title'>Generate Notes from YouTube Video</h4>", unsafe_allow_html=True)

video_url = st.text_input('Enter YouTube Video URL:')

# Initialize session state
if 'all_notes' not in st.session_state:
    st.session_state['all_notes'] = []
if 'video_title' not in st.session_state:
    st.session_state['video_title'] = None
if 'generation_complete' not in st.session_state:
    st.session_state['generation_complete'] = False
if 'generate_button_clicked' not in st.session_state:
    st.session_state['generate_button_clicked'] = False

# Use st.empty() to create a placeholder for the notes
notes_placeholder = st.empty()

# Conditionally display the generate button
if not st.session_state['generate_button_clicked']:
    if st.button('Generate'):
        st.session_state['generate_button_clicked'] = True
        if video_url:
            st.session_state['generation_complete'] = False
            #st.write('Notes are being generated !') # No longer needed, since it will generate each time a note is created

            notes_area = notes_placeholder.container() # create a container to put notes inside
            with notes_area:
                st.session_state['all_notes'] = [] # Reset the notes!
                has_printed_title = False # Helper boolean
                for title, notes in get_youtube_data(video_url):
                    if title and not has_printed_title and title != "None":
                        st.markdown(f"### {title}")  # Display the title only once
                        has_printed_title = True
                        st.session_state['video_title'] = title
                    elif notes and "**Summary:**" not in notes:
                        st.write(notes)  # Append notes to the text area
                        st.session_state['all_notes'].append(notes)  # Accumulate notes
                    elif notes and "**Summary:**" in notes:
                        st.markdown("---")  # Horizontal line
                        st.markdown(f"<h2 style='text-align: center;'>Summary</h2>", unsafe_allow_html=True)  # Larger title
                        summary_text = notes.replace("**Summary:**\n", "")
                        summary_items = summary_text.split("* ")
                        for item in summary_items:
                             if item.strip():
                                  st.markdown(f"* {item.strip()}")
                        st.session_state['all_notes'].append(notes)  # Accumulate notes
                    else:
                        st.error(notes)  # Display the error message
            st.session_state['generation_complete'] = True

        else:
            st.warning('Please Input Video URL')

if st.session_state['generation_complete']:
    if st.button("Clear Notes"):
        notes_placeholder.empty()
        st.session_state['all_notes'] = []
        st.session_state['video_title'] = None
        st.session_state['generation_complete'] = False
        st.session_state['generate_button_clicked'] = False #Reenable the generate button.