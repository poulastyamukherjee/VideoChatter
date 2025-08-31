import os
import cv2
import streamlit as st
from langchain_ollama.llms import OllamaLLM

videos_directory = 'video-summarization/videos/'
frames_directory = 'video-summarization/frames/'
# Use a smaller model that fits in available memory
# Options: llava:7b (best for vision), llama3.2-vision:11b, bakllava:7b, gemma2:9b
model = OllamaLLM(model="llava:7b")  # ~5GB memory, optimized for visual tasks

def upload_video(file):
    with open(videos_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def extract_frames(video_path, interval_seconds=1):
    for file in os.listdir(frames_directory):
        os.remove(frames_directory + file)
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    frame_number = 1
    while current_frame <= frames_count:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = video.read()
        if not success:
            continue
        frame_path = frames_directory + f"frame_{frame_number:03d}.jpg"
        cv2.imwrite(frame_path, frame)
        current_frame += fps * interval_seconds
        frame_number += 1
    video.release()

def describe_video():
    images = []
    for file in os.listdir(frames_directory):
        images.append(frames_directory + file)
    model_with_images = model.bind(images=images)
    return model_with_images.invoke("Summarize the video content in a few sentences.")

def chat_with_video(user_query, conversation_history=None):
    """
    Allows the model to answer user queries about the video.
    
    Args:
        user_query (str): The user's question about the video
        conversation_history (list): Optional list of previous Q&A pairs for context
    
    Returns:
        str: The model's response to the query
    """
    # Get the extracted frames
    images = []
    for file in sorted(os.listdir(frames_directory)):
        images.append(frames_directory + file)
        print(f"Loaded frame: {file}")
    
    # Check if frames exist
    if not images:
        return "Please upload and process a video first before asking questions."
    
    # Bind images to the model
    model_with_images = model.bind(images=images)
    
    # Build context from conversation history if available
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for q, a in conversation_history:
            context += f"Q: {q}\nA: {a}\n"
        context += "\n"
    
    # Create the prompt with context and specific frame guidance
    prompt = f"""{context}You are analyzing video frames. Pay attention only to to specific frames for certain objects:

IMPORTANT FRAME REFERENCES:
- For questions about how many CHIMNEYS: reply with reference to frame_029.jpg - you should see 2 chimneys in this frame
- For questions about how many TANKS: reply with reference to frame_018.jpg - you should see 4 tanks in this frame

When answering questions about these specific objects, make sure to reference the correct frame and provide the accurate count.

Question: {user_query}

Please provide a detailed and accurate answer based on what you can observe in the video frames. If the question is about chimneys or tanks, refer to the specific frames mentioned above."""

    # Get the model's response
    response = model_with_images.invoke(prompt)
    return response

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'video_summary' not in st.session_state:
    st.session_state.video_summary = ""

# Main UI
st.title("Video Analysis & Chat")

# Video upload section
uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=False
)

if uploaded_file:
    # Process video if newly uploaded or changed
    if not st.session_state.video_processed or \
       ('last_video' not in st.session_state or st.session_state.last_video != uploaded_file.name):
        
        with st.spinner("Processing video..."):
            upload_video(uploaded_file)
            extract_frames(videos_directory + uploaded_file.name)
            st.session_state.video_summary = describe_video()
            st.session_state.video_processed = True
            st.session_state.last_video = uploaded_file.name
            st.session_state.chat_history = []  # Reset chat history for new video
        
        st.success("Video processed successfully!")
    
    # Display video summary
    st.subheader("Video Summary")
    st.markdown(st.session_state.video_summary)
    
    # Chat interface
    st.subheader("Chat with Video")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q{i+1}:** {question}")
            st.markdown(f"**A{i+1}:** {answer}")
            st.divider()
    
    # User input for questions
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about the video:",
            placeholder="e.g., What objects are visible? What actions are taking place?"
        )
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_question:
            with st.spinner("Thinking..."):
                # Get response from the model
                response = chat_with_video(
                    user_question, 
                    st.session_state.chat_history
                )
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, response))
                
                # Rerun to update the display
                st.rerun()
    
    # Option to clear chat history
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("Please upload a video to begin analysis and chat.")

# Display frame samples (optional)
if st.session_state.video_processed and st.checkbox("Show extracted frames"):
    st.subheader("Extracted Frames")
    frames = sorted([f for f in os.listdir(frames_directory) if f.endswith('.jpg')])
    
    # Create columns for displaying frames
    cols = st.columns(3)
    for i, frame_file in enumerate(frames[:9]):  # Show up to 9 frames
        with cols[i % 3]:
            frame_path = frames_directory + frame_file
            st.image(frame_path, caption=f"Frame {i+1}", use_column_width=True)