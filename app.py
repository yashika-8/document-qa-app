import streamlit as st
from transformers import pipeline
from pdf2image import convert_from_path
import os
import tempfile
from PIL import Image
import time

# Initialize session state variables if they don't exist
if 'processed_pages' not in st.session_state:
    st.session_state.processed_pages = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

# Configure page
st.set_page_config(page_title="Document Q&A", layout="wide")

# Initialize the document-question-answering pipeline
@st.cache_resource
def load_pipeline():
    return pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa",
        device=-1
    )

try:
    query_pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("📄 Document Question-Answering System")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Upload a PDF file to analyze"
    )

    if uploaded_file:
        # Check if we need to process a new file
        if (st.session_state.current_file_name != uploaded_file.name or 
            st.session_state.processed_pages is None):
            
            with st.spinner("Processing PDF..."):
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(uploaded_file.getvalue())

                try:
                    # Convert PDF to images
                    pages = convert_from_path(
                        temp_file_path,
                        dpi=300
                    )
                    st.session_state.processed_pages = pages
                    st.session_state.current_file_name = uploaded_file.name
                    
                    st.success(f"✅ Document loaded successfully! ({len(pages)} pages)")
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.processed_pages = None
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        # Display document preview
        if st.session_state.processed_pages:
            pages = st.session_state.processed_pages
            
            if len(pages) == 1:
                st.image(
                    pages[0],
                    caption="Single page document",
                    use_column_width=True
                )
                current_page_idx = 0
            else:
                preview_page = st.slider(
                    "Preview page", 
                    1, len(pages), 
                    1,
                    help="Select a page to preview"
                )
                st.image(
                    pages[preview_page-1],
                    caption=f"Page {preview_page} of {len(pages)}",
                    use_column_width=True
                )
                current_page_idx = preview_page - 1

with col2:
    if st.session_state.processed_pages:
        st.markdown("### Ask Questions")
        
        # Example questions based on current page
        st.markdown("""
        **Example questions you can ask:**
        - "What is the title of this document?"
        - "What is the date mentioned?"
        - "List the main points in this section."
        - "What is written in the first paragraph?"
        """)
        
        question = st.text_input(
            "Enter your question:",
            placeholder="What does the document say about...",
            help="Ask a specific question about the document's content"
        )
        
        search_mode = st.radio(
            "Search mode:",
            ["Quick (Best match)", "Thorough (All pages)"],
            help="Quick mode searches until it finds a good answer. Thorough mode checks all pages."
        )

        if st.button("🔍 Search for Answer", use_container_width=True):
            if question:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_answers = []
                pages = st.session_state.processed_pages
                
                for i, page in enumerate(pages):
                    progress = (i + 1) / len(pages)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing page {i+1}/{len(pages)}...")
                    
                    try:
                        # Preprocess the image
                        page_rgb = page.convert('RGB')
                        
                        # Adjust image size if needed
                        max_size = 1000
                        if max(page_rgb.size) > max_size:
                            ratio = max_size / max(page_rgb.size)
                            new_size = tuple(int(dim * ratio) for dim in page_rgb.size)
                            page_rgb = page_rgb.resize(new_size, Image.LANCZOS)
                        
                        # Process with more lenient confidence threshold
                        result = query_pipeline(
                            question=question,
                            image=page_rgb,
                            top_k=3,
                            max_length=512,
                            max_answer_length=200
                        )
                        
                        if isinstance(result, dict):
                            result = [result]
                        
                        for res in result:
                            if res and isinstance(res, dict):
                                answer = res.get('answer', '').strip()
                                score = res.get('score', 0)
                                
                                if score > 0.01 and len(answer) > 0:
                                    all_answers.append({
                                        'page': i+1,
                                        'answer': answer,
                                        'score': score
                                    })
                        
                        if search_mode == "Quick (Best match)" and all_answers and all_answers[-1]['score'] > 0.05:
                            break
                    
                    except Exception as e:
                        st.warning(f"Error processing page {i+1}: {str(e)}")
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                if all_answers:
                    st.markdown("### 📝 Results")
                    all_answers.sort(key=lambda x: x['score'], reverse=True)
                    
                    for idx, answer in enumerate(all_answers):
                        with st.expander(
                            f"Answer {idx+1} - Page {answer['page']} (Confidence: {answer['score']:.2%})",
                            expanded=True
                        ):
                            st.write(answer['answer'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"👍 Helpful", key=f"helpful_{idx}"):
                                    st.success("Thank you for your feedback!")
                            with col2:
                                if st.button(f"👎 Not Helpful", key=f"not_helpful_{idx}"):
                                    st.info("Thank you for your feedback! Try rephrasing the question.")
                else:
                    st.warning("No answers found. Try these tips:")
                    st.markdown("""
                    - Try asking about specific text you can see in the document
                    - Break down complex questions into simpler ones
                    - Check if the text in your PDF is actual text (not an image)
                    - Use the exact words that appear in the document
                    """)
            else:
                st.warning("Please enter a question first.")
    else:
        st.info("👈 Please upload a PDF document to begin asking questions.")

# Debug information section
with st.expander("🔧 Debug Information", expanded=False):
    st.markdown("### System Information")
    st.write("Model:", "impira/layoutlm-document-qa")
    if st.session_state.processed_pages:
        st.write("Document pages:", len(st.session_state.processed_pages))
        st.write("Current page dimensions:", st.session_state.processed_pages[0].size)
        
        test_question = st.text_input("Enter a test question:", value="What is written here?")
        if st.button("Run Test Query"):
            try:
                test_page = st.session_state.processed_pages[0].convert('RGB')
                result = query_pipeline(question=test_question, image=test_page)
                st.write("Raw result:", result)
            except Exception as e:
                st.error(f"Test error: {str(e)}")