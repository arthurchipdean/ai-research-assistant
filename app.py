import gradio as gr
from typing import List, Tuple
import arxiv
import youtube_transcript_api
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from urllib.parse import urlparse, parse_qs
from gtts import gTTS
import markdown
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_NAME = os.getenv('MODEL_NAME')
OPENAI_KEY = os.getenv('OPENAI_KEY')

class ResearchAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.last_processed_content = ""
    
    def process_arxiv_paper(self, paper_id: str) -> Tuple[bool, str]:
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            content = f"## {paper.title}\n\n**Authors:** {', '.join([a.name for a in paper.authors])}\n\n**Summary:** {paper.summary}"
            self.last_processed_content = content
            return True, content
        except arxiv.arxiv.HTTPError:
            return False, "Error: Invalid arXiv ID or paper not found."
        except StopIteration:
            return False, "Error: No paper found with the given ID."
    
    def process_youtube_video(self, video_id: str) -> Tuple[bool, str]:
        try:
            transcript = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
            full_text = ' '.join([entry['text'] for entry in transcript])
            summary = self._generate_summary(full_text)
            self.last_processed_content = summary
            return True, summary
        except youtube_transcript_api.NoTranscriptAvailable:
            return False, "Error: No transcript available for this video."
        except youtube_transcript_api.VideoUnavailable:
            return False, "Error: The video is unavailable or doesn't exist."
    
    def process_webpage(self, url: str) -> Tuple[bool, str]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('article') or soup.find('main') or soup.body
            if not content:
                return False, "Error: Unable to extract content from the webpage."
            text = content.get_text()
            summary = self._generate_summary(text)
            self.last_processed_content = summary
            return True, summary
        except requests.exceptions.RequestException as e:
            return False, f"Error fetching webpage: {str(e)}"
    
    def _generate_summary(self, text: str) -> str:
        chunks = self.text_splitter.split_text(text)
        summaries = []
        
        for chunk in chunks:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Summarize the following text, highlighting key findings. Use markdown formatting for headings, lists, and emphasis:"},
                    {"role": "user", "content": chunk}
                ]
            )
            summaries.append(response.choices[0].message.content)
        
        return self._combine_summaries(summaries)
    
    def _combine_summaries(self, summaries: List[str]) -> str:
        combined_text = "\n\n".join(summaries)
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Combine these summaries into a coherent summary. Use markdown formatting for structure, including headings, lists, and emphasis:"},
                {"role": "user", "content": combined_text}
            ]
        )
        return response.choices[0].message.content

    def answer_question(self, question: str) -> str:
        if not self.last_processed_content:
            return "Error: No content has been processed yet. Please process a paper, video, or webpage first."
        
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a research assistant. Answer the following question based on the provided content. If the answer is not in the content, say so."},
                {"role": "user", "content": f"Content: {self.last_processed_content}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content

assistant = ResearchAssistant()

def validate_arxiv_id(identifier: str) -> Tuple[bool, str]:
    arxiv_pattern = r'\d{4}\.\d{4,5}(v\d+)?$'
    if re.match(arxiv_pattern, identifier):
        return True, identifier
    elif identifier.startswith('http'):
        parsed_url = urlparse(identifier)
        if parsed_url.netloc in ['arxiv.org', 'www.arxiv.org']:
            path_parts = parsed_url.path.split('/')
            if len(path_parts) > 2 and re.match(arxiv_pattern, path_parts[-1]):
                return True, path_parts[-1]
    return False, "Invalid arXiv ID or URL."

def extract_youtube_id(identifier: str) -> Tuple[bool, str]:
    if len(identifier) == 11:
        return True, identifier
    parsed_url = urlparse(identifier)
    if parsed_url.netloc in ['youtube.com', 'www.youtube.com', 'youtu.be']:
        if parsed_url.netloc == 'youtu.be':
            return True, parsed_url.path[1:]
        if parsed_url.path == '/watch':
            return True, parse_qs(parsed_url.query)['v'][0]
    return False, "Invalid YouTube URL or video ID."

def validate_url(url: str) -> Tuple[bool, str]:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        result = urlparse(url)
        return (True, url) if all([result.scheme, result.netloc]) else (False, "Invalid URL format.")
    except:
        return False, "Invalid URL format."

def process_content(content_type: str, identifier: str) -> str:
    if content_type == "Paper":
        valid, result = validate_arxiv_id(identifier)
        if not valid:
            return result
        success, output = assistant.process_arxiv_paper(result)
    elif content_type == "Video":
        valid, result = extract_youtube_id(identifier)
        if not valid:
            return result
        success, output = assistant.process_youtube_video(result)
    elif content_type == "Webpage":
        valid, result = validate_url(identifier)
        if not valid:
            return result
        success, output = assistant.process_webpage(result)
    else:
        return "Error: Invalid content type selected"
    
    return output if success else f"Error: {output}"

def answer_research_question(question: str) -> str:
    return assistant.answer_question(question)

def text_to_speech(text: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(text=text, lang='en')
        tts.save(temp_audio.name)
        return temp_audio.name

custom_css = """
.container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
}

.gradio-button {
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.primary-button {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    color: white !important;
}

.secondary-button {
    background: linear-gradient(90deg, #059669, #10b981) !important;
    color: white !important;
}

.radio-group {
    background: #f8fafc !important;
    padding: 15px !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
}

.textbox {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

.markdown-header {
    color: #1e293b !important;
    font-weight: 600 !important;
}
"""

# Icons (using emoji as placeholders - you can replace these with actual icons)
ICONS = {
    "process": "🔄",
    "play": "🔊",
    "question": "❓",
    "paper": "📄",
    "video": "🎥",
    "web": "🌐"
}

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as application:
    gr.Markdown("# 🤖 AI Research Assistant", elem_classes="markdown-header")
    
    with gr.Row():
        with gr.Column(scale=1):
            content_type = gr.Radio(
                choices=[
                    (ICONS["paper"] + " arXiv Paper", "Paper"),
                    (ICONS["video"] + " YouTube Video", "Video"),
                    (ICONS["web"] + " Webpage", "Webpage")
                ],
                label="Select Content Type",
                value="Paper",
                elem_classes="radio-group"
            )
        
        with gr.Column(scale=2):
            identifier = gr.Textbox(
                label="Enter ID/URL",
                placeholder="e.g., 2311.12399, youtube.com/watch?v=dQw4w9WgXcQ, or example.com",
                elem_classes="textbox"
            )
    
    with gr.Row():
        process_btn = gr.Button(
            ICONS["process"] + " Process Content",
            variant="primary",
            elem_classes=["gradio-button", "primary-button"]
        )
    
    output = gr.Markdown(label="Results")
    
    with gr.Row():
        tts_btn = gr.Button(
            ICONS["play"] + " Read Aloud",
            variant="secondary",
            visible=False,
            elem_classes=["gradio-button", "secondary-button"]
        )
    
    audio_output = gr.Audio(label="Audio Output", visible=False)
    
    gr.Markdown("## " + ICONS["question"] + " Ask Questions", elem_classes="markdown-header")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask a question about the processed content...",
            elem_classes="textbox"
        )
    
    with gr.Row():
        answer_btn = gr.Button(
            "Get Answer",
            variant="primary",
            elem_classes=["gradio-button", "primary-button"]
        )
    
    answer_output = gr.Markdown(label="Answer")
    
    with gr.Row():
        answer_tts_btn = gr.Button(
            ICONS["play"] + " Read Answer",
            variant="secondary",
            visible=False,
            elem_classes=["gradio-button", "secondary-button"]
        )
    
    answer_audio_output = gr.Audio(label="Answer Audio Output", visible=False)

    def process_with_loading(content_type, identifier):
        # Extract the actual content type from the icon+text combination
        actual_content_type = content_type.split(" ")[-1]
        return process_content(actual_content_type, identifier), gr.update(visible=True), gr.update(visible=False)

    def answer_with_loading(question):
        return answer_research_question(question), gr.update(visible=True), gr.update(visible=False)

    def read_aloud(text):
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        plain_text = soup.get_text()
        audio_file = text_to_speech(plain_text)
        return gr.update(value=audio_file, visible=True)

    process_btn.click(
        fn=process_with_loading,
        inputs=[content_type, identifier],
        outputs=[output, tts_btn, audio_output],
        api_name="process_content"
    ).success(lambda: gr.update(interactive=True), None, answer_btn)

    tts_btn.click(
        fn=read_aloud,
        inputs=output,
        outputs=audio_output,
        api_name="read_summary"
    )

    answer_btn.click(
        fn=answer_with_loading,
        inputs=question_input,
        outputs=[answer_output, answer_tts_btn, answer_audio_output],
        api_name="answer_question"
    )

    answer_tts_btn.click(
        fn=read_aloud,
        inputs=answer_output,
        outputs=answer_audio_output,
        api_name="read_answer"
    )

if __name__ == "__main__":
    if not OPENAI_KEY:
        print("Please set your OpenAI API key in the .env file.")
    else:
        application.launch()