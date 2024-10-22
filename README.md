# AI Research Assistant

Research assistant tool that can process academic papers, YouTube videos, and web pages. It provides intelligent summaries and allows you to ask questions about the processed content, with text-to-speech capabilities for accessibility.

## Features

- Process multiple content types:
  - arXiv papers (via paper ID or URL)
  - YouTube videos (via video ID or URL)
  - Web pages (via URL)
- Generate intelligent summaries using GPT models
- Interactive Q&A about processed content
- Text-to-speech capability for summaries and answers
- User-friendly Gradio interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arthurchipdean/ai-research-assistant.git
cd ai-research-assistant
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Get an OpenAI API key:
   - Visit [OpenAI's website](https://platform.openai.com/)
   - Sign up or log in
   - Go to API Keys section
   - Create a new secret key
   - Copy the key (make sure to save it as it won't be shown again)

2. Set up environment variables:
```bash
cp .env.dist .env
```

3. Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

3. Use the interface to:
   - Select content type (arXiv Paper, YouTube Video, or Webpage)
   - Enter the ID or URL of the content
   - Click "Process Content" to generate a summary
   - Ask questions about the processed content
   - Use the "Read Aloud" feature to listen to summaries and answers

## Model Configuration

The application uses GPT-4o-mini by default to optimize for cost and performance. You can modify the `MODEL_NAME` constant to use other OpenAI models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for providing the GPT models
- Gradio team for the excellent UI framework
- All other open-source libraries used in this project

## Note

Please be mindful of API usage costs when using OpenAI models. The GPT-4o-mini model is used by default to minimize costs while maintaining good performance.