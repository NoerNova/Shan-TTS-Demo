import gradio as gr
from tts import synthesize, TTS_EXAMPLES

mms_synthesize = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Dropdown(["original", "nova", "homhom"], label="Model", value="nova"),
        gr.Textbox(label="Input text"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Speed"),
    ],
    outputs=[
        gr.Audio(label="Generated Audio", type="numpy"),
        gr.Textbox(label="Filtered text after processing"),
    ],
    examples=TTS_EXAMPLES,
    title="Text-to-Speech Demo",
    description="Generate audio in your desired language from input text.",
    allow_flagging="never",
)

with gr.Blocks() as demo:
    mms_synthesize.render()

demo.launch()
