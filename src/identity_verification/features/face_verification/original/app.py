import gradio as gr
import numpy as np
from verification import analyze_image, load_models, log_event
from config import get_threshold
from webcam import start_challenge, track_frame


APP_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 40px 24px 56px !important;
}
.app-header { margin-bottom: 28px; }
.app-header h1 {
    margin: 0 0 8px;
    font-size: 2rem;
    letter-spacing: -0.03em;
}
.app-header p {
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 1rem;
}
.section-title { margin: 24px 0 8px; }
.section-title h2 { margin: 0; font-size: 1.05rem; }
.image-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    overflow: hidden;
    background: var(--background-fill-secondary);
}
.action-row { margin: 20px 0 8px; align-items: center; }
#verify-button { min-height: 46px; font-weight: 600; }
#result-panel {
    min-height: 116px;
    margin-top: 16px;
    padding: 18px 20px;
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    background: var(--background-fill-secondary);
}
#result-panel h2 { margin-top: 0; font-size: 1.25rem; }
.footer-note {
    margin-top: 20px;
    color: var(--body-text-color-subdued);
    font-size: 0.86rem;
}
"""



def verify_faces(id_photo, live_photo):
    if id_photo is None or live_photo is None:
        return None, None, '### Images required\n\nUpload both photos to continue.'
    try:
        id_view, id_face = analyze_image(id_photo)
        live_view, live_face = analyze_image(live_photo)
        score = float(np.clip(np.dot(id_face.normed_embedding, live_face.normed_embedding), -1, 1))
        threshold = get_threshold()
        title = 'Face match' if score >= threshold else 'No match'
        message = f'## {title}\n\nSimilarity: **{score:.4f}** | Threshold: **{threshold:.2f}**'
        log_event('comparison_completed', match=score >= threshold)
        return id_view, live_view, message
    except ValueError as error:
        log_event('comparison_rejected', reason=str(error))
        return None, None, f'### Image check failed\n\n{error}'
    except Exception as error:
        log_event('comparison_failed', error_type=type(error).__name__)
        return None, None, '### Comparison unavailable\n\nPlease retry or check the server logs.'


with gr.Blocks(title="Facial Verification") as demo:
    gr.HTML(
        """
        <div class="app-header">
            <h1>Facial Verification</h1>
            <p>Compare an identity document photo with a current photo.</p>
        </div>
        """
    )

    gr.HTML('<div class="section-title"><h2>1. Add photos</h2></div>')
    with gr.Row():
        with gr.Column():
            id_input = gr.Image(
                label="Identity document photo",
                type="numpy",
                height=360,
                elem_classes=["image-card"],
            )
        with gr.Column():
            live_input = gr.Image(
                label="Current photo",
                type="numpy",
                height=360,
                elem_classes=["image-card"],
            )

    with gr.Row(elem_classes=["action-row"]):
        verify_button = gr.Button(
            "Compare photos",
            variant="primary",
            size="lg",
            elem_id="verify-button",
        )
        clear_button = gr.ClearButton(
            [id_input, live_input], value="Clear", size="lg"
        )

    gr.HTML('<div class="section-title"><h2>2. Review result</h2></div>')
    with gr.Row():
        with gr.Column():
            id_output = gr.Image(
                label="Identity photo detection",
                height=320,
                interactive=False,
                elem_classes=["image-card"],
            )
        with gr.Column():
            live_output = gr.Image(
                label="Current photo detection",
                height=320,
                interactive=False,
                elem_classes=["image-card"],
            )

    result_output = gr.Markdown(
        "### Ready to compare\n\nAdd both photos, then select **Compare photos**.",
        elem_id="result-panel",
    )

    gr.HTML(
        '<p class="footer-note">Face detection: YOLOv8 &nbsp; | &nbsp; '
        "Face recognition: AuraFace</p>"
    )

    verify_button.click(
        fn=verify_faces,
        inputs=[id_input, live_input],
        outputs=[id_output, live_output, result_output],
        api_name='verify',
        concurrency_id='models',
    )
    clear_button.add([id_output, live_output, result_output])

    with gr.Accordion('Webcam motion challenge (experimental)', open=False):
        gr.Markdown('Follow the head-turn prompts while one face is tracked. '
                    'This basic check cannot establish liveness or reliably detect printed photos, screens or replay video. '
                    'Photo comparison above is independent of this challenge.')
        challenge_state = gr.State(None)
        start_button = gr.Button('Start new challenge')
        camera = gr.Image(sources=['webcam'], type='numpy', streaming=True, label='Webcam', webcam_options=gr.WebcamOptions(mirror=False))
        tracking_view = gr.Image(label='Tracked face', interactive=False)
        challenge_message = gr.Markdown('Start a challenge, then enable the webcam.')
        start_button.click(start_challenge, outputs=[challenge_state, challenge_message], api_name=False)
        camera.stream(track_frame, inputs=[camera, challenge_state],
                      outputs=[tracking_view, challenge_state, challenge_message],
                      stream_every=0.5, concurrency_limit=1, concurrency_id='models', api_name=False)


if __name__ == "__main__":
    get_threshold()
    load_models()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate"),
        css=APP_CSS,
    )
