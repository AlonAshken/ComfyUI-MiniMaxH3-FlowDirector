"""Save the last frame of a decoded batch, and pass the batch on untouched.
"""

import logging

from comfy_api.latest import io, ui

log = logging.getLogger(__name__)


class MiniMaxH3SaveLastFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SaveLastFrameCS",
            display_name="MiniMax H3 Save Last Frame",
            category="MiniMax H3",
            description=(
                "Saves the LAST frame of an image batch as a PNG, and "
                "passes the whole batch through unchanged. Useful for saving the final continuous frame."
            ),
            inputs=[
                io.Image.Input("images",
                               tooltip="The decoded frames. Passed on unchanged, whether or not a file is written."),
                io.Boolean.Input("save", default=True,
                                 tooltip="Off = pass the frames through and write nothing."),
                io.String.Input("filename_prefix", default="MiniMaxH3/lastframe",
                                tooltip="Prefix for the saved file under ComfyUI output."),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The input batch, untouched."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, save=True, filename_prefix="MiniMaxH3/lastframe") -> io.NodeOutput:
        if not save:
            return io.NodeOutput(images)

        count = int(images.shape[0]) if getattr(images, "shape", None) is not None else 0
        if count < 1:
            log.warning("[MiniMaxSaveLastFrame] the batch is empty — nothing to save.")
            return io.NodeOutput(images)

        last = images[-1:]
        saved = ui.ImageSaveHelper.get_save_images_ui(last, filename_prefix=filename_prefix, cls=cls)
        names = ", ".join(r["filename"] for r in saved.results)
        log.info("[MiniMaxSaveLastFrame] frame %d of %d (%dx%d) -> %s",
                 count, count, int(images.shape[2]), int(images.shape[1]), names)

        return io.NodeOutput(images, ui=saved)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SaveLastFrameCS": MiniMaxH3SaveLastFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SaveLastFrameCS": "MiniMax H3 Save Last Frame"}
