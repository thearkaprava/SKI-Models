from llavidal.demo.gradio_css import code_highlight_css
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes.base import Base
from typing import Iterable


tos_markdown = ("""
<div style="color:grey; text-align: justify;">
LLAVIDAL Key Use Guidelines: This research project demo is solely for non-commercial use. It has finite safety measures, and may inadvertently produce inappropriate material. Strictly no utilization for illicit, damaging, violent, racist, or explicit activities. User interactions may be archived for ongoing research. llavidal may occasionally falter and isn't ideal for precision-dependent tasks. We're persistently enhancing it. Your understanding is appreciated.
</div>
""")

css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""



title_markdown = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a  style="margin-right: 100px; text-decoration: none; display: flex; align-items: center;">
    <img src='/file=llavidal/demo/static/llavidal_logo.png'  alt="LLAVIDAL" style="max-width: 150px; height: auto;">
  </a>
  <div>
    <h1 >LLAVIDAL: Large Language Vision Models for Daily Activities of Living </h1>
  </div>
    <a  style="margin-right: 20px; text-decoration: none; display: flex; align-items: right;">
    <img src='/file=llavidal/demo/static/unc_logo.png'  alt="LLAVIDAL" style="max-width: 200px;height: auto;border-left-width: 0px;margin-left: 60px;">
  </a>
  
</div>
""")

# title = """<h1 align="center"><a href="https://www.dropbox.com/s/papppkr5737mq4l/logo_design_nb2.png?dl=1"><img src="https://www.dropbox.com/s/papppkr5737mq4l/logo_design_nb2.png?dl=1" alt="llavidal" border="0" style="margin: 0 auto; width: 30%;" /></a> </h1>
# """

description = "<h3>This is the demo of LLAVIDAL developed from CHARMLAB in UNCC.</h3> Upload your video and start chatting!"

disclaimer = """ 
            <h2>Disclaimer</h2>
            <h3>The service is a research preview from the UNCC CHARMLAB and is intended for non-commercial use only.</h3> 
            <hr> 
            <h3 align="center">Designed and Developed under UNCC CHARMLAB. We thank MBZUAI Oryx for the initial work</h3>
            """


class Seafoam(Base):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.orange,
            secondary_hue: colors.Color | str = colors.blue,
            neutral_hue: colors.Color | str = colors.gray,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_md,
            font: fonts.Font
                  | str
                  | Iterable[fonts.Font | str] = (
                    fonts.GoogleFont("Source Serif Pro"),
                    "ui-sans-serif",
                    "sans-serif",
            ),
            font_mono: fonts.Font
                       | str
                       | Iterable[fonts.Font | str] = (
                    fonts.GoogleFont("IBM Plex Mono"),
                    "ui-monospace",
                    "monospace",
            ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
