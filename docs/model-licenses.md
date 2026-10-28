# Model and dependency licenses

Historical inventory supplied with the source prototype, dated 11 September 2026.
This publication does not certify the current license status of upstream assets. Package licenses and
model-weight licenses are separate. Preserve the applicable license and notice
files when redistributing. This POC does not establish commercial clearance for
every model asset or training image.

| Component | Upstream license / commercial-use condition |
| --- | --- |
| [YOLO face weights](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/blob/main/README.md) | Model card declares AGPL-3.0. Commercial use is subject to its terms; this is not a proprietary-use permission. |
| [Ultralytics](https://www.ultralytics.com/license) | AGPL-3.0 or an applicable Enterprise agreement. Confirm separately whether any agreement covers these third-party face weights. |
| [AuraFace pack](https://huggingface.co/fal/AuraFace-v1/blob/main/README.md) | Model card declares Apache-2.0 and commercial intent. The pack also contains auxiliary ONNX assets; per-file provenance is not established by this review. Commercial clearance of those assets remains open. |
| [InsightFace code](https://github.com/deepinsight/insightface#license) | MIT code permits commercial use; the project's own pretrained models have separate noncommercial research restrictions. The POC requests the AuraFace pack, not the default InsightFace pack. |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | MIT |
| [OpenCV Python](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt) | MIT packaging; OpenCV uses Apache-2.0. Wheels include third-party components with their own notices/licenses. |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause; bundled components have additional notices |
| [Gradio](https://github.com/gradio-app/gradio/blob/main/LICENSE) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) / [Torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) | BSD-style licenses with third-party notices |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org/stable/project/license.html) | Matplotlib/PSF-based permissive license; retain notices |
| [Python](https://docs.python.org/3/license.html) | PSF license and bundled third-party notices |
| Debian base and apt libraries | Multiple licenses; inspect `/usr/share/doc/*/copyright` in the built image |

Permissive licenses generally allow commercial use subject to their terms;
they do not license personal data or waive model/data restrictions. The package
inventory in the submission evidence records 91 installed versions and declared
license metadata. `ultralytics-thop` is also AGPL-3.0-or-later. The installed
InsightFace and setuptools metadata omit a license; InsightFace is checked
against its upstream policy above, while setuptools remains a metadata gap. Missing/ambiguous metadata requires upstream review and is
not treated as permission. Dependencies introduced by future rebuilds must be
checked again.

The [AT&T/ORL dataset source](https://cam-orl.co.uk/facedatabase.html/) asks users
to credit AT&T Laboratories Cambridge. Images are downloaded locally for this
research evaluation and are not redistributed in Git. This dataset is not a
commercial identity-document validation dataset.
