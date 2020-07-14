Welcome to epic-masks's documentation!
=======================================================

.. toctree::
   :maxdepth: 5
   :caption: Contents:



Installation
------------

.. code-block:: bash

    $ pip install git+https://github.com/epic-kitchens/epic-kitchens-100-object-masks.git

Usage
-----

Visualise the masks like so:

.. code-block:: python

    from typing import Union
    from pathlib import Path
    import PIL.Image
    from epic_kitchens.masks.io import load_detections
    from epic_kitchens.masks.visualisation import DetectionRenderer

    class LazyFrameLoader:
        def __init__(self, path: Union[Path, str], frame_template: str = 'frame_{:010d}.jpg'):
            self.path = Path(path)
            self.frame_template = frame_template

        def __getitem__(self, idx: int) -> PIL.Image.Image:
            return PIL.Image.open(str(self.path / self.frame_template.format(idx + 1)))

    detections = load_detections('detections/P01_101.pkl')
    frames = LazyFrameLoader('frames/P01_101')
    renderer = DetectionRenderer()

    frame_idx = 100
    renderer.render_detections(frames[frame_idx], detections[frame_idx])

A Jupyter notebook example is included that demonstrates how to load
detections and visualise them.


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
