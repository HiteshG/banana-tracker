from setuptools import setup, find_packages

setup(
    name="bananatracker",
    version="0.1.0",
    description="Multi-object tracking using YOLOv8 detection + ByteTrack core",
    author="bananatracker",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "lap>=0.4.0",
        "cython_bbox>=0.1.3",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "bananatracker=bananatracker.run_tracking:main",
        ],
    },
)
