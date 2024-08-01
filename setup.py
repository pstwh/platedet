from setuptools import setup

setup(
    name="platedet",
    version="0.0.3",
    packages=["platedet"],
    include_package_data=True,
    url="https://github.com/pstwh/platedet",
    keywords="plate, detection, brazilian",
    package_data={"platedet": ["artifacts/*.onnx"]},
    python_requires=">=3.5, <4",
    install_requires=["pillow==10.4.0", "opencv-python-headless==4.10.0.84"],
    extras_require={
        'cpu': [
            'onnxruntime==1.18.1',
        ],
        "gpu": [
            "onnxruntime-gpu==1.18.1",
        ],
    },
    entry_points={
        "console_scripts": ["platedet=platedet.cli:main"],
    },
    description="Detect license plates",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
