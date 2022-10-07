from setuptools import setup

'''with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()'''

## edit below variables as per your requirements -
REPO_NAME = "Loan_Prediction"
AUTHOR_USER_NAME = "ravikanur"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = []


setup(
    name=REPO_NAME,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="An ML Algoruthm for Loan Prediction",
    long_description="An ML Algoruthm for Loan Prediction",
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="ravikanur@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)
