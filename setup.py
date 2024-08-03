from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='document_search_engine',
    version='1.0',
    author='ywchoi',
    author_email='chldyddnjs@naver.com',
    #long_description=read('README.md'),
    python_requires='>=3.10',
    #install_requires=['numpy'],
    #package_data={'mypkg': ['*/requirements.txt']},
    #dependency_links = [], ## 최신 패키지를 설치하는 경우 
    description='upstage hackaton for legal AI service ',
    packages=find_packages(include=['upstage_hackaton'])
)