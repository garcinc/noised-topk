from setuptools import setup, find_packages


def index_first_occurence(substring, sentences):
    for i, sentence in enumerate(sentences):
        if substring in sentence:
            return i


with open("README.md", encoding="utf-8") as fh:
    long_description_full = fh.readlines()

start_index = index_first_occurence('paper', long_description_full)
end_index = index_first_occurence('Experiments', long_description_full) - 1

sentences_package = long_description_full[start_index:end_index]
long_description_package = ''.join(sentences_package)

setup(name='pytopk',
      description='Pytorch implementation of a differentiable topk function, a balanced and imbalanced top-k loss for deep learning',
      long_description=long_description_package,
      long_description_content_type="text/markdown",
      author='Camille Garcin',
      packages=find_packages(),
      license='MIT',
      url='https://github.com/garcinc/noised-topk',
      version="0.1.0",
      install_requires=["torch>=1.0",
                        "numpy"],
      keywords=['topk', 'deep learning', 'pytorch', 'differentiable']
      )