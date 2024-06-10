import setuptools


package_name = 'utils'


def get_version():
    with open('VERSION') as f:
        version_str = f.read()
    return str(version_str)
    # url = f'https://pypi.org/project/{package_name}/'
    # response = requests.get(url)
    # soup = BeautifulSoup(response.content, 'html.parser')
    # latest_version = soup.select_one('.release__version').text.strip()
    # return str(latest_version)


def main():
    with open('README.md', 'r', encoding='utf8') as fh:
        long_description = fh.read()
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    print(setuptools.find_packages('src'))
    print('='*100)
    setuptools.setup(
        name=package_name,
        version=get_version(),
        author='zpwang',
        author_email='zhipangwang@gmail.com',
        description='Utilities',
        long_description=long_description,
        long_description_content_type='text/markdown',
        package_dir={'': 'src'},
        packages=setuptools.find_packages('src'),
        # install_requires=required,
    )


if __name__ == '__main__':
    main()
