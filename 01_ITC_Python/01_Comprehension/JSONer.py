"""בס׳׳ד
This is my implementation of the current exercise
Author: Jeremy Bensoussan
"""

TEST_URL = r'https://www.welovepython.com/onelinerexercise?python=-3.56%2C-2.55&is=-2.75%2C-4.92&awesome=-4.63%2C4.12' \
           r'&I=-5.75%2C-3.45&am=-1.0%2C-8.31&telling=-7.38%2C-2.41&you=-3.5%2C-3.6&but=-4.38%2C-2.92&you=-7.13%2C-6.' \
           r'0&can=1.63%2C-2.36&probably=-5.75%2C-6.51&see=-3.25%2C-3.64&that=1.75%2C-2.15&for=-3.75%2C5.08&name=your' \
           r'self&ec=-4.88&soc=-2.05'


def jsoner(url_to_convert):
    """performs the required calculation in just one line and returns a string"""
    return ',\n'.join([str(line_dict) for line_dict in [{url_parameter.split('=')[0]: {'interesting': url_parameter.split('=')[1].split('%2C')[0], 'value': url_parameter.split('=')[1].split('%2C')[1]}} for url_parameter in url_to_convert[url_to_convert.find('?')+1:].replace('name=', '').replace('&ec', '').replace('&soc=', '%2C').split('&')]])


def main():
    """Prints the result of the one liner"""
    print(jsoner(TEST_URL))


if __name__ == '__main__':
    main()
