from setuptools import setup, Extension

module1 = Extension('logistic',
                    sources = ['logistic.cpp'])

setup (name = 'Logistic Regression Model',
    version = '1.0',
    description = 'This is a Logistic Regression Model writen in C++',
    ext_modules = [module1])
