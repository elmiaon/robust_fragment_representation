from setuptools import Extension, setup


module = Extension("emd",sources=['emdmodule.c','emd.c'])

setup(
    name="emd",
    version = "1.0",
    description="An implementation of the Earth Movers Distance.",
    ext_modules=[module]
)