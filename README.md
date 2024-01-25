# iEACOP
An improved EACOP implementation for real-parameter single objective optimization problems
## Usage
On linux, you can compile the CEC 2017 benchmark suite using the following command:
```shell
$ cd CEC/TestSuite2017
$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) cec17_test_func.cpp -o cec17_test_func$(python3-config --extension-suffix)
```
A shell script is available at **CEC/TestSuite2017/compile.sh**.

This assumes that [pybind11](https://github.com/pybind/pybind11) has been installed using pip or conda.

For more details [check the official documentation](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually).

The script **run_cec17.py** executes iEACOP on the CEC 2017 benchmark suite.