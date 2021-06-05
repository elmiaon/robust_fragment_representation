#include <Python.h>
#include <stdio.h>
#include "emd.h"

static PyObject *emdlib_emd(PyObject *self, PyObject *args){
    PyObject *weight1,*weight2,*cost_matrix;

    if (!PyArg_ParseTuple(args, "OOO", &weight1, &weight2, &cost_matrix)) {
        return NULL;
    }
    int n1 = PyObject_Length(weight1);
    int n2 = PyObject_Length(weight2);
    Py_ssize_t list_size = PyList_Size(cost_matrix);
    
    feature_t f1[n1],f2[n2];
    float w1[n1],w2[n2];
    float COST[n1][n2];

    float sum = 0;

    for(int i=0; i< n1 ;i++){
        PyFloatObject *weight = PyList_GetItem(weight1, i);
        double w_double = PyFloat_AsDouble(weight);
        float w = (float) w_double;
        w1[i] = w;
        f1[i] = i;
    }

    for(int i=0; i< n2 ;i++){
        PyFloatObject *weight = PyList_GetItem(weight2, i);
        double w_double = PyFloat_AsDouble(weight);
        float w = (float) w_double;
        w2[i] = w;
        f2[i] = i;
    }

    for (Py_ssize_t i = 0; i < list_size; i++){
        PyObject *sublist = PyList_GetItem(cost_matrix,i);
        Py_ssize_t sublist_size = PyList_Size(sublist);

        for (Py_ssize_t j = 0; j < sublist_size; j++){
            PyFloatObject *value = PyList_GetItem(sublist, j);
            double cost = PyFloat_AsDouble(value);
            float c = (float) cost;
            COST[i][j] = c;
        }
    }

    signature_t s1 = {n1,f1,w1},
                s2 = {n2,f2,w2};
    
    float e;

    e = emd(&s1, &s2, &COST, 0, 0);

    return Py_BuildValue("f", e);
}

static PyObject *emdlib_emdwithflow(PyObject *self, PyObject *args){
    PyObject *weight1,*weight2,*cost_matrix;

    if (!PyArg_ParseTuple(args, "OOO", &weight1, &weight2, &cost_matrix)) {
        return NULL;
    }
    int n1 = PyObject_Length(weight1);
    int n2 = PyObject_Length(weight2);
    Py_ssize_t list_size = PyList_Size(cost_matrix);
    
    feature_t f1[n1],f2[n2];
    float w1[n1],w2[n2];
    float COST[n1][n2];
    int flow_num = n1+n2-1;

    float sum = 0;

    for(int i=0; i< n1 ;i++){
        PyFloatObject *weight = PyList_GetItem(weight1, i);
        double w_double = PyFloat_AsDouble(weight);
        float w = (float) w_double;
        w1[i] = w;
        f1[i] = i;
    }

    for(int i=0; i< n2 ;i++){
        PyFloatObject *weight = PyList_GetItem(weight2, i);
        double w_double = PyFloat_AsDouble(weight);
        float w = (float) w_double;
        w2[i] = w;
        f2[i] = i;
    }

    for (Py_ssize_t i = 0; i < list_size; i++){
        PyObject *sublist = PyList_GetItem(cost_matrix,i);
        Py_ssize_t sublist_size = PyList_Size(sublist);

        for (Py_ssize_t j = 0; j < sublist_size; j++){
            PyFloatObject *value = PyList_GetItem(sublist, j);
            double cost = PyFloat_AsDouble(value);
            float c = (float) cost;
            COST[i][j] = c;
        }
    }

    signature_t s1 = {n1,f1,w1},
                s2 = {n2,f2,w2};
    
    flow_t flow[flow_num];

    int flow_size;
    
    float e;
    int from,to;
    float amount;
    
    int p[2];
    float q[1];

    e = emd(&s1, &s2, &COST, flow, &flow_size);

    // printf("emd=%f\n", e);
    // printf("\nflow: %d\n",flow_size);
    // printf("\t\tfrom\tto\tamount\t\tcount\n");
    // int count = 0;
    // for (Py_ssize_t i=0; i < flow_num; i++){
    //         printf ("%.6f\t", flow[i].amount);
    //         if (flow[i].amount > 0){
    //             count++;
    //             printf("%d\t%d\t%f\t%d\n", flow[i].from, flow[i].to, flow[i].amount, count);
    //         }
    //         else
    //         {
    //             printf("\n");
    //         }
    // }

    PyObject *result = PyTuple_New(flow_size+1);
    PyTuple_SET_ITEM(result, 0, Py_BuildValue("f", e));
    for (Py_ssize_t i = 1; i < flow_size+1; i++) {
        PyObject *items = PyTuple_New(3);
        PyTuple_SET_ITEM(items, 0, Py_BuildValue("i",flow[i-1].from));
        PyTuple_SET_ITEM(items, 1, Py_BuildValue("i",flow[i-1].to));
        PyTuple_SET_ITEM(items, 2, Py_BuildValue("f",flow[i-1].amount));
        PyTuple_SET_ITEM(result, i, items);
    }

    return result;
}

static PyMethodDef method[] = {
    {
        "emd",
        emdlib_emd,
        METH_VARARGS,
        "An implementation of the Earth Movers Distance."
    },
    {
        "emd_with_flow",
        emdlib_emdwithflow,
        METH_VARARGS, 
        "An implementation of the Earth Movers Distance with flow."
    },
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "emd",
    "An implementation of the Earth Movers Distance.",
    -1,
    method
};

PyMODINIT_FUNC PyInit_emd(void){
    return PyModule_Create(&module_def);
}