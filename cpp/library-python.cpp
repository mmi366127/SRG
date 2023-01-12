#include "RBTree.h"

extern "C" {


RBTree *T = nullptr;

void getObject(int n, double x) {
    if(T == nullptr) {
        T = new RBTree(n, x);    
        // printf("get object...\n");
    }
    return;
}

void delObject() {
    // printf("del object...\n");
    delete T;
    T = nullptr;
}

void insertObject(double val, int idx) {
    // printf("insert %d %f\n", idx, val);
    T->insert(val, idx);
}

void updateObject(double val, int idx) {
    // printf("update %d %f\n", idx, val);
    T->update(idx, val);
}

int sample() {
    // printf("sample...\n");
    return T->sample();
}

double getWeight(int idx) {
    return T->getWeight(idx);
}

}  // extern "C"